"""
MIP optimisation. Solves the same trade problem twice, once with OR-Tools
CP-SAT (primary) and once with PuLP+CBC (comparison check).

Inputs:
  - Two candidate pools (from_a goes to B, from_b goes to A)
  - A per-player valuation in the receiving team's context (from
    valuation_model.py)
  - The forced_out set from the SAT layer (player IDs we must not trade)

Decision variables:
  x[p] in {0, 1}, 1 if p is in the final trade.

Objective (maximise total net value):
  for each traded player, gain = (receiving team's valuation) - (sending
  team's valuation). Same player has different value to different teams,
  which is what makes the swap positive-sum.

Constraints:
  C1. x[p] = 0 for any p in forced_out (SAT preprocessing).
  C2. Salary matching per team: incoming <= outgoing * 1.25 + $100K.
  C3. Hard cap per team: post-trade total <= $165M (when toggle is on).
  C4. At least one player moves each direction (rule out the empty trade).

CP-SAT is integer-only, so we scale salaries to $1K units and valuations
by 1000 to keep precision. PuLP/CBC takes floats directly. After solving,
CP-SAT's objective is divided by 1000 to compare against CBC's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from constraints_config import ConstraintsConfig
from data_fetcher import PlayerRecord
from sat_layer import SATResult

# ── OR-Tools ──────────────────────────────────────────────────────────────────
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("[mip_layer] ortools not installed; OR-Tools solver unavailable.")

# ── PuLP ──────────────────────────────────────────────────────────────────────
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("[mip_layer] pulp not installed; PuLP/CBC solver unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MIPResult:
    """
    What one MIP solve returns. solver_name is "OR-Tools CP-SAT" or
    "PuLP CBC". status mirrors the solver's own status string ("OPTIMAL",
    "FEASIBLE", "INFEASIBLE", etc.) and `optimal` is True only when the
    solver proved optimality. objective_value is the net trade valuation
    gain. The four salary fields are totals in USD (not scaled).
    solve_time_ms is wall-clock for the solve call only.
    """
    solver_name: str
    status: str
    optimal: bool
    objective_value: float
    players_traded_from_a: list[PlayerRecord]
    players_traded_from_b: list[PlayerRecord]
    salary_out_a: float
    salary_in_a: float
    salary_out_b: float
    salary_in_b: float
    solve_time_ms: float

    def display(self) -> str:
        """Pretty-print the result."""
        sep = "─" * 60
        lines = [
            sep,
            f"Solver        : {self.solver_name}",
            f"Status        : {self.status}",
            f"Objective     : {self.objective_value:+.4f}  (net valuation gain)",
            sep,
            "Players from A → B:",
        ]
        if self.players_traded_from_a:
            for p in self.players_traded_from_a:
                lines.append(f"  {p.name:<24} ${p.salary/1e6:.2f}M  val={p.valuation:+.3f}")
        else:
            lines.append("  (none)")

        lines.append("Players from B → A:")
        if self.players_traded_from_b:
            for p in self.players_traded_from_b:
                lines.append(f"  {p.name:<24} ${p.salary/1e6:.2f}M  val={p.valuation:+.3f}")
        else:
            lines.append("  (none)")

        lines += [
            sep,
            f"Team A salary: out=${self.salary_out_a/1e6:.2f}M  in=${self.salary_in_a/1e6:.2f}M",
            f"Team B salary: out=${self.salary_out_b/1e6:.2f}M  in=${self.salary_in_b/1e6:.2f}M",
            f"Solve time    : {self.solve_time_ms:.1f} ms",
            sep,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: scale floats to int for CP-SAT
# ─────────────────────────────────────────────────────────────────────────────

_SAL_SCALE = 1_000       # $1K units  ($165M → 165,000)
_VAL_SCALE = 1_000       # 3 decimal places of precision for valuations


def _to_int_sal(dollars: float) -> int:
    """Convert dollar amount to integer thousands (rounding)."""
    return int(round(dollars / _SAL_SCALE))


def _to_int_val(val: float) -> int:
    """Convert valuation float to integer (× 1000, rounding)."""
    return int(round(val * _VAL_SCALE))


# ─────────────────────────────────────────────────────────────────────────────
# OR-Tools CP-SAT solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_ortools(
    candidates_from_a: list[PlayerRecord],  # going to B
    candidates_from_b: list[PlayerRecord],  # going to A
    roster_a: list[PlayerRecord],
    roster_b: list[PlayerRecord],
    sat_result: SATResult,
    config: ConstraintsConfig,
    team_a: str = "A",
    team_b: str = "B",
    time_limit_s: float = 30.0,
) -> MIPResult:
    """
    Solve the trade optimisation with OR-Tools CP-SAT.

    All integer arithmetic; salaries in $1K units, valuations times 1000.
    """
    import time as _time

    if not ORTOOLS_AVAILABLE:
        return MIPResult(
            solver_name="OR-Tools CP-SAT",
            status="UNAVAILABLE",
            optimal=False,
            objective_value=0.0,
            players_traded_from_a=[],
            players_traded_from_b=[],
            salary_out_a=0, salary_in_a=0, salary_out_b=0, salary_in_b=0,
            solve_time_ms=0,
        )

    model = cp_model.CpModel()

    # ── Decision variables ────────────────────────────────────────────────
    # xa[i] = 1 if candidates_from_a[i] is included in the trade (goes to B)
    # xb[j] = 1 if candidates_from_b[j] is included in the trade (goes to A)
    xa = [model.NewBoolVar(f"xa_{p.player_id}") for p in candidates_from_a]
    xb = [model.NewBoolVar(f"xb_{p.player_id}") for p in candidates_from_b]

    # ── C1: SAT-fixed forced-out players ─────────────────────────────────
    # Any player whose player_id is in sat_result.forced_out must NOT trade.
    for i, p in enumerate(candidates_from_a):
        if p.player_id in sat_result.forced_out:
            model.Add(xa[i] == 0)

    for j, p in enumerate(candidates_from_b):
        if p.player_id in sat_result.forced_out:
            model.Add(xb[j] == 0)

    # ── C2: Salary matching per team ──────────────────────────────────────
    # For Team A:
    #   outgoing  = Σ salary(p) * xa[i]      (A sends these players to B)
    #   incoming  = Σ salary(p) * xb[j]      (A receives these players from B)
    #   incoming ≤ outgoing * threshold + bonus
    #
    # In integer $1K units (multiply both sides by 4 to clear the 1.25 = 5/4
    # fraction and avoid floating point):
    #   4 * incoming_k ≤ 5 * outgoing_k + 4 * bonus_k
    #   ↔  4 * Σ sal_k(xb) ≤ 5 * Σ sal_k(xa) + 400    (bonus = $100K = 100 k-units)

    sal_a = [_to_int_sal(p.salary) for p in candidates_from_a]
    sal_b = [_to_int_sal(p.salary) for p in candidates_from_b]
    bonus_k = _to_int_sal(config.salary_matching_bonus)

    # Threshold in fractional form: threshold = p/q
    # 1.25 → 5/4   so  incoming ≤ outgoing * 5/4 + bonus
    # Multiply by 4: 4*incoming ≤ 5*outgoing + 4*bonus
    thr_num, thr_den = 5, 4   # hard-coded for 125 %; adjust if threshold changes

    if config.salary_matching_threshold != 1.25:
        # General case: approximate as nearest rational with denominator ≤ 100
        import fractions
        f = fractions.Fraction(config.salary_matching_threshold).limit_denominator(100)
        thr_num, thr_den = f.numerator, f.denominator

    # Team A: incoming (from B) ≤ outgoing (from A) × threshold + bonus
    incoming_a = model.NewIntVar(0, 10**9, "incoming_a")
    outgoing_a = model.NewIntVar(0, 10**9, "outgoing_a")
    model.Add(incoming_a == sum(sal_b[j] * xb[j] for j in range(len(xb))))
    model.Add(outgoing_a == sum(sal_a[i] * xa[i] for i in range(len(xa))))
    # thr_den * incoming_a ≤ thr_num * outgoing_a + thr_den * bonus_k
    model.Add(thr_den * incoming_a <= thr_num * outgoing_a + thr_den * bonus_k)

    # Team B: incoming (from A) ≤ outgoing (from B) × threshold + bonus
    incoming_b = model.NewIntVar(0, 10**9, "incoming_b")
    outgoing_b = model.NewIntVar(0, 10**9, "outgoing_b")
    model.Add(incoming_b == sum(sal_a[i] * xa[i] for i in range(len(xa))))
    model.Add(outgoing_b == sum(sal_b[j] * xb[j] for j in range(len(xb))))
    model.Add(thr_den * incoming_b <= thr_num * outgoing_b + thr_den * bonus_k)

    # ── C3: Hard cap ──────────────────────────────────────────────────────
    if config.enforce_hard_cap:
        hard_cap_k = _to_int_sal(config.hard_cap_threshold)

        # Team A post-trade salary = (current total) - outgoing + incoming
        current_sal_a_k = _to_int_sal(sum(p.salary for p in roster_a))
        post_sal_a = model.NewIntVar(0, 10**9, "post_sal_a")
        model.Add(post_sal_a == current_sal_a_k - outgoing_a + incoming_a)
        model.Add(post_sal_a <= hard_cap_k)

        # Team B post-trade salary
        current_sal_b_k = _to_int_sal(sum(p.salary for p in roster_b))
        post_sal_b = model.NewIntVar(0, 10**9, "post_sal_b")
        model.Add(post_sal_b == current_sal_b_k - outgoing_b + incoming_b)
        model.Add(post_sal_b <= hard_cap_k)

    # ── C4: Non-trivial trade (at least 1 player each direction) ─────────
    model.Add(sum(xa) >= 1)
    model.Add(sum(xb) >= 1)

    # ── Objective ─────────────────────────────────────────────────────────
    # Maximise total net value:
    #   A gains val_for_a of each player received (from B)
    #   B gains val_for_b of each player received (from A)
    #   A loses val_for_a of each player sent (from A, but those valuations
    #     are from B's context, not A's; we use each player's stored .valuation
    #     which was set in the context of their RECEIVING team by main.py)
    #
    # So: obj = Σ val_recv(xb) + Σ val_recv(xa)
    #         - Σ val_send(xa) - Σ val_send(xb)
    # where val_recv is the valuation from the receiving team's context and
    # val_send is the valuation from the sending team's context.
    # Since main.py sets p.valuation = receiving-team valuation already,
    # we sum them directly:

    val_a_int = [_to_int_val(p.valuation) for p in candidates_from_b]  # A receives B's players
    val_b_int = [_to_int_val(p.valuation) for p in candidates_from_a]  # B receives A's players

    obj = (
        sum(val_a_int[j] * xb[j] for j in range(len(xb)))
        + sum(val_b_int[i] * xa[i] for i in range(len(xa)))
    )
    model.Maximize(obj)

    # ── Solve ─────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.log_search_progress = False   # suppress verbose output

    t0 = _time.perf_counter()
    status_code = solver.Solve(model)
    elapsed_ms = (_time.perf_counter() - t0) * 1000

    # ── Decode solution ───────────────────────────────────────────────────
    status_map = {
        cp_model.OPTIMAL:   ("OPTIMAL",   True),
        cp_model.FEASIBLE:  ("FEASIBLE",  False),
        cp_model.INFEASIBLE:("INFEASIBLE",False),
        cp_model.UNKNOWN:   ("UNKNOWN",   False),
        cp_model.MODEL_INVALID: ("MODEL_INVALID", False),
    }
    status_str, is_optimal = status_map.get(status_code, ("UNKNOWN", False))

    traded_a: list[PlayerRecord] = []
    traded_b: list[PlayerRecord] = []
    obj_val = 0.0

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i, p in enumerate(candidates_from_a):
            if solver.Value(xa[i]) == 1:
                traded_a.append(p)
        for j, p in enumerate(candidates_from_b):
            if solver.Value(xb[j]) == 1:
                traded_b.append(p)
        obj_val = solver.ObjectiveValue() / _VAL_SCALE

    sal_out_a = sum(p.salary for p in traded_a)
    sal_in_a  = sum(p.salary for p in traded_b)
    sal_out_b = sum(p.salary for p in traded_b)
    sal_in_b  = sum(p.salary for p in traded_a)

    return MIPResult(
        solver_name="OR-Tools CP-SAT",
        status=status_str,
        optimal=is_optimal,
        objective_value=obj_val,
        players_traded_from_a=traded_a,
        players_traded_from_b=traded_b,
        salary_out_a=sal_out_a,
        salary_in_a=sal_in_a,
        salary_out_b=sal_out_b,
        salary_in_b=sal_in_b,
        solve_time_ms=elapsed_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PuLP + CBC solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_pulp(
    candidates_from_a: list[PlayerRecord],
    candidates_from_b: list[PlayerRecord],
    roster_a: list[PlayerRecord],
    roster_b: list[PlayerRecord],
    sat_result: SATResult,
    config: ConstraintsConfig,
    team_a: str = "A",
    team_b: str = "B",
    time_limit_s: float = 30.0,
) -> MIPResult:
    """
    Solve the same trade optimisation with PuLP + CBC.

    PuLP accepts float coefficients directly, so no integer scaling is needed.
    This solver serves as a comparison / validation for the CP-SAT solution.
    """
    import time as _time

    if not PULP_AVAILABLE:
        return MIPResult(
            solver_name="PuLP CBC",
            status="UNAVAILABLE",
            optimal=False,
            objective_value=0.0,
            players_traded_from_a=[],
            players_traded_from_b=[],
            salary_out_a=0, salary_in_a=0, salary_out_b=0, salary_in_b=0,
            solve_time_ms=0,
        )

    prob = pulp.LpProblem("NBA_Trade_Optimizer", pulp.LpMaximize)

    # ── Decision variables ────────────────────────────────────────────────
    xa = {p.player_id: pulp.LpVariable(f"xa_{p.player_id}", cat="Binary")
          for p in candidates_from_a}
    xb = {p.player_id: pulp.LpVariable(f"xb_{p.player_id}", cat="Binary")
          for p in candidates_from_b}

    # ── C1: SAT-fixed ─────────────────────────────────────────────────────
    for p in candidates_from_a:
        if p.player_id in sat_result.forced_out:
            prob += (xa[p.player_id] == 0, f"sat_lock_a_{p.player_id}")
    for p in candidates_from_b:
        if p.player_id in sat_result.forced_out:
            prob += (xb[p.player_id] == 0, f"sat_lock_b_{p.player_id}")

    # ── C2: Salary matching ───────────────────────────────────────────────
    threshold = config.salary_matching_threshold
    bonus     = config.salary_matching_bonus

    out_a = pulp.lpSum(p.salary * xa[p.player_id] for p in candidates_from_a)
    in_a  = pulp.lpSum(p.salary * xb[p.player_id] for p in candidates_from_b)
    out_b = pulp.lpSum(p.salary * xb[p.player_id] for p in candidates_from_b)
    in_b  = pulp.lpSum(p.salary * xa[p.player_id] for p in candidates_from_a)

    prob += (in_a  <= threshold * out_a + bonus,  "salary_match_A")
    prob += (in_b  <= threshold * out_b + bonus,  "salary_match_B")

    # ── C3: Hard cap ──────────────────────────────────────────────────────
    if config.enforce_hard_cap:
        cap = config.hard_cap_threshold
        cur_a = sum(p.salary for p in roster_a)
        cur_b = sum(p.salary for p in roster_b)
        prob += (cur_a - out_a + in_a <= cap, "hard_cap_A")
        prob += (cur_b - out_b + in_b <= cap, "hard_cap_B")

    # ── C4: Non-trivial ───────────────────────────────────────────────────
    prob += (pulp.lpSum(xa.values()) >= 1, "min_trade_a")
    prob += (pulp.lpSum(xb.values()) >= 1, "min_trade_b")

    # ── Objective ─────────────────────────────────────────────────────────
    # A gains: val of players received from B (in A's context)
    # B gains: val of players received from A (in B's context)
    val_gain_a = pulp.lpSum(p.valuation * xb[p.player_id] for p in candidates_from_b)
    val_gain_b = pulp.lpSum(p.valuation * xa[p.player_id] for p in candidates_from_a)
    prob += val_gain_a + val_gain_b

    # ── Solve ─────────────────────────────────────────────────────────────
    solver_obj = pulp.PULP_CBC_CMD(
        msg=0,
        timeLimit=time_limit_s,
    )
    t0 = _time.perf_counter()
    prob.solve(solver_obj)
    elapsed_ms = (_time.perf_counter() - t0) * 1000

    status_str = pulp.LpStatus[prob.status]
    is_optimal = (prob.status == 1)   # 1 = Optimal in PuLP

    traded_a: list[PlayerRecord] = []
    traded_b: list[PlayerRecord] = []
    obj_val = 0.0

    if is_optimal or prob.status == 1:
        for p in candidates_from_a:
            if pulp.value(xa[p.player_id]) and pulp.value(xa[p.player_id]) > 0.5:
                traded_a.append(p)
        for p in candidates_from_b:
            if pulp.value(xb[p.player_id]) and pulp.value(xb[p.player_id]) > 0.5:
                traded_b.append(p)
        obj_val = pulp.value(prob.objective) or 0.0

    sal_out_a = sum(p.salary for p in traded_a)
    sal_in_a  = sum(p.salary for p in traded_b)
    sal_out_b = sum(p.salary for p in traded_b)
    sal_in_b  = sum(p.salary for p in traded_a)

    return MIPResult(
        solver_name="PuLP CBC",
        status=status_str,
        optimal=is_optimal,
        objective_value=obj_val,
        players_traded_from_a=traded_a,
        players_traded_from_b=traded_b,
        salary_out_a=sal_out_a,
        salary_in_a=sal_in_a,
        salary_out_b=sal_out_b,
        salary_in_b=sal_in_b,
        solve_time_ms=elapsed_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def solve_both(
    candidates_from_a: list[PlayerRecord],
    candidates_from_b: list[PlayerRecord],
    roster_a: list[PlayerRecord],
    roster_b: list[PlayerRecord],
    sat_result: SATResult,
    config: ConstraintsConfig,
    team_a: str = "A",
    team_b: str = "B",
    time_limit_s: float = 30.0,
) -> tuple[MIPResult, MIPResult]:
    """
    Run both solvers and return (ortools_result, pulp_result).
    """
    kwargs = dict(
        candidates_from_a=candidates_from_a,
        candidates_from_b=candidates_from_b,
        roster_a=roster_a,
        roster_b=roster_b,
        sat_result=sat_result,
        config=config,
        team_a=team_a,
        team_b=team_b,
        time_limit_s=time_limit_s,
    )
    ortools_result = solve_ortools(**kwargs)
    pulp_result    = solve_pulp(**kwargs)
    return ortools_result, pulp_result
