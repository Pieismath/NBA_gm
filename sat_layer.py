"""
sat_layer.py
------------
SAT feasibility layer using PicoSAT (via the python-sat / pysat library).

Responsibility
--------------
Encode NBA CBA *boolean* trade constraints as a propositional SAT formula and
check satisfiability.  If the formula is satisfiable the model returns a dict
of variable assignments (True/False per player) that the MIP layer will treat
as fixed inputs.

Three constraint families are encoded here:
  1. Roster-size limits  – each team must have 13–15 players after the trade.
  2. No-trade clauses    – any player with has_ntc=True cannot be traded.
  3. Recently-signed     – players signed < 12 months ago cannot be traded.

Salary matching is intentionally NOT encoded here; it is handled as a linear
constraint in the MIP layer (mip_layer.py).

SAT Encoding Overview
---------------------
For a 2-team trade (Team A ↔ Team B) with candidate player pools:

  Variable  traded[p]  ∈ {0, 1}
    = 1  if player p is included in the trade (moves to the other team)
    = 0  if player p stays on their current team

Constraints:
  • NTC / recently-signed:  ¬traded[p]   (unit clause → traded[p] = False)
  • Roster size for Team A:
        |roster_A| - |{p ∈ A : traded[p]}| + |{p ∈ B : traded[p]}|  ∈ [13, 15]
    Encoded as two PseudoBoolean / cardinality constraints via CardEnc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from constraints_config import ConstraintsConfig
from data_fetcher import PlayerRecord

# ── PySAT imports (python-sat) ────────────────────────────────────────────────
try:
    from pysat.solvers import Solver          # wraps PicoSAT and others
    from pysat.card import CardEnc, EncType  # cardinality constraint encoder
    from pysat.formula import CNF
    PYSAT_AVAILABLE = True
except ImportError:
    PYSAT_AVAILABLE = False
    print("[sat_layer] python-sat not installed – SAT layer will use a "
          "lightweight fallback that checks constraints procedurally.")


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

class SATResult:
    """
    Returned by SATFeasibilityChecker.check().

    Attributes
    ----------
    feasible : bool
        True iff the SAT formula is satisfiable.
    forced_out : set[int]
        Player IDs that the SAT solver has forced to traded=False.
        (NTC, recently-signed, or roster constraints.)
    model : dict[int, bool]
        Full variable assignment player_id → traded (True/False).
    violations : list[str]
        Human-readable explanation of any infeasibility.
    """

    def __init__(
        self,
        feasible: bool,
        forced_out: set[int],
        model: dict[int, bool],
        violations: list[str],
    ):
        self.feasible = feasible
        self.forced_out = forced_out
        self.model = model
        self.violations = violations

    def __repr__(self) -> str:
        status = "FEASIBLE" if self.feasible else "INFEASIBLE"
        return (
            f"<SATResult {status} | forced_out={len(self.forced_out)} players"
            f" | violations={self.violations}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main checker class
# ─────────────────────────────────────────────────────────────────────────────

class SATFeasibilityChecker:
    """
    Encodes a proposed trade as a CNF formula and solves it with PicoSAT.

    Usage
    -----
    checker = SATFeasibilityChecker(config)
    result  = checker.check(
        roster_a, roster_b,
        candidates_from_a, candidates_from_b
    )
    """

    def __init__(self, config: ConstraintsConfig):
        self.config = config

    # ── Public entry point ────────────────────────────────────────────────

    def check(
        self,
        roster_a: list[PlayerRecord],
        roster_b: list[PlayerRecord],
        candidates_from_a: list[PlayerRecord],
        candidates_from_b: list[PlayerRecord],
    ) -> SATResult:
        """
        Check SAT feasibility of a proposed trade.

        Parameters
        ----------
        roster_a / roster_b
            Complete current rosters (including the candidate players).
        candidates_from_a
            Players on Team A that *might* be included in the trade (going to B).
        candidates_from_b
            Players on Team B that *might* be included in the trade (going to A).

        Returns
        -------
        SATResult
        """
        if PYSAT_AVAILABLE:
            return self._check_pysat(roster_a, roster_b,
                                     candidates_from_a, candidates_from_b)
        else:
            return self._check_fallback(roster_a, roster_b,
                                        candidates_from_a, candidates_from_b)

    # ── PySAT / PicoSAT path ─────────────────────────────────────────────

    def _check_pysat(
        self,
        roster_a: list[PlayerRecord],
        roster_b: list[PlayerRecord],
        candidates_from_a: list[PlayerRecord],
        candidates_from_b: list[PlayerRecord],
    ) -> SATResult:
        """Full SAT encoding using PicoSAT via python-sat."""

        all_candidates = candidates_from_a + candidates_from_b
        violations: list[str] = []

        # ── Step 1: assign a unique integer SAT variable to each candidate ──
        # pysat uses positive integers as variable IDs (1-indexed).
        var: dict[int, int] = {}          # player_id → SAT var id
        next_var = [1]

        def new_var() -> int:
            v = next_var[0]
            next_var[0] += 1
            return v

        for p in all_candidates:
            var[p.player_id] = new_var()

        # ── Step 2: build the CNF formula ──────────────────────────────────
        cnf = CNF()
        forced_out: set[int] = set()

        # 2a. No-trade clause: ¬traded[p]  (unit clause, literal = -var[p])
        if self.config.enforce_no_trade_clauses:
            for p in all_candidates:
                if p.has_ntc:
                    cnf.append([-var[p.player_id]])          # force traded = False
                    forced_out.add(p.player_id)
                    violations.append(
                        f"NTC: {p.name} has a no-trade clause – cannot be traded."
                    )

        # 2b. Recently-signed rule: ¬traded[p] for players signed < threshold
        if self.config.enforce_recently_signed:
            for p in all_candidates:
                if p.is_recently_signed(self.config.recently_signed_months):
                    if p.player_id not in forced_out:
                        cnf.append([-var[p.player_id]])
                        forced_out.add(p.player_id)
                        violations.append(
                            f"RECENTLY SIGNED: {p.name} signed "
                            f"{p.months_since_signing} months ago – cannot be traded."
                        )

        # 2c. Roster-size cardinality constraints ──────────────────────────
        # After the trade:
        #   roster_a_size = |roster_a| - n_traded_from_a + n_traded_from_b
        #   roster_b_size = |roster_b| - n_traded_from_b + n_traded_from_a
        #
        # Let xa_i = traded[p_i] for p_i ∈ candidates_from_a
        #     xb_j = traded[p_j] for p_j ∈ candidates_from_b
        #
        # Constraint (Team A in [min, max]):
        #   min ≤ |roster_a| - Σxa + Σxb ≤ max
        #   → Σxa - Σxb ≤ |roster_a| - min        ... (upper bound on net outflow)
        #   → Σxb - Σxa ≤ max - |roster_a|         ... (upper bound on net inflow)
        #
        # These are pseudo-boolean constraints.  We encode them as cardinality
        # constraints using CardEnc.atmost on auxiliary literal sets.

        if self.config.min_roster_size > 0:  # always True, but guard for clarity
            lits_a = [var[p.player_id] for p in candidates_from_a]
            lits_b = [var[p.player_id] for p in candidates_from_b]

            n_a = len(roster_a)
            n_b = len(roster_b)
            lo = self.config.min_roster_size
            hi = self.config.max_roster_size

            top_id = next_var[0] - 1  # highest var used so far

            # Helper: encode Σlits ≤ bound, append resulting clauses to cnf
            def add_atmost(lits: list[int], bound: int, label: str):
                nonlocal top_id
                if not lits:
                    return
                if bound < 0:
                    # No solution possible
                    cnf.append([])   # empty clause = always False
                    violations.append(label)
                    return
                enc = CardEnc.atmost(
                    lits=lits,
                    bound=bound,
                    top_id=top_id,
                    encoding=EncType.seqcounter,
                )
                for clause in enc.clauses:
                    cnf.append(clause)
                top_id = enc.nv  # update top variable counter

            # Team A roster bounds:
            #   final_A = n_a - Σxa + Σxb   must be in [lo, hi]
            #   Σxa ≤ n_a - lo               (can't trade away so many A players
            #                                  that roster falls below min)
            #   Σxb ≤ hi - n_a + Σxa        (can't receive so many B players
            #                                  that roster exceeds max)
            # The second constraint is dynamic (depends on Σxa), so we approximate
            # with the worst case: Σxb ≤ hi - n_a + len(candidates_from_a)
            add_atmost(
                lits_a,
                n_a - lo,
                f"ROSTER: Sending too many players would drop {roster_a[0].team if roster_a else 'TeamA'} below {lo}.",
            )
            add_atmost(
                lits_b,
                hi - n_b + len(candidates_from_b),
                f"ROSTER: Receiving too many players would push {roster_b[0].team if roster_b else 'TeamB'} above {hi}.",
            )

            # Team B roster bounds (symmetric):
            add_atmost(
                lits_b,
                n_b - lo,
                f"ROSTER: Sending too many players would drop {roster_b[0].team if roster_b else 'TeamB'} below {lo}.",
            )
            add_atmost(
                lits_a,
                hi - n_a + len(candidates_from_a),
                f"ROSTER: Receiving too many players would push {roster_a[0].team if roster_a else 'TeamA'} above {hi}.",
            )

            # Update next_var counter so it stays consistent
            next_var[0] = top_id + 1

        # ── Step 3: solve with a DPLL-family SAT solver ───────────────────
        # The ideal backend is PicoSAT, but some python-sat builds omit it.
        # We try 'picosat' first, then fall back to 'minisat22' (same DPLL
        # family, identical interface, equivalent performance at this scale).
        for solver_name in ("picosat", "minisat22", "cadical103"):
            try:
                _test = Solver(name=solver_name)
                _test.delete()
                break
            except Exception:
                solver_name = None
        if solver_name is None:
            solver_name = "minisat22"   # last-resort default

        with Solver(name=solver_name, bootstrap_with=cnf) as solver:
            sat = solver.solve()
            raw_model = solver.get_model() if sat else None

        # ── Step 4: decode variable assignments ───────────────────────────
        model_dict: dict[int, bool] = {}
        if raw_model is not None:
            # raw_model is a list of signed integers; positive = True, negative = False
            lit_set = set(raw_model)
            for p in all_candidates:
                v = var[p.player_id]
                model_dict[p.player_id] = (v in lit_set)
        else:
            # Infeasible: mark every forced_out player as False, rest undefined
            for p in all_candidates:
                model_dict[p.player_id] = p.player_id not in forced_out

        feasible = sat
        # Even if SAT is technically satisfiable, report key violations for
        # the user so they understand which players are locked out.
        if not sat:
            violations.append("SAT formula is UNSATISFIABLE – trade is infeasible.")

        return SATResult(
            feasible=feasible,
            forced_out=forced_out,
            model=model_dict,
            violations=violations,
        )

    # ── Pure-Python fallback (no pysat) ───────────────────────────────────

    def _check_fallback(
        self,
        roster_a: list[PlayerRecord],
        roster_b: list[PlayerRecord],
        candidates_from_a: list[PlayerRecord],
        candidates_from_b: list[PlayerRecord],
    ) -> SATResult:
        """
        Lightweight procedural feasibility check used when python-sat is not
        installed.  Checks the same three constraint families without actual SAT
        solving (just direct evaluation).
        """
        violations: list[str] = []
        forced_out: set[int] = set()
        model: dict[int, bool] = {}

        all_candidates = candidates_from_a + candidates_from_b

        # Initialise every player as tradeable
        for p in all_candidates:
            model[p.player_id] = True

        # NTC check
        if self.config.enforce_no_trade_clauses:
            for p in all_candidates:
                if p.has_ntc:
                    model[p.player_id] = False
                    forced_out.add(p.player_id)
                    violations.append(f"NTC: {p.name} cannot be traded.")

        # Recently-signed check
        if self.config.enforce_recently_signed:
            for p in all_candidates:
                if p.is_recently_signed(self.config.recently_signed_months):
                    if p.player_id not in forced_out:
                        model[p.player_id] = False
                        forced_out.add(p.player_id)
                        violations.append(
                            f"RECENTLY SIGNED: {p.name} ({p.months_since_signing} mo)."
                        )

        # Roster-size check (after fixing forced_out players to not trade)
        tradeable_from_a = [p for p in candidates_from_a if model.get(p.player_id, True)]
        tradeable_from_b = [p for p in candidates_from_b if model.get(p.player_id, True)]

        final_a = len(roster_a) - len(tradeable_from_a) + len(tradeable_from_b)
        final_b = len(roster_b) - len(tradeable_from_b) + len(tradeable_from_a)
        lo, hi = self.config.min_roster_size, self.config.max_roster_size

        if not (lo <= final_a <= hi):
            violations.append(
                f"ROSTER SIZE: {roster_a[0].team if roster_a else 'TeamA'} "
                f"would have {final_a} players (need {lo}–{hi})."
            )
        if not (lo <= final_b <= hi):
            violations.append(
                f"ROSTER SIZE: {roster_b[0].team if roster_b else 'TeamB'} "
                f"would have {final_b} players (need {lo}–{hi})."
            )

        feasible = len([v for v in violations if "SAT" not in v and "ROSTER SIZE" in v]) == 0
        # Consider feasible if only NTC/recently-signed violations exist
        # (those just remove players from candidate pool, the core trade can still proceed
        #  as long as the remaining candidates form a valid trade)
        feasible = not any("ROSTER SIZE" in v for v in violations)

        return SATResult(
            feasible=feasible,
            forced_out=forced_out,
            model=model,
            violations=violations,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_fetcher import get_lakers_roster, get_nets_roster, _DEMO_PLAYERS

    cfg = ConstraintsConfig()
    checker = SATFeasibilityChecker(cfg)

    lakers = get_lakers_roster()
    nets   = get_nets_roster()

    # Demo: Nets send Ben Simmons → Lakers; Lakers send Anthony Davis → Nets
    from_nets   = [_DEMO_PLAYERS["ben_simmons"]]
    from_lakers = [_DEMO_PLAYERS["anthony_davis"]]

    result = checker.check(lakers, nets, from_lakers, from_nets)
    print(result)
    for v in result.violations:
        print(" !", v)
    print("Model:", result.model)
