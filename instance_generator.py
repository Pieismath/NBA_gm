"""
Synthetic trade-scenario generator for benchmarking. Builds NBA-like trade
scenarios from scratch (no real data needed); the output drops straight into
the SAT/MIP layers like a real scenario would.

Knobs:
  n_teams              2 or 3
  n_players_each       candidates offered per team
  salary_variance      0 = uniform salaries, 1 = max spread
  constraint_tightness 0 = nearly every trade is legal,
                       1 = nearly none (lots of NTC players, tight rosters)
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import Optional

from data_fetcher import PlayerRecord
from constraints_config import ConstraintsConfig


@dataclass
class TradeInstance:
    """
    One synthetic trade scenario, structured the same way real scenarios are
    so the SAT and MIP layers can consume it directly.

    teams holds the abbreviations (e.g. ["A", "B"]). rosters and candidates
    are dicts keyed by abbr; candidates is a subset of the roster that the
    instance is "offering up" for trade. config is the ConstraintsConfig the
    generator used. expected_feasible is whether we built this to have at
    least one legal trade. seed is the RNG seed for reproducibility.
    """
    teams: list[str]
    rosters: dict[str, list[PlayerRecord]]
    candidates: dict[str, list[PlayerRecord]]
    config: ConstraintsConfig
    expected_feasible: bool
    seed: int

    def summary(self) -> str:
        lines = [f"TradeInstance(seed={self.seed}, "
                 f"expected_feasible={self.expected_feasible})"]
        for team in self.teams:
            roster = self.rosters[team]
            cands  = self.candidates[team]
            tot_sal = sum(p.salary for p in roster)
            lines.append(
                f"  {team}: {len(roster)} players, "
                f"${tot_sal/1e6:.1f}M total salary, "
                f"{len(cands)} candidates"
            )
            for p in cands:
                ntc_tag = " [NTC]" if p.has_ntc else ""
                rs_tag  = f" [RS:{p.months_since_signing}mo]" if p.months_since_signing < 12 else ""
                lines.append(
                    f"    • {p.name:<22} {p.position}  "
                    f"${p.salary/1e6:.1f}M  "
                    f"BPM={p.bpm:+.1f}  VORP={p.vorp:.1f}{ntc_tag}{rs_tag}"
                )
        return "\n".join(lines)


# ---- Internal helpers ----

_POSITIONS = ["PG", "SG", "SF", "PF", "C"]

# Approximate NBA salary ranges for synthetic generation (2024-25)
_MIN_SALARY = 1_119_563    # rookie minimum
_MID_SALARY = 12_000_000   # mid-level exception territory
_MAX_SALARY = 50_000_000   # supermax territory


def _random_player(
    player_id: int,
    team: str,
    rng: random.Random,
    salary_variance: float = 0.5,
    constraint_tightness: float = 0.3,
) -> PlayerRecord:
    """
    Generate a single synthetic PlayerRecord.

    Higher `salary_variance`   → salaries spread from min to max.
    Higher `constraint_tightness` → more NTCs and recently-signed players.
    """
    # Salary: interpolate between flat $10M and full min-max spread
    base_salary = _MID_SALARY
    spread = (_MAX_SALARY - _MIN_SALARY) * salary_variance
    salary = rng.gauss(base_salary, spread / 4)
    salary = max(_MIN_SALARY, min(_MAX_SALARY, salary))

    # Stats: loosely correlated with salary (star players cost more)
    star_factor = (salary - _MIN_SALARY) / (_MAX_SALARY - _MIN_SALARY)  # 0-1
    bpm   = rng.gauss(star_factor * 5 - 1, 1.5)          # better players → higher BPM
    vorp  = max(0.0, rng.gauss(star_factor * 4, 1.0))
    ts    = rng.gauss(0.555 + star_factor * 0.04, 0.03)
    age   = rng.randint(19, 37)

    # NTC: rare normally, more common at high constraint tightness
    ntc_prob = 0.05 + 0.25 * constraint_tightness
    has_ntc  = rng.random() < ntc_prob

    # Recently signed: controlled by tightness
    if rng.random() < constraint_tightness * 0.4:
        months_since_signing = rng.randint(1, 11)   # recently signed → locked
    else:
        months_since_signing = rng.randint(12, 60)  # safe to trade

    # Generate a readable synthetic name
    first_names = ["Alex", "Jordan", "Tyler", "Marcus", "Devon", "Malik",
                   "Isaiah", "Jaylen", "Chris", "Trae", "Zion", "Luka",
                   "James", "Kevin", "Stephen", "Damian", "Jimmy", "Bam"]
    last_names  = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Davis",
                   "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas",
                   "Jackson", "White", "Harris", "Martin", "Young", "Walker"]
    name = f"{rng.choice(first_names)} {rng.choice(last_names)} #{player_id}"

    return PlayerRecord(
        player_id         = player_id,
        name              = name,
        team              = team,
        position          = rng.choice(_POSITIONS),
        age               = age,
        salary            = salary,
        bpm               = bpm,
        vorp              = vorp,
        ts_pct            = ts,
        has_ntc           = has_ntc,
        months_since_signing = months_since_signing,
    )


def _build_roster(
    team: str,
    n_candidates: int,
    roster_size: int,
    rng: random.Random,
    salary_variance: float,
    constraint_tightness: float,
    id_offset: int,
) -> tuple[list[PlayerRecord], list[PlayerRecord]]:
    """
    Build a full roster and a candidate subset for one team.

    Returns (full_roster, candidates).
    """
    full_roster: list[PlayerRecord] = []
    for i in range(roster_size):
        pid = id_offset + i
        p = _random_player(pid, team, rng, salary_variance, constraint_tightness)
        full_roster.append(p)

    # Candidates are the first n_candidates players
    candidates = full_roster[:n_candidates]
    return full_roster, candidates


# ---- Public generator ----

def generate_instance(
    n_teams: int = 2,
    n_players_each: int = 3,
    salary_variance: float = 0.5,
    constraint_tightness: float = 0.3,
    roster_size: int = 13,
    seed: Optional[int] = None,
    config: Optional[ConstraintsConfig] = None,
) -> TradeInstance:
    """
    Generate one synthetic trade instance.

    Parameters
    ----------
    n_teams : int
        Number of teams (2 or 3 supported).
    n_players_each : int
        Number of trade candidates per team.
    salary_variance : float [0, 1]
        How spread out salaries are.
    constraint_tightness : float [0, 1]
        Fraction of players that violate boolean constraints.
        Higher → harder instances, fewer feasible solutions.
    roster_size : int
        Full roster size per team (must stay in [13, 15] after trade).
    seed : int or None
        Random seed for reproducibility.
    config : ConstraintsConfig or None
        Constraint settings to embed in the instance (uses default if None).

    Returns
    -------
    TradeInstance
    """
    if n_teams not in (2, 3):
        raise ValueError(f"n_teams must be 2 or 3, got {n_teams}")
    if seed is None:
        seed = random.randint(0, 10**9)

    rng = random.Random(seed)
    cfg = config or ConstraintsConfig()

    team_names = [chr(65 + i) for i in range(n_teams)]   # ["A", "B"] or ["A", "B", "C"]
    rosters: dict[str, list[PlayerRecord]] = {}
    candidates: dict[str, list[PlayerRecord]] = {}

    id_counter = 1
    for t in team_names:
        full, cands = _build_roster(
            team=t,
            n_candidates=n_players_each,
            roster_size=roster_size,
            rng=rng,
            salary_variance=salary_variance,
            constraint_tightness=constraint_tightness,
            id_offset=id_counter,
        )
        rosters[t]    = full
        candidates[t] = cands
        id_counter   += roster_size

    # Determine expected feasibility:
    # A trade is expected infeasible if every candidate on any team side is
    # locked (NTC or recently signed) when those constraints are active.
    def is_locked(p: PlayerRecord) -> bool:
        if cfg.enforce_no_trade_clauses and p.has_ntc:
            return True
        if cfg.enforce_recently_signed and p.months_since_signing < cfg.recently_signed_months:
            return True
        return False

    expected_feasible = True
    for t in team_names:
        tradeable = [p for p in candidates[t] if not is_locked(p)]
        if len(tradeable) == 0:
            expected_feasible = False
            break

    return TradeInstance(
        teams            = team_names,
        rosters          = rosters,
        candidates       = candidates,
        config           = cfg,
        expected_feasible = expected_feasible,
        seed             = seed,
    )


def generate_benchmark_suite(
    k: int = 10,
    n_teams: int = 2,
    n_players_each: int = 3,
    salary_variance: float = 0.5,
    constraint_tightness: float = 0.3,
    base_seed: int = 0,
) -> list[TradeInstance]:
    """
    Generate k independent trade instances with consecutive seeds.

    Useful for benchmarking solver runtime across many different scenarios.
    """
    return [
        generate_instance(
            n_teams=n_teams,
            n_players_each=n_players_each,
            salary_variance=salary_variance,
            constraint_tightness=constraint_tightness,
            seed=base_seed + i,
        )
        for i in range(k)
    ]


if __name__ == "__main__":
    # quick self-test
    print("=== 2-team instance ===")
    inst = generate_instance(n_teams=2, n_players_each=3, seed=7)
    print(inst.summary())

    print("\n=== 3-team instance (high tightness) ===")
    inst3 = generate_instance(
        n_teams=3, n_players_each=2, constraint_tightness=0.9, seed=99
    )
    print(inst3.summary())
    print(f"\nExpected feasible: {inst3.expected_feasible}")

    print("\n=== Benchmark suite (5 instances) ===")
    suite = generate_benchmark_suite(k=5)
    for s in suite:
        print(f"  seed={s.seed} | feasible={s.expected_feasible}")
