"""
main.py
-------
Entry point for GM Mode: NBA Trade Package Optimizer.

Runs the full three-layer pipeline on a demo trade between two real teams,
using the current league year's rosters and salaries. The headline players are
whoever is actually the biggest contract on each side today, so the scenario
changes as the league does rather than being pinned to one old trade.

Pipeline:
  1. Load player data (data_fetcher.py)
  2. Train the GBT valuation model (valuation_model.py)
  3. Compute valuations for all trade participants
  4. Run SAT feasibility check (sat_layer.py)
  5. Run MIP optimisation with OR-Tools CP-SAT and PuLP CBC (mip_layer.py)
  6. Display full results

Also runs a small benchmark on 5 synthetic instances from instance_generator.py.

Usage:
    python3 main.py                      # live rosters, LAL vs BRK
    python3 main.py --teams BOS NYK      # any two teams
    python3 main.py --refresh            # force a refetch
    python3 main.py --offline            # hardcoded fallback pool, no network
"""

import argparse
import sys
import time

# ── Project modules ───────────────────────────────────────────────────────────
from constraints_config import ConstraintsConfig, FIRST_APRON
from data_fetcher import (
    TEAM_NAMES,
    current_season,
    dataset_provenance,
    get_demo_players,
    get_lakers_roster,
    get_nets_roster,
    load_dataset,
    season_label,
    _row_to_record,
)
from sat_layer import SATFeasibilityChecker
from valuation_model import PlayerValuationModel, TeamContext, LAL_CONTEXT, BKN_CONTEXT
from mip_layer import solve_both
from instance_generator import generate_benchmark_suite


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    """Print a bold section header."""
    width = 65
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def check_salary_match(
    outgoing: float,
    incoming: float,
    config: ConstraintsConfig,
    team: str,
) -> bool:
    """Check salary matching and print result. Returns True if valid."""
    cap = config.salary_cap(outgoing)
    ok  = incoming <= cap
    sym = "✓" if ok else "✗"
    print(
        f"  {sym} {team}: outgoing=${outgoing/1e6:.2f}M  "
        f"incoming=${incoming/1e6:.2f}M  "
        f"cap=${cap/1e6:.3f}M  {'OK' if ok else 'VIOLATION'}"
    )
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_rosters(team_a: str, team_b: str, offline: bool, refresh: bool):
    """Return (roster_a, roster_b, provenance) for the two teams.

    Uses the live dataset unless --offline was passed. Falls back to the
    hardcoded pool if the live load produced nothing for either team, so the
    demo still runs on a machine with no network.
    """
    if not offline:
        df = load_dataset(force_refresh=refresh)
        prov = dataset_provenance(df)
        ra = [_row_to_record(r) for _, r in df[df["team"] == team_a].iterrows()]
        rb = [_row_to_record(r) for _, r in df[df["team"] == team_b].iterrows()]
        if ra and rb:
            return ra, rb, prov
        print(f"  ! No live rows for {team_a}/{team_b}; using the offline pool.")

    return get_lakers_roster(), get_nets_roster(), {"is_live": False}


def main():
    ap = argparse.ArgumentParser(description="GM Mode: NBA trade package optimizer")
    ap.add_argument("--teams", nargs=2, metavar=("A", "B"), default=["LAL", "BRK"],
                    help="two team abbreviations, e.g. --teams BOS NYK")
    ap.add_argument("--refresh", action="store_true", help="force a data refetch")
    ap.add_argument("--offline", action="store_true",
                    help="use the hardcoded fallback pool instead of live data")
    args = ap.parse_args()

    team_a, team_b = (t.upper() for t in args.teams)
    for t in (team_a, team_b):
        if t not in TEAM_NAMES:
            sys.exit(f"Unknown team '{t}'. Valid: {', '.join(sorted(TEAM_NAMES))}")
    if team_a == team_b:
        sys.exit("Pick two different teams.")

    print("\n" + "█" * 65)
    print("  GM MODE: NBA Trade Package Optimizer")
    print("  Constraint Satisfaction + MIP + Gradient Boosted Trees")
    print("█" * 65)

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 0: Configuration
    # ─────────────────────────────────────────────────────────────────────
    section("0. Constraint Configuration")

    config = ConstraintsConfig(
        enforce_hard_cap         = True,
        hard_cap_threshold       = FIRST_APRON,
        enforce_no_trade_clauses = True,
        enforce_recently_signed  = True,
        recently_signed_months   = 12,
        salary_matching_threshold = 1.25,
        salary_matching_bonus    = 100_000,
    )
    print(config.describe())
    print("\n  (All toggles can be changed at the top of main.py.)")

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 1: Load player data
    # ─────────────────────────────────────────────────────────────────────
    section("1. Load Player Data")

    roster_a, roster_b, prov = load_rosters(
        team_a, team_b, offline=args.offline, refresh=args.refresh
    )

    if prov.get("is_live"):
        age = prov.get("age_hours")
        age_str = f"{age:.1f}h ago" if age is not None else "unknown"
        print(f"\n  Data: {prov['season_label']} rosters and salaries, "
              f"{prov['stats_season_label']} stats (fetched {age_str}).")
    else:
        print("\n  Data: offline fallback pool (not live).")

    for abbr, roster in ((team_a, roster_a), (team_b, roster_b)):
        payroll = sum(p.salary for p in roster)
        print(f"\n  {TEAM_NAMES.get(abbr, abbr)} roster: {len(roster)} players "
              f"· payroll ${payroll/1e6:.1f}M")
        for p in sorted(roster, key=lambda x: -x.salary):
            tag = " [NTC]" if p.has_ntc else ""
            rs  = f" [RS:{p.months_since_signing}mo]" if p.months_since_signing < 12 else ""
            print(f"    {p.name:<24} {p.position}  ${p.salary/1e6:.2f}M{tag}{rs}")

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 2: Train the valuation model
    # ─────────────────────────────────────────────────────────────────────
    section("2. Train GBT Valuation Model")

    model = PlayerValuationModel(n_estimators=200, max_depth=4, seed=42)
    all_players = roster_a + roster_b
    model.fit(players=all_players)
    print("  GBT model ready.")

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 3: Demo trade scenario
    # ─────────────────────────────────────────────────────────────────────
    section("3. Demo Trade Scenario")

    # Headline the biggest contract on each side. On live data this is a real,
    # current pairing; on the offline pool it reproduces the AD/Simmons demo.
    ad    = max(roster_a, key=lambda p: p.salary)   # team A → team B
    simmo = max(roster_b,   key=lambda p: p.salary)   # team B → team A

    print("\n  Proposed trade:")
    print(f"    {TEAM_NAMES.get(team_b, team_b)} send: {simmo.name} (${simmo.salary/1e6:.2f}M)")
    print(f"    {TEAM_NAMES.get(team_a, team_a)} send: {ad.name} (${ad.salary/1e6:.2f}M)")
    print("\n  (A 'first-round pick' has $0 cap value in this model.)")

    # For the MIP, the candidate pools define what is *available* to trade.
    # We make the specific trade players the sole candidates so the MIP
    # is forced to choose from exactly the proposed package.
    # (In a real GM tool you'd pass larger pools and let the MIP optimise.)
    candidates_from_lakers = [ad]      # Lakers offer AD
    candidates_from_nets   = [simmo]   # Nets offer Simmons

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 4: Compute valuations
    # ─────────────────────────────────────────────────────────────────────
    section("4. GBT Valuation Scores")

    # Reuse the report's contender-vs-rebuild framing, but attached to whichever
    # two teams were picked rather than to LAL/BKN by name.
    ctx_a = TeamContext(team_abbr=team_a, rebuild_score=LAL_CONTEXT.rebuild_score,
                        positional_needs=dict(LAL_CONTEXT.positional_needs))
    ctx_b = TeamContext(team_abbr=team_b, rebuild_score=BKN_CONTEXT.rebuild_score,
                        positional_needs=dict(BKN_CONTEXT.positional_needs))

    # `ad` is going to team B → value him in B's context
    ad_val_b     = model.predict(ad, ctx_b)
    ad.valuation = ad_val_b            # store on player (used by MIP)

    # `simmo` is going to team A → value him in A's context
    sim_val_a       = model.predict(simmo, ctx_a)
    simmo.valuation = sim_val_a

    # Also compute "staying" valuations for context
    ad_val_a  = model.predict(ad,    ctx_a)
    sim_val_b = model.predict(simmo, ctx_b)

    print(f"\n  {ad.name}")
    print(f"    Value to {team_a} (current team)  : {ad_val_a:+.4f}")
    print(f"    Value to {team_b} (receiving team): {ad_val_b:+.4f}")

    print(f"\n  {simmo.name}")
    print(f"    Value to {team_b} (current team)  : {sim_val_b:+.4f}")
    print(f"    Value to {team_a} (receiving team): {sim_val_a:+.4f}")

    print(f"\n  Net trade value ({team_b} gains {ad.name}, {team_a} gains {simmo.name}):")
    net_b = ad_val_b  - sim_val_b   # B gains `ad`, loses `simmo`
    net_a = sim_val_a - ad_val_a    # A gains `simmo`, loses `ad`
    print(f"    {team_b} net: {net_b:+.4f}  {'(gain)' if net_b > 0 else '(loss)'}")
    print(f"    {team_a} net: {net_a:+.4f}  {'(gain)' if net_a > 0 else '(loss)'}")

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 5: SAT feasibility check
    # ─────────────────────────────────────────────────────────────────────
    section("5. SAT Feasibility Check (PicoSAT)")

    checker    = SATFeasibilityChecker(config)
    sat_result = checker.check(
        roster_a           = roster_a,
        roster_b           = roster_b,
        candidates_from_a  = candidates_from_lakers,   # LAL offers
        candidates_from_b  = candidates_from_nets,     # BKN offers
    )

    print(f"\n  SAT result: {'✓ FEASIBLE' if sat_result.feasible else '✗ INFEASIBLE'}")
    if sat_result.violations:
        print("\n  Constraint violations detected:")
        for v in sat_result.violations:
            print(f"    ! {v}")
    else:
        print("  No boolean constraint violations.")

    print(f"\n  Players forced out of trade: {len(sat_result.forced_out)}")
    if sat_result.forced_out:
        for pid in sat_result.forced_out:
            # Find name
            for p in candidates_from_lakers + candidates_from_nets:
                if p.player_id == pid:
                    print(f"    - {p.name}")

    print("\n  Variable assignments from SAT model:")
    for player in candidates_from_lakers + candidates_from_nets:
        traded_flag = sat_result.model.get(player.player_id, "?")
        print(f"    traded[{player.name}] = {traded_flag}")

    # Manual salary matching check (not in SAT, handled by MIP)
    print("\n  Salary matching check (MIP constraint, shown here for info):")
    sal_out_a = ad.salary       # A sends `ad`
    sal_in_a  = simmo.salary    # A receives `simmo`
    sal_out_b = simmo.salary
    sal_in_b  = ad.salary

    sm_a = check_salary_match(sal_out_a, sal_in_a, config, team_a)
    sm_b = check_salary_match(sal_out_b, sal_in_b, config, team_b)

    # Hard-cap check
    a_total_after = sum(p.salary for p in roster_a) - sal_out_a + sal_in_a
    b_total_after = sum(p.salary for p in roster_b)   - sal_out_b + sal_in_b
    print(f"\n  Hard cap check (${config.hard_cap_threshold/1e6:.1f}M limit):")
    a_ok = a_total_after <= config.hard_cap_threshold
    b_ok = b_total_after <= config.hard_cap_threshold
    print(f"    {'✓' if a_ok else '✗'} {team_a} post-trade payroll: ${a_total_after/1e6:.2f}M")
    print(f"    {'✓' if b_ok else '✗'} {team_b} post-trade payroll: ${b_total_after/1e6:.2f}M")

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 6: MIP optimisation
    # ─────────────────────────────────────────────────────────────────────
    section("6. MIP Optimisation")

    print("\n  Running OR-Tools CP-SAT and PuLP CBC …")

    ortools_result, pulp_result = solve_both(
        candidates_from_a  = candidates_from_lakers,
        candidates_from_b  = candidates_from_nets,
        roster_a           = roster_a,
        roster_b           = roster_b,
        sat_result         = sat_result,
        config             = config,
        team_a             = team_a,
        team_b             = team_b,
    )

    print("\n--- OR-Tools CP-SAT ---")
    print(ortools_result.display())

    print("\n--- PuLP CBC ---")
    print(pulp_result.display())

    # Compare the two solvers
    if ortools_result.optimal and pulp_result.optimal:
        delta = abs(ortools_result.objective_value - pulp_result.objective_value)
        print(f"\n  Solver agreement check: |obj_OR - obj_PuLP| = {delta:.6f}")
        if delta < 1e-3:
            print("  ✓ Both solvers agree on the optimal objective value.")
        else:
            print("  ! Solvers disagree, check model formulations.")

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 6b: Re-run with hard cap DISABLED (constraint toggle demo)
    # ─────────────────────────────────────────────────────────────────────
    section("6b. User Constraint Toggle: Hard Cap OFF")

    config_no_cap = ConstraintsConfig(
        enforce_hard_cap          = False,   # <── toggled off
        enforce_no_trade_clauses  = True,
        enforce_recently_signed   = True,
        salary_matching_threshold = 1.25,
        salary_matching_bonus     = 100_000,
    )
    print("\n  Disabling hard cap enforcement and re-solving …")
    or2, pu2 = solve_both(
        candidates_from_a  = candidates_from_lakers,
        candidates_from_b  = candidates_from_nets,
        roster_a           = roster_a,
        roster_b           = roster_b,
        sat_result         = sat_result,
        config             = config_no_cap,
        team_a             = team_a,
        team_b             = team_b,
    )
    print("\n--- OR-Tools CP-SAT (no hard cap) ---")
    print(or2.display())
    print("--- PuLP CBC (no hard cap) ---")
    print(pu2.display())

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 7: Benchmarking on synthetic instances
    # ─────────────────────────────────────────────────────────────────────
    section("7. Benchmark on Synthetic Trade Instances")

    print("\n  Generating 5 synthetic 2-team trade instances …\n")
    suite = generate_benchmark_suite(
        k                    = 5,
        n_teams              = 2,
        n_players_each       = 4,
        salary_variance      = 0.5,
        constraint_tightness = 0.3,
        base_seed            = 1000,
    )

    header = (
        f"  {'Seed':>6}  {'Exp.Feas':>9}  "
        f"{'SAT':>8}  {'OR-status':>12}  {'OR-obj':>8}  "
        f"{'PuLP-status':>12}  {'PuLP-obj':>8}  "
        f"{'OR-ms':>7}  {'PuLP-ms':>7}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for inst in suite:
        # Synthetic instances have their own team labels; keep them out of the
        # outer team_a/team_b so the real matchup above is not clobbered.
        bench_a, bench_b = inst.teams[0], inst.teams[1]
        ra = inst.rosters[bench_a]
        rb = inst.rosters[bench_b]
        ca = inst.candidates[bench_a]
        cb = inst.candidates[bench_b]

        # Compute synthetic valuations using neutral context
        bctx_a = TeamContext(team_abbr=bench_a)
        bctx_b = TeamContext(team_abbr=bench_b)

        for p in ca:
            p.valuation = model.predict(p, bctx_b)   # A's players valued by B
        for p in cb:
            p.valuation = model.predict(p, bctx_a)   # B's players valued by A

        # SAT check
        sat_r = checker.check(ra, rb, ca, cb)

        # MIP solve
        or_r, pu_r = solve_both(ca, cb, ra, rb, sat_r, inst.config, bench_a, bench_b)

        print(
            f"  {inst.seed:>6}  {str(inst.expected_feasible):>9}  "
            f"{'FEAS' if sat_r.feasible else 'INFEAS':>8}  "
            f"{or_r.status:>12}  {or_r.objective_value:>8.3f}  "
            f"{pu_r.status:>12}  {pu_r.objective_value:>8.3f}  "
            f"{or_r.solve_time_ms:>7.1f}  {pu_r.solve_time_ms:>7.1f}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Done
    # ─────────────────────────────────────────────────────────────────────
    section("Pipeline Complete")
    print("\n  All three layers executed successfully.")
    print("  See above for SAT feasibility, MIP solutions, and valuations.\n")


if __name__ == "__main__":
    main()
