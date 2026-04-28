"""
main.py
-------
Entry point for GM Mode: NBA Trade Package Optimizer.

Runs the full three-layer pipeline on the demo scenario:
  Brooklyn Nets trade Ben Simmons to the Los Angeles Lakers
  in exchange for Anthony Davis.

Pipeline:
  1. Load player data (data_fetcher.py)
  2. Train the GBT valuation model (valuation_model.py)
  3. Compute valuations for all trade participants
  4. Run SAT feasibility check (sat_layer.py)
  5. Run MIP optimisation with OR-Tools CP-SAT and PuLP CBC (mip_layer.py)
  6. Display full results

Also runs a small benchmark on 5 synthetic instances from instance_generator.py.
"""

import sys
import time

# ── Project modules ───────────────────────────────────────────────────────────
from constraints_config import ConstraintsConfig
from data_fetcher import (
    get_lakers_roster,
    get_nets_roster,
    _DEMO_PLAYERS,
)
from sat_layer import SATFeasibilityChecker
from valuation_model import PlayerValuationModel, LAL_CONTEXT, BKN_CONTEXT
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

def main():
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
        hard_cap_threshold       = 165_000_000,
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

    lakers_roster = get_lakers_roster()
    nets_roster   = get_nets_roster()

    print(f"\n  Lakers roster: {len(lakers_roster)} players")
    for p in lakers_roster:
        tag = " [NTC]" if p.has_ntc else ""
        rs  = f" [RS:{p.months_since_signing}mo]" if p.months_since_signing < 12 else ""
        print(f"    {p.name:<24} {p.position}  ${p.salary/1e6:.2f}M{tag}{rs}")

    print(f"\n  Nets roster: {len(nets_roster)} players")
    for p in nets_roster:
        tag = " [NTC]" if p.has_ntc else ""
        rs  = f" [RS:{p.months_since_signing}mo]" if p.months_since_signing < 12 else ""
        print(f"    {p.name:<24} {p.position}  ${p.salary/1e6:.2f}M{tag}{rs}")

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 2: Train the valuation model
    # ─────────────────────────────────────────────────────────────────────
    section("2. Train GBT Valuation Model")

    model = PlayerValuationModel(n_estimators=200, max_depth=4, seed=42)
    all_players = lakers_roster + nets_roster
    model.fit(players=all_players)
    print("  GBT model ready.")

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 3: Demo trade scenario
    # ─────────────────────────────────────────────────────────────────────
    section("3. Demo Trade Scenario")

    ad     = _DEMO_PLAYERS["anthony_davis"]    # Lakers → Nets
    simmo  = _DEMO_PLAYERS["ben_simmons"]      # Nets → Lakers

    print("\n  Proposed trade:")
    print(f"    Brooklyn Nets  send: {simmo.name} (${simmo.salary/1e6:.2f}M)")
    print(f"    Los Angeles Lakers send: {ad.name} (${ad.salary/1e6:.2f}M)")
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

    # Anthony Davis is going to the Nets → value him in BKN context
    ad_val_bkn   = model.predict(ad,    BKN_CONTEXT)
    ad.valuation = ad_val_bkn          # store on player (used by MIP)

    # Ben Simmons is going to the Lakers → value him in LAL context
    sim_val_lal  = model.predict(simmo, LAL_CONTEXT)
    simmo.valuation = sim_val_lal

    # Also compute "staying" valuations for context
    ad_val_lal   = model.predict(ad,    LAL_CONTEXT)
    sim_val_bkn  = model.predict(simmo, BKN_CONTEXT)

    print(f"\n  Anthony Davis")
    print(f"    Value to LAL (current team)  : {ad_val_lal:+.4f}")
    print(f"    Value to BKN (receiving team): {ad_val_bkn:+.4f}")

    print(f"\n  Ben Simmons")
    print(f"    Value to BKN (current team)  : {sim_val_bkn:+.4f}")
    print(f"    Value to LAL (receiving team): {sim_val_lal:+.4f}")

    print(f"\n  Net trade value (BKN gains AD, LAL gains Simmons):")
    net_bkn = ad_val_bkn  - sim_val_bkn   # BKN gains AD, loses Simmons
    net_lal = sim_val_lal - ad_val_lal    # LAL gains Simmons, loses AD
    print(f"    BKN net: {net_bkn:+.4f}  {'(gain)' if net_bkn > 0 else '(loss)'}")
    print(f"    LAL net: {net_lal:+.4f}  {'(gain)' if net_lal > 0 else '(loss)'}")

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 5: SAT feasibility check
    # ─────────────────────────────────────────────────────────────────────
    section("5. SAT Feasibility Check (PicoSAT)")

    checker    = SATFeasibilityChecker(config)
    sat_result = checker.check(
        roster_a           = lakers_roster,
        roster_b           = nets_roster,
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
    sal_out_lal = ad.salary      # LAL sends AD
    sal_in_lal  = simmo.salary   # LAL receives Simmons
    sal_out_bkn = simmo.salary
    sal_in_bkn  = ad.salary

    sm_lal = check_salary_match(sal_out_lal, sal_in_lal, config, "LAL")
    sm_bkn = check_salary_match(sal_out_bkn, sal_in_bkn, config, "BKN")

    # Hard-cap check
    lal_total_after = sum(p.salary for p in lakers_roster) - sal_out_lal + sal_in_lal
    bkn_total_after = sum(p.salary for p in nets_roster)   - sal_out_bkn + sal_in_bkn
    print(f"\n  Hard cap check (${config.hard_cap_threshold/1e6:.0f}M limit):")
    lal_ok = lal_total_after <= config.hard_cap_threshold
    bkn_ok = bkn_total_after <= config.hard_cap_threshold
    print(f"    {'✓' if lal_ok else '✗'} LAL post-trade payroll: ${lal_total_after/1e6:.2f}M")
    print(f"    {'✓' if bkn_ok else '✗'} BKN post-trade payroll: ${bkn_total_after/1e6:.2f}M")

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 6: MIP optimisation
    # ─────────────────────────────────────────────────────────────────────
    section("6. MIP Optimisation")

    print("\n  Running OR-Tools CP-SAT and PuLP CBC …")

    ortools_result, pulp_result = solve_both(
        candidates_from_a  = candidates_from_lakers,
        candidates_from_b  = candidates_from_nets,
        roster_a           = lakers_roster,
        roster_b           = nets_roster,
        sat_result         = sat_result,
        config             = config,
        team_a             = "LAL",
        team_b             = "BKN",
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
        roster_a           = lakers_roster,
        roster_b           = nets_roster,
        sat_result         = sat_result,
        config             = config_no_cap,
        team_a             = "LAL",
        team_b             = "BKN",
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
        team_a, team_b = inst.teams[0], inst.teams[1]
        ra = inst.rosters[team_a]
        rb = inst.rosters[team_b]
        ca = inst.candidates[team_a]
        cb = inst.candidates[team_b]

        # Compute synthetic valuations using neutral context
        from valuation_model import TeamContext
        ctx_a = TeamContext(team_abbr=team_a)
        ctx_b = TeamContext(team_abbr=team_b)

        for p in ca:
            p.valuation = model.predict(p, ctx_b)   # A's players valued by B
        for p in cb:
            p.valuation = model.predict(p, ctx_a)   # B's players valued by A

        # SAT check
        sat_r = checker.check(ra, rb, ca, cb)

        # MIP solve
        or_r, pu_r = solve_both(ca, cb, ra, rb, sat_r, inst.config, team_a, team_b)

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
