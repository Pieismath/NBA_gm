"""
benchmark_sweep.py
------------------
Two sweeps that back the experiments described in the final report.

Sweep A: constraint_tightness from 0.0 to 1.0 in steps of 0.1, 50 random
2-team trade instances per level. Records SAT feasibility, MIP feasibility,
solver agreement, and median solve time per solver.

Sweep B: n_players_each from 2 to 10, 30 random instances per size, at
fixed mid-tightness. Shows how solve time scales with candidate pool size
across the two MIP backends.

Outputs:
  benchmark_results_tightness.csv
  benchmark_results_size.csv
  benchmark_feasibility.png
  benchmark_solvetime_tightness.png
  benchmark_solvetime_size.png
  prints summary tables to stdout

Run:  python3 benchmark_sweep.py
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from instance_generator import generate_instance
from sat_layer import SATFeasibilityChecker
from valuation_model import PlayerValuationModel, TeamContext
from mip_layer import solve_both


HERE = Path(__file__).parent

# Solver-status strings that count as "found a feasible solution"
_OK_STATUSES = {"OPTIMAL", "FEASIBLE", "Optimal", "Feasible"}


def _solve_instance(inst, model):
    """Run SAT + both MIPs on one instance, return a dict of measurements."""
    team_a, team_b = inst.teams[0], inst.teams[1]
    ra = inst.rosters[team_a]
    rb = inst.rosters[team_b]
    ca = inst.candidates[team_a]
    cb = inst.candidates[team_b]

    ctx_a = TeamContext(team_abbr=team_a)
    ctx_b = TeamContext(team_abbr=team_b)
    for p in ca:
        p.valuation = model.predict(p, ctx_b)
    for p in cb:
        p.valuation = model.predict(p, ctx_a)

    checker = SATFeasibilityChecker(inst.config)
    sat_r = checker.check(ra, rb, ca, cb)

    row = {
        "expected_feasible": inst.expected_feasible,
        "sat_feasible": sat_r.feasible,
        "n_forced_out": len(sat_r.forced_out),
        "or_status": "",
        "or_obj": float("nan"),
        "or_ms": float("nan"),
        "pu_status": "",
        "pu_obj": float("nan"),
        "pu_ms": float("nan"),
        "or_found_solution": False,
        "pu_found_solution": False,
        "solvers_agree": False,
    }

    if sat_r.feasible:
        or_r, pu_r = solve_both(ca, cb, ra, rb, sat_r, inst.config, team_a, team_b)
        row.update({
            "or_status": or_r.status,
            "or_obj": or_r.objective_value,
            "or_ms": or_r.solve_time_ms,
            "pu_status": pu_r.status,
            "pu_obj": pu_r.objective_value,
            "pu_ms": pu_r.solve_time_ms,
            "or_found_solution": or_r.status in _OK_STATUSES,
            "pu_found_solution": pu_r.status in _OK_STATUSES,
        })
        if row["or_found_solution"] and row["pu_found_solution"]:
            row["solvers_agree"] = abs(or_r.objective_value - pu_r.objective_value) < 1e-2
    return row


def sweep_tightness(model, n_per_level=50, base_seed=1000):
    """Sweep constraint_tightness from 0.0 to 1.0."""
    rows = []
    levels = [round(i * 0.1, 2) for i in range(11)]
    for t in levels:
        print(f"\n[tightness sweep] t={t:.1f} ({n_per_level} instances)")
        for k in range(n_per_level):
            seed = base_seed + int(t * 100) * 1000 + k
            inst = generate_instance(
                n_teams=2,
                n_players_each=4,
                salary_variance=0.5,
                constraint_tightness=t,
                seed=seed,
            )
            row = _solve_instance(inst, model)
            row["tightness"] = t
            row["n_players_each"] = 4
            row["seed"] = seed
            rows.append(row)
        feas = sum(1 for r in rows[-n_per_level:] if r["sat_feasible"]) / n_per_level
        mip_feas = sum(1 for r in rows[-n_per_level:] if r["or_found_solution"]) / n_per_level
        print(f"  SAT feasible: {feas*100:.0f}%   MIP feasible: {mip_feas*100:.0f}%")
    return rows


def sweep_size(model, n_per_size=30, base_seed=5000):
    """Sweep n_players_each from 2 to 10 at fixed mid-tightness."""
    rows = []
    sizes = list(range(2, 11))
    for n in sizes:
        print(f"\n[size sweep] n_players_each={n} ({n_per_size} instances)")
        for k in range(n_per_size):
            seed = base_seed + n * 1000 + k
            inst = generate_instance(
                n_teams=2,
                n_players_each=n,
                salary_variance=0.5,
                constraint_tightness=0.3,
                seed=seed,
            )
            row = _solve_instance(inst, model)
            row["n_players_each"] = n
            row["seed"] = seed
            rows.append(row)
        or_med = statistics.median(
            [r["or_ms"] for r in rows[-n_per_size:] if r["or_found_solution"]]
        ) if any(r["or_found_solution"] for r in rows[-n_per_size:]) else float("nan")
        print(f"  median OR-Tools: {or_med:.2f} ms")
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def aggregate_tightness(rows):
    by_t = {}
    for r in rows:
        by_t.setdefault(r["tightness"], []).append(r)
    summary = []
    for t in sorted(by_t):
        bucket = by_t[t]
        sat_feas = [r for r in bucket if r["sat_feasible"]]
        mip_ok = [r for r in bucket if r["or_found_solution"]]
        or_times = [r["or_ms"] for r in mip_ok]
        pu_times = [r["pu_ms"] for r in bucket if r["pu_found_solution"]]
        agree = sum(1 for r in bucket if r["solvers_agree"])
        summary.append({
            "tightness": t,
            "n": len(bucket),
            "sat_feas_rate": len(sat_feas) / len(bucket),
            "mip_feas_rate": len(mip_ok) / len(bucket),
            "agree_rate": agree / len(bucket),
            "or_median_ms": statistics.median(or_times) if or_times else float("nan"),
            "pu_median_ms": statistics.median(pu_times) if pu_times else float("nan"),
            "avg_forced_out": statistics.mean([r["n_forced_out"] for r in bucket]),
        })
    return summary


def aggregate_size(rows):
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_players_each"], []).append(r)
    summary = []
    for n in sorted(by_n):
        bucket = by_n[n]
        or_times = [r["or_ms"] for r in bucket if r["or_found_solution"]]
        pu_times = [r["pu_ms"] for r in bucket if r["pu_found_solution"]]
        summary.append({
            "n_players_each": n,
            "n_instances": len(bucket),
            "or_median_ms": statistics.median(or_times) if or_times else float("nan"),
            "or_mean_ms": statistics.mean(or_times) if or_times else float("nan"),
            "pu_median_ms": statistics.median(pu_times) if pu_times else float("nan"),
            "pu_mean_ms": statistics.mean(pu_times) if pu_times else float("nan"),
            "mip_feas_rate": sum(1 for r in bucket if r["or_found_solution"]) / len(bucket),
        })
    return summary


def print_tightness(summary):
    print()
    print("  TIGHTNESS SWEEP SUMMARY")
    print(f"  {'t':>4} {'n':>4} {'SAT %':>7} {'MIP %':>7} "
          f"{'agree %':>9} {'OR ms':>8} {'PuLP ms':>9} {'avg forced':>11}")
    print("  " + "-" * 68)
    for s in summary:
        print(
            f"  {s['tightness']:>4.1f} {s['n']:>4d} "
            f"{s['sat_feas_rate']*100:>6.1f}% {s['mip_feas_rate']*100:>6.1f}% "
            f"{s['agree_rate']*100:>8.1f}% "
            f"{s['or_median_ms']:>8.2f} {s['pu_median_ms']:>9.2f} "
            f"{s['avg_forced_out']:>11.2f}"
        )


def print_size(summary):
    print()
    print("  SIZE SWEEP SUMMARY (tightness fixed at 0.3)")
    print(f"  {'n_each':>7} {'inst':>5} {'MIP %':>7} "
          f"{'OR med ms':>10} {'OR mean ms':>11} {'PuLP med ms':>12}")
    print("  " + "-" * 60)
    for s in summary:
        print(
            f"  {s['n_players_each']:>7d} {s['n_instances']:>5d} "
            f"{s['mip_feas_rate']*100:>6.1f}% "
            f"{s['or_median_ms']:>10.2f} {s['or_mean_ms']:>11.2f} "
            f"{s['pu_median_ms']:>12.2f}"
        )


def make_plots(t_sum, s_sum):
    # Feasibility (SAT and MIP) vs tightness
    ts = [s["tightness"] for s in t_sum]
    sat_f = [s["sat_feas_rate"] * 100 for s in t_sum]
    mip_f = [s["mip_feas_rate"] * 100 for s in t_sum]
    forced = [s["avg_forced_out"] for s in t_sum]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(ts, sat_f, marker="o", linewidth=2, label="SAT feasible", color="#1F8E5A")
    ax1.plot(ts, mip_f, marker="s", linewidth=2, label="MIP found solution", color="#1D428A")
    ax1.set_xlabel("constraint tightness")
    ax1.set_ylabel("rate (%)")
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left")

    ax2 = ax1.twinx()
    ax2.plot(ts, forced, marker="^", linewidth=1.5, linestyle="--",
             label="avg players forced out by SAT", color="#E86A1A", alpha=0.7)
    ax2.set_ylabel("avg forced-out players")
    ax2.legend(loc="upper right")

    plt.title("SAT vs MIP feasibility, and SAT preprocessing effect")
    fig.tight_layout()
    fig.savefig(HERE / "benchmark_feasibility.png", dpi=150)
    plt.close(fig)
    print(f"  wrote benchmark_feasibility.png")

    # Solve time vs tightness
    or_t = [s["or_median_ms"] for s in t_sum]
    pu_t = [s["pu_median_ms"] for s in t_sum]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ts, or_t, marker="o", linewidth=2, label="OR-Tools CP-SAT", color="#E86A1A")
    ax.plot(ts, pu_t, marker="s", linewidth=2, label="PuLP / CBC", color="#1D428A")
    ax.set_xlabel("constraint tightness")
    ax.set_ylabel("median solve time (ms)")
    ax.set_title("MIP solve time vs constraint tightness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "benchmark_solvetime_tightness.png", dpi=150)
    plt.close(fig)
    print(f"  wrote benchmark_solvetime_tightness.png")

    # Solve time vs size
    ns = [s["n_players_each"] for s in s_sum]
    or_t = [s["or_median_ms"] for s in s_sum]
    pu_t = [s["pu_median_ms"] for s in s_sum]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ns, or_t, marker="o", linewidth=2, label="OR-Tools CP-SAT", color="#E86A1A")
    ax.plot(ns, pu_t, marker="s", linewidth=2, label="PuLP / CBC", color="#1D428A")
    ax.set_xlabel("candidate pool size per team")
    ax.set_ylabel("median solve time (ms)")
    ax.set_title("MIP solve time vs problem size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "benchmark_solvetime_size.png", dpi=150)
    plt.close(fig)
    print(f"  wrote benchmark_solvetime_size.png")


if __name__ == "__main__":
    print("Training valuation model once...")
    model = PlayerValuationModel(n_estimators=200, max_depth=4, seed=42)
    model.fit(players=[])
    print()

    t_rows = sweep_tightness(model)
    write_csv(t_rows, HERE / "benchmark_results_tightness.csv")

    s_rows = sweep_size(model)
    write_csv(s_rows, HERE / "benchmark_results_size.csv")

    t_sum = aggregate_tightness(t_rows)
    s_sum = aggregate_size(s_rows)

    print_tightness(t_sum)
    print_size(s_sum)

    make_plots(t_sum, s_sum)
