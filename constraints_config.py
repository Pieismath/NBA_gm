"""
constraints_config.py
---------------------
User-configurable runtime toggles for the GM Mode optimizer.

All constraint knobs live in one ConstraintsConfig dataclass so they can be
passed through every layer (SAT → MIP) without being hard-coded anywhere else.
The user can flip any toggle before running the pipeline.
"""

from dataclasses import dataclass, field


@dataclass
class ConstraintsConfig:
    """
    Single source of truth for which NBA CBA constraints are active and
    what their numerical thresholds are.

    Attributes
    ----------
    enforce_hard_cap : bool
        If True, the MIP will reject any trade that pushes a team's total
        post-trade salary above hard_cap_threshold.
    hard_cap_threshold : float
        The NBA hard-cap dollar value (default $165 million for 2024-25).
    enforce_no_trade_clauses : bool
        If True, the SAT layer will add unit clauses forcing any player with
        a no-trade clause to remain on their current team.
    enforce_recently_signed : bool
        If True, the SAT layer will block players who signed within
        recently_signed_months months from being traded.
    recently_signed_months : int
        How many months constitutes "recently signed" (default 12 = 1 year).
    salary_matching_threshold : float
        The multiplier applied to outgoing salary to compute the max incoming
        salary.  NBA CBA default is 1.25 (125 %).
    salary_matching_bonus : float
        Flat dollar amount added on top of the percentage cap.
        NBA CBA default is $100,000.
    min_roster_size : int
        Minimum players a team must have after the trade (default 13).
    max_roster_size : int
        Maximum players a team may have after the trade (default 15, active
        roster limit; full roster is 20 but we model active slots).
    """

    # ── Hard cap ──────────────────────────────────────────────────────────
    enforce_hard_cap: bool = True
    hard_cap_threshold: float = 165_000_000.0   # $165 M

    # ── No-trade clauses ─────────────────────────────────────────────────
    enforce_no_trade_clauses: bool = True

    # ── Recently-signed rule ──────────────────────────────────────────────
    enforce_recently_signed: bool = True
    recently_signed_months: int = 12            # 1 year

    # ── Salary matching ───────────────────────────────────────────────────
    salary_matching_threshold: float = 1.25     # 125 %
    salary_matching_bonus: float = 100_000.0    # $100 K

    # ── Roster size ───────────────────────────────────────────────────────
    min_roster_size: int = 13
    max_roster_size: int = 15

    # ─────────────────────────────────────────────────────────────────────

    def salary_cap(self, outgoing_salary: float) -> float:
        """
        Return the maximum incoming salary allowed for a given outgoing total.

        Formula (NBA CBA): incoming ≤ outgoing × threshold + bonus
        """
        return outgoing_salary * self.salary_matching_threshold + self.salary_matching_bonus

    def describe(self) -> str:
        """Pretty-print the current configuration for display in main.py."""
        lines = [
            "╔═══════════════════════════════════════╗",
            "║      Constraint Configuration          ║",
            "╠═══════════════════════════════════════╣",
            f"║  Hard Cap Enforcement : {'ON ' if self.enforce_hard_cap else 'OFF'}  "
            f"(${self.hard_cap_threshold / 1e6:.0f}M)       ║",
            f"║  No-Trade Clauses     : {'ON ' if self.enforce_no_trade_clauses else 'OFF'}                 ║",
            f"║  Recently Signed Rule : {'ON ' if self.enforce_recently_signed else 'OFF'}  "
            f"({self.recently_signed_months} months)  ║",
            f"║  Salary Match Cap     : {self.salary_matching_threshold*100:.0f}% "
            f"+ ${self.salary_matching_bonus/1e3:.0f}K           ║",
            f"║  Roster Size Limits   : {self.min_roster_size}–{self.max_roster_size} players         ║",
            "╚═══════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ── Quick sanity check if run directly ───────────────────────────────────────
if __name__ == "__main__":
    cfg = ConstraintsConfig()
    print(cfg.describe())

    # Show the salary cap for a $35 M outgoing package
    outgoing = 35_000_000
    cap = cfg.salary_cap(outgoing)
    print(f"\nOutgoing ${outgoing/1e6:.1f}M → max incoming ${cap/1e6:.3f}M")
