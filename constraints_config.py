"""
All the user-toggleable CBA constraints in one dataclass. Every layer
(SAT, MIP, UI) takes a ConstraintsConfig argument so toggles flow through
without being hardcoded anywhere. Flip any of them before running.
"""

from dataclasses import dataclass, field


@dataclass
class ConstraintsConfig:
    """
    Which CBA constraints are active and what their thresholds are.

    Hard cap: post-trade total salary must stay under hard_cap_threshold
    when enforce_hard_cap is on. Default threshold is $165M (2024-25).

    NTC and recently-signed: when enforce_no_trade_clauses or
    enforce_recently_signed is on, the SAT layer adds unit clauses forcing
    those players to traded=False. recently_signed_months sets the window
    (12 by default).

    Salary matching: incoming <= outgoing * salary_matching_threshold +
    salary_matching_bonus per team. CBA default is 1.25 and $100K.

    Roster size: min_roster_size and max_roster_size bound each team's
    roster after the trade. Defaults are 13 and 20, matching the full NBA
    roster (15 active + 2 two-way + hardship/exhibit slots).
    """

    # Hard cap
    enforce_hard_cap: bool = True
    hard_cap_threshold: float = 165_000_000.0   # $165M

    # NTC / recently-signed
    enforce_no_trade_clauses: bool = True
    enforce_recently_signed: bool = True
    recently_signed_months: int = 12

    # Salary matching: incoming <= outgoing * threshold + bonus
    salary_matching_threshold: float = 1.25
    salary_matching_bonus: float = 100_000.0

    # Roster size after trade
    min_roster_size: int = 13
    max_roster_size: int = 20

    def salary_cap(self, outgoing_salary: float) -> float:
        """Max incoming salary for a given outgoing total (CBA formula)."""
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
            f"║  Roster Size Limits   : {self.min_roster_size}-{self.max_roster_size} players         ║",
            "╚═══════════════════════════════════════╝",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    cfg = ConstraintsConfig()
    print(cfg.describe())

    # quick sanity check on the salary cap for a $35M outgoing package
    outgoing = 35_000_000
    cap = cfg.salary_cap(outgoing)
    print(f"\nOutgoing ${outgoing/1e6:.1f}M -> max incoming ${cap/1e6:.3f}M")
