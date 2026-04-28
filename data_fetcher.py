"""
Pulls full-league NBA data from Basketball-Reference (all 30 teams) and
joins per-game, advanced, and contract tables into one dataset. Cached to
.cache/nba_<season>.parquet on first run so subsequent runs are instant.

There is a small hardcoded `_DEMO_PLAYERS` pool at the bottom so the CLI
demo runs even with no network on a fresh machine.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlayerRecord:
    player_id: int
    name: str
    team: str                       # 3-letter abbreviation (BBR style)
    position: str                   # "PG", "SG", "SF", "PF", "C"
    age: int
    salary: float                   # current-season salary in USD

    bpm: float = 0.0
    vorp: float = 0.0
    ts_pct: float = 0.55

    has_ntc: bool = False
    months_since_signing: int = 24

    jersey_num: str = ""            # from nba_api (live)
    valuation: float = 0.0

    def is_recently_signed(self, threshold_months: int = 12) -> bool:
        return self.months_since_signing < threshold_months

    def __repr__(self) -> str:
        ntc_tag = " [NTC]" if self.has_ntc else ""
        return (
            f"<{self.name} ({self.team}, {self.position}) "
            f"${self.salary/1e6:.1f}M{ntc_tag}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# Curated set of well-known NTC holders (public knowledge, approximate).
# Real NTC status is a CBA detail not published on bbref; this is a demo proxy.
_KNOWN_NTC = {
    "LeBron James", "Stephen Curry", "Kevin Durant", "James Harden",
    "Bradley Beal", "Damian Lillard", "Russell Westbrook",
}

TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BRK": "Brooklyn Nets",
    "CHO": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


# ─────────────────────────────────────────────────────────────────────────────
# Basketball-Reference scrapers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_salary(raw) -> float:
    if pd.isna(raw):
        return 0.0
    s = str(raw).replace("$", "").replace(",", "").strip()
    try:
        return float(s) if s else 0.0
    except ValueError:
        return 0.0


def _fetch_per_game(season: int) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"
    df = pd.read_html(url)[0]
    df = df[df["Rk"] != "Rk"].copy()           # drop repeated headers
    df = df.dropna(subset=["Player"])
    # Keep one row per player (their aggregated row when traded mid-season is team "2TM"/"3TM")
    df = df.drop_duplicates(subset=["Player"], keep="first")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(25).astype(int)
    return df[["Player", "Age", "Team", "Pos", "PTS", "TRB", "AST"]].rename(
        columns={"Player": "name", "Age": "age", "Team": "team",
                 "Pos": "position", "PTS": "pts", "TRB": "trb", "AST": "ast"}
    )


def _fetch_advanced(season: int) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
    df = pd.read_html(url)[0]
    df = df[df["Rk"] != "Rk"].copy()
    df = df.dropna(subset=["Player"])
    df = df.drop_duplicates(subset=["Player"], keep="first")
    for col in ["TS%", "BPM", "VORP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df[["Player", "TS%", "BPM", "VORP"]].rename(
        columns={"Player": "name", "TS%": "ts_pct", "BPM": "bpm", "VORP": "vorp"}
    )


def _fetch_contracts() -> pd.DataFrame:
    url = "https://www.basketball-reference.com/contracts/players.html"
    df = pd.read_html(url)[0]
    df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]
    df = df[df["Player"].notna() & (df["Player"] != "Player")].copy()
    df = df.drop_duplicates(subset=["Player"], keep="first")
    # Grab the current-season salary column (first "20xx-xx" header)
    season_cols = [c for c in df.columns if isinstance(c, str) and "-" in c and c[:2] == "20"]
    if not season_cols:
        return pd.DataFrame(columns=["name", "salary"])
    current = season_cols[0]
    df["salary"] = df[current].apply(_parse_salary)
    return df[["Player", "salary"]].rename(columns={"Player": "name"})


def _clean_position(pos: str) -> str:
    if not isinstance(pos, str):
        return "SF"
    # Multi-position rows like "PG-SG" → take first
    return pos.split("-")[0].strip() or "SF"


def _normalize_team(t: str) -> str:
    # Bbref uses BRK, CHO, PHO; "2TM"/"3TM" are aggregates for traded players.
    if not isinstance(t, str):
        return ""
    if t.endswith("TM"):
        return ""
    return t


# ─────────────────────────────────────────────────────────────────────────────
# nba_api supplement: live rosters and jersey numbers
# ─────────────────────────────────────────────────────────────────────────────

# nba_api team abbreviations → Basketball-Reference abbreviations where they diverge.
_NBA_TO_BBR = {"BKN": "BRK", "PHX": "PHO", "CHA": "CHO"}

# nba_api season string for a BBR season number (bbref "2026" == NBA "2025-26").
def _nba_season_str(season: int) -> str:
    return f"{season - 1}-{str(season)[-2:]}"


def _normalize_name(n: str) -> str:
    """Strip accents + lowercase for fuzzy name matching across sources."""
    import unicodedata
    if not isinstance(n, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", n)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_only.lower().replace(".", "").replace("'", "").strip()


def _fetch_nba_rosters(season: int) -> pd.DataFrame:
    """Pull current rosters from stats.nba.com via nba_api.

    Returns DataFrame with columns: name, name_key, team (BBR abbr), jersey_num, nba_player_id.
    """
    try:
        from nba_api.stats.endpoints import CommonTeamRoster
        from nba_api.stats.static import teams as nba_teams
    except ImportError:
        print("[data_fetcher] nba_api not installed; skipping live roster merge.")
        return pd.DataFrame(columns=["name", "name_key", "team", "jersey_num", "nba_player_id"])

    season_str = _nba_season_str(season)
    rows = []
    for t in nba_teams.get_teams():
        nba_abbr = t["abbreviation"]
        bbr_abbr = _NBA_TO_BBR.get(nba_abbr, nba_abbr)
        try:
            r = CommonTeamRoster(team_id=t["id"], season=season_str, timeout=15).get_data_frames()[0]
        except Exception as e:
            print(f"[data_fetcher] nba_api failed for {nba_abbr}: {e}")
            continue
        for _, row in r.iterrows():
            rows.append({
                "name": row["PLAYER"],
                "name_key": _normalize_name(row["PLAYER"]),
                "team": bbr_abbr,
                "jersey_num": str(row.get("NUM", "")).strip() or "--",
                "nba_player_id": int(row["PLAYER_ID"]) if pd.notna(row["PLAYER_ID"]) else 0,
            })
        time.sleep(0.6)  # courtesy pause; stats.nba.com rate-limits aggressively
    return pd.DataFrame(rows)


def _build_dataset(season: int) -> pd.DataFrame:
    print(f"[data_fetcher] Fetching per-game…")
    pg = _fetch_per_game(season)
    time.sleep(0.8)  # courtesy pause
    print(f"[data_fetcher] Fetching advanced…")
    adv = _fetch_advanced(season)
    time.sleep(0.8)
    print(f"[data_fetcher] Fetching contracts…")
    contracts = _fetch_contracts()

    df = pg.merge(adv, on="name", how="left")
    df = df.merge(contracts, on="name", how="left")

    df["salary"] = df["salary"].fillna(0.0)
    df["bpm"] = df["bpm"].fillna(0.0)
    df["vorp"] = df["vorp"].fillna(0.0)
    df["ts_pct"] = df["ts_pct"].fillna(0.55)
    df["position"] = df["position"].apply(_clean_position)
    df["team"] = df["team"].apply(_normalize_team)

    # ── nba_api supplement: override team + jersey number with live roster ──
    print(f"[data_fetcher] Fetching live rosters from nba_api (30 teams)…")
    nba_df = _fetch_nba_rosters(season)
    if not nba_df.empty:
        df["name_key"] = df["name"].apply(_normalize_name)
        nba_lookup = nba_df.set_index("name_key")[["team", "jersey_num"]].to_dict("index")

        # Override BBR team with nba_api's current team (handles mid-season trades).
        # Add jersey number. If nba_api doesn't have the player, keep BBR team + no jersey.
        def _apply_live(row):
            info = nba_lookup.get(row["name_key"])
            if info is None:
                return row["team"], ""
            return info["team"], info["jersey_num"]
        live = df.apply(_apply_live, axis=1)
        df["team"] = [t for t, _ in live]
        df["jersey_num"] = [j for _, j in live]
        df = df.drop(columns=["name_key"])

        # Also: add nba_api-only players BBR missed (rookies signed late, two-ways)
        existing = set(df["name"].apply(_normalize_name))
        extras = nba_df[~nba_df["name_key"].isin(existing)]
        if not extras.empty:
            extras = extras.rename(columns={})
            for _, r in extras.iterrows():
                df = pd.concat([df, pd.DataFrame([{
                    "name": r["name"], "team": r["team"], "position": "SF",
                    "age": 25, "pts": 0.0, "trb": 0.0, "ast": 0.0,
                    "ts_pct": 0.55, "bpm": 0.0, "vorp": 0.0,
                    "salary": 0.0, "jersey_num": r["jersey_num"],
                }])], ignore_index=True)
    else:
        df["jersey_num"] = ""

    # Filter to current-roster players: require a non-empty team code
    df = df[df["team"].isin(TEAM_NAMES.keys())].copy()

    # Assign stable pseudo IDs
    df = df.reset_index(drop=True)
    df["player_id"] = df.index + 1_000_000

    # NTC heuristic
    df["has_ntc"] = df["name"].isin(_KNOWN_NTC)

    # Months-since-signing heuristic: spread deterministically by name hash
    df["months_since_signing"] = df["name"].apply(
        lambda n: 6 + (abs(hash(n)) % 30)   # 6-35 months
    )

    return df


def _cache_path(season: int) -> Path:
    return CACHE_DIR / f"nba_{season}.parquet"


def load_dataset(season: int = 2026, force_refresh: bool = False) -> pd.DataFrame:
    """Load (and cache) the joined league dataset."""
    cache = _cache_path(season)
    if cache.exists() and not force_refresh:
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass

    try:
        df = _build_dataset(season)
        try:
            df.to_parquet(cache, index=False)
        except Exception as e:
            print(f"[data_fetcher] Could not cache parquet ({e}); continuing.")
        return df
    except Exception as e:
        print(f"[data_fetcher] Network fetch failed ({e}); falling back to demo data.")
        return _demo_dataframe()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_record(row) -> PlayerRecord:
    return PlayerRecord(
        player_id=int(row["player_id"]),
        name=str(row["name"]),
        team=str(row["team"]),
        position=str(row["position"]),
        age=int(row["age"]),
        salary=float(row["salary"]),
        bpm=float(row["bpm"]),
        vorp=float(row["vorp"]),
        ts_pct=float(row["ts_pct"]),
        has_ntc=bool(row["has_ntc"]),
        months_since_signing=int(row["months_since_signing"]),
        jersey_num=str(row.get("jersey_num", "") or ""),
    )


def get_all_players(season: int = 2026, force_refresh: bool = False) -> list[PlayerRecord]:
    df = load_dataset(season, force_refresh=force_refresh)
    return [_row_to_record(r) for _, r in df.iterrows()]


def get_team_roster(abbr: str, season: int = 2026) -> list[PlayerRecord]:
    df = load_dataset(season)
    sub = df[df["team"] == abbr]
    return [_row_to_record(r) for _, r in sub.iterrows()]


def get_all_teams(season: int = 2026) -> list[tuple[str, str, int]]:
    """Return [(abbr, display_name, roster_size), ...] sorted by display name."""
    df = load_dataset(season)
    counts = df.groupby("team").size().to_dict()
    out = [(abbr, TEAM_NAMES[abbr], counts.get(abbr, 0)) for abbr in TEAM_NAMES]
    out = [t for t in out if t[2] > 0]
    return sorted(out, key=lambda x: x[1])


# ─────────────────────────────────────────────────────────────────────────────
# Demo fallback (small, so the app still works offline on first run)
# ─────────────────────────────────────────────────────────────────────────────

def _demo_dataframe() -> pd.DataFrame:
    demo = list(get_demo_players().values())
    rows = []
    for p in demo:
        rows.append({
            "player_id": p.player_id, "name": p.name, "team": p.team,
            "position": p.position, "age": p.age, "salary": p.salary,
            "bpm": p.bpm, "vorp": p.vorp, "ts_pct": p.ts_pct,
            "has_ntc": p.has_ntc, "months_since_signing": p.months_since_signing,
            "pts": 0.0, "trb": 0.0, "ast": 0.0,
        })
    return pd.DataFrame(rows)


_DEMO_FALLBACK: dict[str, PlayerRecord] = {
    "anthony_davis": PlayerRecord(203076, "Anthony Davis", "LAL", "C", 31, 40_600_080, 6.1, 4.2, 0.623, False, 24),
    "lebron_james":  PlayerRecord(  2544, "LeBron James",  "LAL", "SF", 39, 47_607_350, 4.2, 2.8, 0.601, True, 10),
    "austin_reaves": PlayerRecord(1630559, "Austin Reaves", "LAL", "SG", 26, 12_000_000, 1.2, 1.4, 0.601, False, 18),
    "dlo":           PlayerRecord(1626164, "D'Angelo Russell", "LAL", "PG", 28, 18_000_000, -0.5, 0.6, 0.556, False, 20),
    "rui":           PlayerRecord(1629744, "Rui Hachimura", "LAL", "PF", 26, 17_000_000, 0.2, 0.5, 0.592, False, 16),
    "vando":         PlayerRecord(1629714, "Jarred Vanderbilt", "LAL", "PF", 25, 13_000_000, 0.4, 0.5, 0.550, False, 18),
    "gabe":          PlayerRecord(1628964, "Gabe Vincent", "LAL", "PG", 28, 11_000_000, -1.1, 0.1, 0.511, False, 15),
    "taurean":       PlayerRecord(1627884, "Taurean Prince", "LAL", "SF", 30, 4_500_000, -0.3, 0.3, 0.578, False, 20),
    "max_christie":  PlayerRecord(1631217, "Max Christie", "LAL", "SG", 21, 2_000_000, -1.5, 0.0, 0.520, False, 20),
    "alex_len":      PlayerRecord(203458, "Alex Len", "LAL", "C", 31, 1_800_000, -1.8, -0.1, 0.560, False, 18),
    "cam":           PlayerRecord(1631094, "Cam Reddish", "LAL", "SF", 25, 3_500_000, -0.9, 0.1, 0.541, False, 14),
    "christian_wood":PlayerRecord(1626174, "Christian Wood", "LAL", "C", 28, 2_700_000, 0.8, 0.4, 0.610, False, 22),
    "jalen_hood":    PlayerRecord(1631218, "Jalen Hood-Schifino", "LAL", "PG", 22, 4_000_000, -2.0, -0.1, 0.480, False, 11),

    "ben_simmons":   PlayerRecord(1627732, "Ben Simmons", "BRK", "PG", 27, 37_893_408, 1.3, 0.4, 0.601, False, 30),
    "nic_claxton":   PlayerRecord(1629651, "Nic Claxton", "BRK", "C", 25, 20_000_000, 1.9, 1.5, 0.680, False, 8),
    "mikal_bridges": PlayerRecord(1628969, "Mikal Bridges", "BRK", "SF", 27, 23_000_000, 1.0, 1.2, 0.588, False, 24),
    "cam_johnson":   PlayerRecord(1629661, "Cameron Johnson", "BRK", "SF", 28, 22_000_000, 0.6, 0.9, 0.598, False, 9),
    "dorian_finney": PlayerRecord(1629628, "Dorian Finney-Smith", "BRK", "SF", 30, 15_000_000, -0.1, 0.4, 0.558, False, 22),
    "spencer_dinwiddie":PlayerRecord(203915, "Spencer Dinwiddie", "BRK", "PG", 31, 8_500_000, -0.4, 0.3, 0.542, False, 16),
    "royce_oneill":  PlayerRecord(203109, "Royce O'Neale", "BRK", "SF", 30, 9_000_000, 0.1, 0.4, 0.531, False, 19),
    "lonnie_walker": PlayerRecord(1629018, "Lonnie Walker IV", "BRK", "SG", 25, 2_000_000, -1.2, 0.0, 0.520, False, 14),
    "day_ron_sharpe":PlayerRecord(1630549, "Day'Ron Sharpe", "BRK", "C", 22, 4_100_000, -1.0, 0.0, 0.560, False, 20),
    "kyshawn_george":PlayerRecord(1641754, "Kyshawn George", "BRK", "SF", 21, 3_600_000, -2.0, -0.1, 0.500, False, 10),
    "noah_clowney":  PlayerRecord(1641755, "Noah Clowney", "BRK", "PF", 20, 3_400_000, -2.2, -0.1, 0.490, False, 20),
    "keon_johnson":  PlayerRecord(1630529, "Keon Johnson", "BRK", "SG", 22, 2_100_000, -1.8, -0.1, 0.495, False, 18),
    "darius_bazley": PlayerRecord(1629647, "Darius Bazley", "BRK", "PF", 24, 1_900_000, -1.5, 0.0, 0.515, False, 16),
}


def get_demo_players() -> dict[str, PlayerRecord]:
    return dict(_DEMO_FALLBACK)


def get_lakers_roster() -> list[PlayerRecord]:
    return [p for p in _DEMO_FALLBACK.values() if p.team == "LAL"]


def get_nets_roster() -> list[PlayerRecord]:
    return [p for p in _DEMO_FALLBACK.values() if p.team == "BRK"]


# Back-compat stub (old main.py imports this)
def get_team_roster_live(team_abbr: str):
    try:
        roster = get_team_roster(team_abbr)
        return roster if roster else None
    except Exception:
        return None


if __name__ == "__main__":
    players = get_all_players()
    print(f"\nTotal players loaded: {len(players)}")
    teams = get_all_teams()
    print(f"Teams: {len(teams)}")
    for abbr, name, n in teams[:5]:
        print(f"  {abbr}  {name:30s}  {n} players")
