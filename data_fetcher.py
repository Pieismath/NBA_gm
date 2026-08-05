"""
Pulls full-league NBA data from Basketball-Reference (all 30 teams) and
joins contract, per-game, and advanced tables into one dataset. Cached to
.cache/nba_<season>.parquet, with a freshness window so the cache never
silently serves a stale league.

The *roster spine* is the Basketball-Reference contracts page. That page
lists every player under contract with the team they are currently on and
their salary for the current league year, so it reflects trades and free
agency the moment BBR posts them. Per-game and advanced stats are joined
on top from the most recent season that has actually been played — during
the offseason that is last season, because the upcoming one has no box
scores yet.

There is a small hardcoded fallback pool at the bottom (see
`get_demo_players`) so the CLI demo runs even with no network.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
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

# How long a cached parquet stays trusted before we refetch. Rosters move
# daily in the offseason, so a shipped cache is a seed, not a source of truth.
CACHE_MAX_AGE_HOURS = 24.0

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
# Season detection
#
# A season is identified by its *ending* year, matching Basketball-Reference:
# the 2026-27 season is "2027". Two different seasons matter here and they are
# not the same during the offseason:
#
#   current_season()      which league year the rosters and salaries belong to
#   latest_stats_season() which season actually has box scores to join on
#
# In August 2026 those are 2027 and 2026 respectively: contracts are for
# 2026-27, but the only games played are 2025-26's.
# ─────────────────────────────────────────────────────────────────────────────

def current_season(today: Optional[date] = None) -> int:
    """Season whose rosters and salaries are in force right now.

    The NBA league year flips on July 1, when new contracts and free-agent
    signings take effect, so July onward belongs to the next season.
    """
    d = today or date.today()
    return d.year + 1 if d.month >= 7 else d.year


def latest_stats_season(today: Optional[date] = None) -> int:
    """Most recent season that has been played, so has per-game/advanced data.

    Before opening night (mid-to-late October) the current season has no box
    scores, so the previous one is the newest with anything to join.
    """
    d = today or date.today()
    season = current_season(d)
    tipped_off = d.month >= 11 or d.month <= 6 or (d.month == 10 and d.day >= 20)
    return season if tipped_off else season - 1


def season_label(season: int) -> str:
    """BBR season number to the human label: 2027 -> '2026-27'."""
    return f"{season - 1}-{str(season)[-2:]}"


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


def _dedupe_players(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player, preferring the full-season total for traded players.

    A player dealt mid-season gets one row per team plus a combined "2TM"/"3TM"
    row that BBR lists first. That combined row is the one we want — it is the
    player's whole season — so sort the aggregates to the front and keep those.
    """
    df = df.copy()
    df["_is_total"] = df["Team"].astype(str).str.endswith("TM")
    df = df.sort_values("_is_total", ascending=False, kind="stable")
    df = df.drop_duplicates(subset=["Player"], keep="first")
    return df.drop(columns=["_is_total"])


def _fetch_per_game(season: int) -> pd.DataFrame:
    """Per-game box score stats. Team is deliberately dropped: it is where the
    player played *last* season, which is exactly the stale-roster trap. The
    contracts page supplies the current team instead."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"
    df = pd.read_html(url)[0]
    df = df[df["Rk"] != "Rk"].copy()           # drop repeated headers
    df = df.dropna(subset=["Player"])
    df = _dedupe_players(df)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(25).astype(int)
    return df[["Player", "Age", "Pos", "PTS", "TRB", "AST"]].rename(
        columns={"Player": "name", "Age": "age",
                 "Pos": "position", "PTS": "pts", "TRB": "trb", "AST": "ast"}
    )


def _fetch_advanced(season: int) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
    df = pd.read_html(url)[0]
    df = df[df["Rk"] != "Rk"].copy()
    df = df.dropna(subset=["Player"])
    df = _dedupe_players(df)
    for col in ["TS%", "BPM", "VORP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df[["Player", "TS%", "BPM", "VORP"]].rename(
        columns={"Player": "name", "TS%": "ts_pct", "BPM": "bpm", "VORP": "vorp"}
    )


def _fetch_contracts(season: int) -> pd.DataFrame:
    """Every player under contract, with their current team and salary.

    This is the roster spine. The `Tm` column is BBR's live team assignment,
    so a player traded yesterday already shows up on their new team here.
    """
    url = "https://www.basketball-reference.com/contracts/players.html"
    df = pd.read_html(url)[0]
    df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]
    df = df[df["Player"].notna() & (df["Player"] != "Player")].copy()
    df = df.drop_duplicates(subset=["Player"], keep="first")

    # Prefer the column for the season we asked for; fall back to the leftmost
    # "20xx-xx" header, which is whatever BBR currently treats as this year.
    season_cols = [c for c in df.columns if isinstance(c, str) and "-" in c and c[:2] == "20"]
    if not season_cols:
        return pd.DataFrame(columns=["name", "team", "salary"])
    wanted = season_label(season)
    current = wanted if wanted in season_cols else season_cols[0]
    if current != wanted:
        print(f"[data_fetcher] Contracts page has no {wanted} column; using {current}.")

    df["salary"] = df[current].apply(_parse_salary)
    df["team"] = df["Tm"].apply(_normalize_team) if "Tm" in df.columns else ""
    return df[["Player", "team", "salary"]].rename(columns={"Player": "name"})


# nba_api reports coarse positions ("G", "F", "C", "G-F"); the valuation model
# and positional-need vectors are keyed on the five BBR slots.
_COARSE_POSITION = {"G": "SG", "F": "SF", "C": "C"}


def _clean_position(pos: str) -> str:
    if not isinstance(pos, str) or not pos.strip():
        return "SF"
    # Multi-position rows like "PG-SG" or "G-F" → take the first listed
    first = pos.split("-")[0].strip()
    return _COARSE_POSITION.get(first, first) or "SF"


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

# stats.nba.com blocks datacenter IPs, so on a host like Streamlit Cloud every
# request hangs until it times out. At 15s across 30 teams that is 7.5 minutes
# of a user watching a spinner for data that is never coming.
_NBA_API_TIMEOUT_S = 8
_NBA_API_GIVE_UP_AFTER = 3

# nba_api season string for a BBR season number (bbref "2027" == NBA "2026-27").
def _nba_season_str(season: int) -> str:
    return season_label(season)


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

    Returns DataFrame with columns: name, name_key, team (BBR abbr), jersey_num,
    position, age, nba_player_id.

    Best-effort: stats.nba.com blocks most datacenter IPs, so this returns empty
    on Streamlit Cloud. Callers must treat it as a supplement, never a source.
    """
    empty = pd.DataFrame(columns=["name", "name_key", "team", "jersey_num",
                                  "position", "age", "nba_player_id"])
    try:
        from nba_api.stats.endpoints import CommonTeamRoster
        from nba_api.stats.static import teams as nba_teams
    except ImportError:
        print("[data_fetcher] nba_api not installed; skipping live roster merge.")
        return empty

    season_str = _nba_season_str(season)
    rows = []
    failures = 0
    consecutive = 0
    for t in nba_teams.get_teams():
        nba_abbr = t["abbreviation"]
        bbr_abbr = _NBA_TO_BBR.get(nba_abbr, nba_abbr)
        try:
            r = CommonTeamRoster(team_id=t["id"], season=season_str,
                                 timeout=_NBA_API_TIMEOUT_S).get_data_frames()[0]
            consecutive = 0
        except Exception as e:
            failures += 1
            consecutive += 1
            # One line per team is noise when the whole endpoint is blocked.
            if failures == 1:
                print(f"[data_fetcher] nba_api failed for {nba_abbr}: {e}")
            # When the host is blocking us outright, every team will time out.
            # Waiting through all 30 costs minutes and buys nothing, so give up
            # once it is clear this is not a one-off — jersey numbers are the
            # only thing at stake.
            if consecutive >= _NBA_API_GIVE_UP_AFTER:
                print(f"[data_fetcher] nba_api unreachable after "
                      f"{consecutive} straight failures; skipping the rest.")
                break
            continue
        for _, row in r.iterrows():
            rows.append({
                "name": row["PLAYER"],
                "name_key": _normalize_name(row["PLAYER"]),
                "team": bbr_abbr,
                "jersey_num": str(row.get("NUM", "")).strip() or "--",
                "position": str(row.get("POSITION", "") or ""),
                "age": int(row["AGE"]) if pd.notna(row.get("AGE")) else 0,
                "nba_player_id": int(row["PLAYER_ID"]) if pd.notna(row["PLAYER_ID"]) else 0,
            })
        time.sleep(0.6)  # courtesy pause; stats.nba.com rate-limits aggressively
    if failures:
        print(f"[data_fetcher] nba_api unreachable for {failures}/30 teams.")
    return pd.DataFrame(rows) if rows else empty


def _stable_months_since_signing(name: str) -> int:
    """Deterministic 6-35 month spread, used as a stand-in for real signing dates.

    Python's built-in hash() is salted per process (PYTHONHASHSEED), so using it
    here made a player "recently signed" — and therefore untradeable — on one run
    and freely tradeable on the next. md5 keeps the flag reproducible across runs
    and machines, which the benchmarks in the report depend on.
    """
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return 6 + (int(digest[:8], 16) % 30)


def _build_dataset(season: int, stats_season: Optional[int] = None) -> pd.DataFrame:
    """Join live contracts (who is on which team, for how much) with the most
    recently played season's stats."""
    if stats_season is None:
        stats_season = latest_stats_season()

    print(f"[data_fetcher] Building {season_label(season)} rosters "
          f"with {season_label(stats_season)} stats…")

    print("[data_fetcher] Fetching contracts (live rosters + salaries)…")
    contracts = _fetch_contracts(season)
    if contracts.empty:
        raise RuntimeError("contracts page returned no rows")
    time.sleep(0.8)  # courtesy pause; BBR rate-limits

    print("[data_fetcher] Fetching per-game…")
    pg = _fetch_per_game(stats_season)
    time.sleep(0.8)
    print("[data_fetcher] Fetching advanced…")
    adv = _fetch_advanced(stats_season)

    stats = pg.merge(adv, on="name", how="left")

    # Contracts drive the join: a player on a roster with no stats (rookie,
    # returning from injury) still belongs on the team. A player with stats but
    # no contract has left the league and should not appear.
    df = contracts.merge(stats, on="name", how="left")

    # Stats are from last season, so everyone is a year older now.
    seasons_elapsed = max(0, season - stats_season)
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(24) + seasons_elapsed
    df["age"] = df["age"].astype(int)

    df["bpm"] = df["bpm"].fillna(0.0)
    df["vorp"] = df["vorp"].fillna(0.0)
    df["ts_pct"] = df["ts_pct"].fillna(0.55)
    for col in ("pts", "trb", "ast"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["salary"] = df["salary"].fillna(0.0)
    df["position"] = df["position"].apply(_clean_position)

    # ── nba_api supplement: jersey numbers, plus anyone contracts missed ──
    # Best-effort only. stats.nba.com blocks most cloud IPs, so Streamlit Cloud
    # normally gets nothing here — the dataset must stand up without it.
    print("[data_fetcher] Fetching live rosters from nba_api (30 teams)…")
    nba_df = _fetch_nba_rosters(season)
    if not nba_df.empty:
        df["name_key"] = df["name"].apply(_normalize_name)
        jersey = nba_df.set_index("name_key")["jersey_num"].to_dict()
        df["jersey_num"] = df["name_key"].map(jersey).fillna("")

        # Players nba_api lists but contracts does not are camp invites and
        # Exhibit-10 deals with no guaranteed money. They are deliberately left
        # out: a $0 player can satisfy salary matching for free, which would let
        # the optimizer "balance" any trade with filler that does not exist.
        extras = (~nba_df["name_key"].isin(set(df["name_key"]))).sum()
        df = df.drop(columns=["name_key"])
        if extras:
            print(f"[data_fetcher] Skipped {extras} nba_api players with no contract.")
    else:
        print("[data_fetcher] nba_api unavailable; continuing with contracts data only.")
        df["jersey_num"] = ""

    # Filter to players actually rostered: require a real team code
    df = df[df["team"].isin(TEAM_NAMES.keys())].copy()
    if df.empty:
        raise RuntimeError("no rostered players survived the join")

    # Assign stable pseudo IDs
    df = df.reset_index(drop=True)
    df["player_id"] = df.index + 1_000_000

    # NTC heuristic
    df["has_ntc"] = df["name"].isin(_KNOWN_NTC)
    df["months_since_signing"] = df["name"].apply(_stable_months_since_signing)

    # Provenance, so the UI can show how fresh this is and what it is made of.
    df["season"] = season
    df["stats_season"] = stats_season
    df["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"[data_fetcher] Built {len(df)} players across {df['team'].nunique()} teams.")
    return df


def _cache_path(season: int) -> Path:
    return CACHE_DIR / f"nba_{season}.parquet"


def cache_age_hours(season: int) -> Optional[float]:
    """Age of the cached dataset in hours, or None if there is no usable cache."""
    cache = _cache_path(season)
    if not cache.exists():
        return None
    try:
        stamp = pd.read_parquet(cache, columns=["fetched_at"])["fetched_at"].iloc[0]
        fetched = datetime.fromisoformat(str(stamp))
    except Exception:
        # Cache predates the fetched_at stamp; treat it as indefinitely old so
        # it gets refreshed rather than trusted forever.
        return float("inf")
    delta = datetime.now(timezone.utc) - fetched
    return delta.total_seconds() / 3600.0


def load_dataset(
    season: Optional[int] = None,
    force_refresh: bool = False,
    max_age_hours: float = CACHE_MAX_AGE_HOURS,
) -> pd.DataFrame:
    """Load the joined league dataset for `season`, refetching a stale cache.

    Defaults to whatever season is in force today rather than a pinned year.
    A cache older than `max_age_hours` is refetched; if that refetch fails the
    stale copy is still returned, because old rosters beat no rosters.
    """
    if season is None:
        season = current_season()

    cache = _cache_path(season)
    cached: Optional[pd.DataFrame] = None
    if cache.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(cache)
            age = cache_age_hours(season)
            if age is not None and age <= max_age_hours:
                return cached
            print(f"[data_fetcher] Cache for {season_label(season)} is stale; refreshing.")
        except Exception:
            cached = None

    try:
        df = _build_dataset(season)
        try:
            df.to_parquet(cache, index=False)
        except Exception as e:
            print(f"[data_fetcher] Could not cache parquet ({e}); continuing.")
        return df
    except Exception as e:
        print(f"[data_fetcher] Live fetch failed ({e}).")
        if cached is not None:
            print("[data_fetcher] Serving the stale cache instead.")
            return cached
        print("[data_fetcher] Falling back to the offline demo pool.")
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


def get_all_players(season: Optional[int] = None, force_refresh: bool = False) -> list[PlayerRecord]:
    df = load_dataset(season, force_refresh=force_refresh)
    return [_row_to_record(r) for _, r in df.iterrows()]


def get_team_roster(abbr: str, season: Optional[int] = None) -> list[PlayerRecord]:
    df = load_dataset(season)
    sub = df[df["team"] == abbr]
    return [_row_to_record(r) for _, r in sub.iterrows()]


def get_all_teams(season: Optional[int] = None) -> list[tuple[str, str, int]]:
    """Return [(abbr, display_name, roster_size), ...] sorted by display name."""
    df = load_dataset(season)
    counts = df.groupby("team").size().to_dict()
    out = [(abbr, TEAM_NAMES[abbr], counts.get(abbr, 0)) for abbr in TEAM_NAMES]
    out = [t for t in out if t[2] > 0]
    return sorted(out, key=lambda x: x[1])


def dataset_provenance(df: pd.DataFrame) -> dict:
    """Where this dataset came from, for display in the UI.

    Returns season/stats_season labels, the UTC fetch timestamp, and whether
    the rows are live or the offline demo pool.
    """
    def _first(col, default=None):
        if col not in df.columns or df.empty:
            return default
        val = df[col].iloc[0]
        return default if pd.isna(val) else val

    season = _first("season")
    stats = _first("stats_season")
    fetched = _first("fetched_at")
    age_hours = None
    if fetched:
        try:
            age_hours = (datetime.now(timezone.utc)
                         - datetime.fromisoformat(str(fetched))).total_seconds() / 3600.0
        except Exception:
            age_hours = None
    return {
        "season": int(season) if season else None,
        "season_label": season_label(int(season)) if season else "unknown",
        "stats_season_label": season_label(int(stats)) if stats else "unknown",
        "fetched_at": str(fetched) if fetched else None,
        "age_hours": age_hours,
        "is_live": season is not None,
    }


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
            "pts": 0.0, "trb": 0.0, "ast": 0.0, "jersey_num": "",
        })
    # No season/fetched_at columns: dataset_provenance() reports is_live=False
    # off their absence, so the UI can say plainly that this is not live data.
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
    import argparse

    ap = argparse.ArgumentParser(description="Fetch and inspect the league dataset.")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and refetch")
    ap.add_argument("--season", type=int, default=None, help="season ending year, e.g. 2027")
    args = ap.parse_args()

    season = args.season or current_season()
    print(f"Current season : {season_label(season)}")
    print(f"Stats season   : {season_label(latest_stats_season())}")

    df = load_dataset(season, force_refresh=args.refresh)
    prov = dataset_provenance(df)
    print(f"\nProvenance     : {prov}")
    print(f"Total players  : {len(df)}")

    teams = get_all_teams(season)
    print(f"Teams          : {len(teams)}")
    for abbr, name, n in teams[:5]:
        print(f"  {abbr}  {name:30s}  {n} players")
