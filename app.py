"""
app.py: GM Mode
================
Pared-down, pixel-art-flavored Streamlit interface for the three-layer NBA
trade optimizer. Each player gets a tiny animated pixel sprite colored by
their team. Full-league data from Basketball-Reference (all 30 teams, ~500
players), cached to disk after first load.

Run:
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
        -m streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import List

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))

from constraints_config import ConstraintsConfig
from data_fetcher import PlayerRecord, get_all_teams, load_dataset
from sat_layer import SATFeasibilityChecker
from valuation_model import PlayerValuationModel, TeamContext
from mip_layer import solve_both


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="GM Mode", layout="wide", initial_sidebar_state="expanded")

# Minimal palette
INK     = "#111318"
PAPER   = "#FBF7EF"
PAPER_2 = "#F3EDDF"
RULE    = "#E4DCC8"
MUTED   = "#8A8576"
ORANGE  = "#E86A1A"
COURT   = "#1F8E5A"
RED     = "#B43A3A"


# ─────────────────────────────────────────────────────────────────────────────
# Team colors (for pixel sprites)
# ─────────────────────────────────────────────────────────────────────────────

TEAM_COLORS = {
    "ATL": ("#E03A3E", "#C1D32F"), "BOS": ("#007A33", "#BA9653"),
    "BRK": ("#111111", "#FFFFFF"), "CHO": ("#1D1160", "#00788C"),
    "CHI": ("#CE1141", "#111111"), "CLE": ("#860038", "#041E42"),
    "DAL": ("#00538C", "#002B5E"), "DEN": ("#0E2240", "#FEC524"),
    "DET": ("#C8102E", "#1D42BA"), "GSW": ("#1D428A", "#FFC72C"),
    "HOU": ("#CE1141", "#111111"), "IND": ("#002D62", "#FDBB30"),
    "LAC": ("#C8102E", "#1D428A"), "LAL": ("#552583", "#FDB927"),
    "MEM": ("#5D76A9", "#12173F"), "MIA": ("#98002E", "#F9A01B"),
    "MIL": ("#00471B", "#EEE1C6"), "MIN": ("#0C2340", "#236192"),
    "NOP": ("#0C2340", "#C8102E"), "NYK": ("#006BB6", "#F58426"),
    "OKC": ("#007AC1", "#EF3B24"), "ORL": ("#0077C0", "#C4CED4"),
    "PHI": ("#006BB6", "#ED174C"), "PHO": ("#1D1160", "#E56020"),
    "POR": ("#E03A3E", "#111111"), "SAC": ("#5A2D81", "#63727A"),
    "SAS": ("#C4CED4", "#111111"), "TOR": ("#CE1141", "#111111"),
    "UTA": ("#002B5C", "#00471B"), "WAS": ("#002B5C", "#E31837"),
}

SKIN_TONES = ["#F1C8A6", "#E0A878", "#C18B5E", "#8F5A3B", "#60381F"]
HAIR_TONES = ["#1A1410", "#2B1B12", "#3C2416", "#5B3A20", "#3A2E25"]


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-sprite generator (procedural, per-player)
# ─────────────────────────────────────────────────────────────────────────────

# 12-wide × 14-tall sprite; '.' is transparent.
#   X = hair outline   H = hair
#   S = skin           J = jersey
#   P = shorts         K = shoes
_PLAYER_SPRITE = [
    "....XXXX....",
    "...XHHHHX...",
    "...XHHHHX...",
    "....XSSX....",
    "....SSSS....",
    "..JJJJJJJJ..",
    ".JJJJJJJJJJ.",
    ".SJJJJJJJJS.",
    "..JJJJJJJJ..",
    "..JJJJJJJJ..",
    "..PPPPPPPP..",
    "..PP....PP..",
    "..SS....SS..",
    "..KK....KK..",
]

# 12×14 GM in a suit: white shirt, orange tie, navy jacket, clipboard.
#   X/H = hair   S = skin   J = suit   W = shirt   T = tie
#   P = pants (same tone)   K = shoes   C = clipboard   B = clipboard band
_GM_SPRITE = [
    "....XXXX....",
    "...XHHHHX...",
    "...XHHHHX...",
    "....XSSX....",
    "...WSSSSW...",
    "..JWWTTWWJ..",
    ".JJJWTTWJJJ.",
    ".SJJWTTWJJS.",
    "..JJJTTJJJCC",
    "..JJJJJJJJCC",
    "..JJJJJJJJ..",
    "..PPPPPPPP..",
    "..PP....PP..",
    "..KK....KK..",
]

_SPR_W, _SPR_H = 12, 14
_BALL_Y_TOP    = _SPR_H
_SPR_TOTAL_H   = _SPR_H + 4


def pixel_sprite(player: PlayerRecord, px: int = 3) -> str:
    """Render an SVG pixel sprite for a single player, team-colored, with a dribbling ball."""
    primary, secondary = TEAM_COLORS.get(player.team, ("#333333", "#DDDDDD"))
    h = hash(player.name)
    skin  = SKIN_TONES[h % len(SKIN_TONES)]
    hair  = HAIR_TONES[(h >> 3) % len(HAIR_TONES)]
    color_map = {
        "X": hair, "H": hair, "S": skin,
        "J": primary, "P": secondary, "K": INK,
    }

    rects = []
    for y, row in enumerate(_PLAYER_SPRITE):
        for x, c in enumerate(row):
            if c == ".":
                continue
            rects.append(
                f'<rect x="{x*px}" y="{y*px}" width="{px}" height="{px}" '
                f'fill="{color_map[c]}" shape-rendering="crispEdges"/>'
            )

    ball_x = 5 * px
    ball_y = _BALL_Y_TOP * px + 2
    ball_size = 2 * px
    ball_svg = (
        f'<g class="gm-ball">'
        f'  <rect x="{ball_x}" y="{ball_y}" width="{ball_size}" height="{ball_size}" fill="{ORANGE}" shape-rendering="crispEdges"/>'
        f'  <rect x="{ball_x-px//2 if px>1 else ball_x}" y="{ball_y+px//2}" width="{ball_size+px}" height="{px}" fill="{ORANGE}" shape-rendering="crispEdges"/>'
        f'  <rect x="{ball_x}" y="{ball_y+ball_size-1}" width="{ball_size}" height="1" fill="{INK}" opacity="0.25"/>'
        f'</g>'
    )

    w = _SPR_W * px
    h_total = _SPR_TOTAL_H * px
    delay_ms = (hash(player.name) % 800)
    return (
        f'<svg viewBox="0 0 {w} {h_total}" width="{w}" height="{h_total}" '
        f'style="image-rendering: pixelated; image-rendering: crisp-edges; '
        f'animation-delay: {delay_ms}ms;" class="gm-sprite">'
        f'{"".join(rects)}{ball_svg}'
        f'</svg>'
    )


# 10-wide × 11-tall jersey silhouette: J=primary, S=secondary (trim), W=white plate
_JERSEY = [
    "..JJ..JJ..",
    ".JJJSSJJJ.",
    "JJJSSSSJJJ",
    "JJSSSSSSJJ",
    "JJWWWWWWJJ",
    "JJWWWWWWJJ",
    "JJWWWWWWJJ",
    "JJSSSSSSJJ",
    "JJJJJJJJJJ",
    ".JJJJJJJJ.",
    "..JJJJJJ..",
]


def team_logo(abbr: str, px: int = 6) -> str:
    """Render a pixel jersey in the team's colors with the abbr on the chest."""
    primary, secondary = TEAM_COLORS.get(abbr, ("#333333", "#DDDDDD"))
    color_map = {"J": primary, "S": secondary, "W": "#F8F4EA"}
    rects = []
    for y, row in enumerate(_JERSEY):
        for x, c in enumerate(row):
            if c == ".":
                continue
            rects.append(
                f'<rect x="{x*px}" y="{y*px}" width="{px}" height="{px}" '
                f'fill="{color_map[c]}" shape-rendering="crispEdges"/>'
            )
    w = 10 * px
    h = 11 * px
    # Text abbr on the white chest plate: Silkscreen, 3-letter
    text_x = w / 2
    text_y = 6 * px + px  # center of white plate
    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'style="image-rendering: pixelated; image-rendering: crisp-edges;" class="gm-jersey">'
        f'{"".join(rects)}'
        f'<text x="{text_x}" y="{text_y}" text-anchor="middle" '
        f'font-family="Silkscreen, monospace" font-size="{px*1.6:.1f}" '
        f'font-weight="700" fill="{primary}">{abbr}</text>'
        f'</svg>'
    )


def gm_sprite(px: int = 4) -> str:
    """The GM mascot: a suited character holding a clipboard. No basketball."""
    suit  = INK            # black suit
    shirt = "#FFFFFF"      # crisp white shirt
    tie   = ORANGE
    skin  = SKIN_TONES[2]
    hair  = HAIR_TONES[0]
    clip  = "#C4B087"      # clipboard wood
    color_map = {
        "X": hair, "H": hair, "S": skin,
        "J": suit, "W": shirt, "T": tie, "P": suit, "K": INK,
        "C": clip, "B": INK,
    }

    rects = []
    for y, row in enumerate(_GM_SPRITE):
        for x, c in enumerate(row):
            if c == ".":
                continue
            rects.append(
                f'<rect x="{x*px}" y="{y*px}" width="{px}" height="{px}" '
                f'fill="{color_map[c]}" shape-rendering="crispEdges"/>'
            )

    # Paper clip at top of clipboard
    clip_band_x = 10 * px
    clip_band_y = 7 * px
    rects.append(
        f'<rect x="{clip_band_x}" y="{clip_band_y}" width="{px*2}" height="{px}" '
        f'fill="{INK}" shape-rendering="crispEdges"/>'
    )

    w = _SPR_W * px
    h = _SPR_H * px
    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'style="image-rendering: pixelated; image-rendering: crisp-edges;" class="gm-coach">'
        f'{"".join(rects)}'
        f'</svg>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700;9..144,900&family=Inter:wght@400;500;600;700&family=VT323&family=Silkscreen:wght@400;700&display=swap');

  .stApp {{ background: {PAPER}; color: {INK}; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  footer {{ visibility: hidden; }}

  html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {INK};
  }}
  h1, h2, h3 {{ font-family: 'Fraunces', Georgia, serif; letter-spacing: -0.02em; }}

  /* Pixel-style sidebar */
  section[data-testid="stSidebar"] {{
    background: {PAPER_2};
    border-right: 2px solid {INK};
  }}
  section[data-testid="stSidebar"] * {{ color: {INK} !important; }}
  section[data-testid="stSidebar"] label {{ font-weight: 500; font-size: 13px; }}

  /* PIXEL BUTTON: unmistakable, with hard drop shadow */
  .stButton > button {{
    background: {ORANGE} !important;
    color: {PAPER} !important;
    border: 2px solid {INK} !important;
    border-radius: 4px !important;
    padding: 10px 22px !important;
    font-family: 'Silkscreen', 'VT323', monospace !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    box-shadow: 4px 4px 0 {INK} !important;
    transition: transform 0.08s, box-shadow 0.08s !important;
  }}
  .stButton > button:hover {{
    transform: translate(-1px, -1px);
    box-shadow: 5px 5px 0 {INK} !important;
    background: #F97A2C !important;
  }}
  .stButton > button:active {{
    transform: translate(4px, 4px);
    box-shadow: 0 0 0 {INK} !important;
  }}
  .stButton > button:focus {{ outline: none !important; }}

  /* Inputs: pixel borders, flat */
  .stSelectbox [data-baseweb="select"] > div,
  .stNumberInput input,
  .stMultiSelect [data-baseweb="select"] > div {{
    background: {PAPER} !important;
    border: 1.5px solid {INK} !important;
    border-radius: 4px !important;
    color: {INK} !important;
  }}
  .stMultiSelect span[data-baseweb="tag"] {{
    background: {PAPER} !important; color: {INK} !important;
    border: 1.5px solid {INK} !important;
    border-radius: 3px !important;
    font-weight: 500 !important;
  }}

  /* Expander (rules):flat */
  [data-testid="stExpander"] {{
    border: 1.5px solid {INK} !important;
    border-radius: 4px !important;
    background: {PAPER} !important;
  }}
  [data-testid="stExpander"] summary {{ font-family: 'Silkscreen', monospace; font-size: 12px; letter-spacing: 0.1em; }}

  /* Checkboxes: pixel square */
  .stCheckbox [data-baseweb="checkbox"] > div:first-child {{
    border: 2px solid {INK} !important;
    border-radius: 2px !important;
    background: {PAPER} !important;
  }}

  /* Hero */
  .gm-eyebrow {{
    font-family: 'Silkscreen', 'VT323', monospace; font-size: 12px;
    letter-spacing: 0.18em; color: {ORANGE}; text-transform: uppercase;
    margin-bottom: 6px;
  }}
  .gm-title {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 900; font-size: 56px; line-height: 1;
    margin: 0 0 6px 0; color: {INK};
  }}
  .gm-sub {{ color: {MUTED}; font-size: 14px; margin-bottom: 18px; }}

  /* Summary strip: flat pixel panel */
  .gm-strip {{
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 20px; margin: 28px 0 14px 0;
    padding: 16px 22px;
    background: transparent;
    border-top: 2px solid {INK};
    border-bottom: 2px solid {INK};
    position: relative;
  }}
  .gm-strip > div {{
    padding-right: 20px;
    border-right: 2px dotted {RULE};
  }}
  .gm-strip > div:last-child {{ border-right: none; padding-right: 0; }}
  .gm-strip-label {{
    font-family: 'Silkscreen', monospace;
    font-size: 11px; letter-spacing: 0.14em; color: {MUTED};
    text-transform: uppercase; margin-bottom: 6px;
  }}
  .gm-strip-value {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 900; font-size: 34px; color: {INK}; line-height: 1;
  }}
  .gm-note {{ font-family: 'VT323', monospace; font-size: 16px; color: {MUTED}; margin-top: 4px; }}
  .gm-note.ok  {{ color: {COURT}; font-weight: 700; }}
  .gm-note.bad {{ color: {RED};   font-weight: 700; }}

  /* Roster heading: clean pixel label */
  .gm-rhead {{
    font-family: 'Silkscreen', monospace;
    font-weight: 400; font-size: 12px; letter-spacing: 0.16em;
    text-transform: uppercase; color: {MUTED};
    padding-bottom: 6px; margin: 26px 0 6px 0;
    border-bottom: 1px solid {RULE};
  }}

  /* Player row */
  .gm-player-line {{
    display: flex; align-items: center; gap: 14px;
    padding: 12px 6px;
    border-bottom: 1px dashed {RULE};
  }}
  .gm-player-line:last-child {{ border-bottom: none; }}
  .gm-player-line:hover {{ background: rgba(232,106,26,0.04); }}
  .gm-sprite-wrap {{
    width: 36px; min-width: 36px; display: flex; justify-content: center;
  }}
  .gm-player-info {{ flex: 1; }}
  .gm-player-name {{
    font-weight: 600; font-size: 15px; color: {INK};
  }}
  .gm-player-meta {{
    font-family: 'VT323', monospace;
    color: {MUTED}; font-size: 15px; margin-top: 1px; letter-spacing: 0.02em;
  }}
  .gm-player-sal {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 700; font-size: 18px; color: {INK};
    font-variant-numeric: tabular-nums;
  }}
  .gm-tag {{
    display: inline-block; padding: 1px 6px; margin-left: 6px;
    font-family: 'Silkscreen', monospace;
    font-size: 8px; font-weight: 700; letter-spacing: 0.1em;
    border: 1.5px solid {INK};
    border-radius: 2px;
    text-transform: uppercase; vertical-align: middle;
  }}
  .gm-tag-ntc    {{ background: #F6DADA; color: {INK}; }}
  .gm-tag-recent {{ background: #FDEBC8; color: {INK}; }}

  /* Verdict: pixel panel */
  .gm-verdict {{
    display: flex; gap: 16px; align-items: center;
    margin: 24px 0; padding: 18px 22px;
    border: 2px solid {INK}; border-radius: 4px;
    background: {PAPER_2};
    box-shadow: 4px 4px 0 {INK};
  }}
  .gm-verdict-ok  {{ background: #E6F1E7; }}
  .gm-verdict-bad {{ background: #F6E0E0; }}
  .gm-verdict .gm-verdict-dot {{
    width: 16px; height: 16px; border: 2px solid {INK};
    border-radius: 2px;
    flex-shrink: 0;
  }}
  .gm-verdict-ok  .gm-verdict-dot {{ background: {COURT}; }}
  .gm-verdict-bad .gm-verdict-dot {{ background: {RED}; }}
  .gm-verdict-title {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 900; font-size: 22px; line-height: 1.1;
  }}
  .gm-verdict-ok  .gm-verdict-title {{ color: {COURT}; }}
  .gm-verdict-bad .gm-verdict-title {{ color: {RED}; }}
  .gm-verdict-sub {{
    font-family: 'VT323', monospace;
    color: {MUTED}; font-size: 16px; margin-top: 2px;
  }}

  /* Sprite animation */
  @keyframes gm-ball-bounce {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(-4px); }}
  }}
  @keyframes gm-sprite-step {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(-1px); }}
  }}
  .gm-sprite .gm-ball {{
    animation: gm-ball-bounce 0.6s steps(2, end) infinite;
    transform-origin: center;
  }}
  .gm-sprite {{ animation: gm-sprite-step 0.6s steps(2, end) infinite; }}

  /* GM mascot: gentle idle bob */
  @keyframes gm-coach-bob {{
    0%, 100% {{ transform: translateY(0); }}
    50%      {{ transform: translateY(-2px); }}
  }}
  .gm-coach {{ animation: gm-coach-bob 1.6s steps(2, end) infinite; }}
  .gm-hero-mascot {{
    padding: 6px 10px 0 4px;
  }}

  /* Running indicator */
  .gm-running {{
    display: flex; align-items: center; gap: 14px;
    padding: 14px 0;
    font-family: 'Silkscreen', monospace;
    font-size: 14px; letter-spacing: 0.1em; text-transform: uppercase;
    color: {INK};
  }}
  .gm-running::after {{
    content: "▌"; animation: gm-blink 0.8s steps(2) infinite; color: {ORANGE};
  }}
  @keyframes gm-blink {{ 50% {{ opacity: 0; }} }}

  /* Tabs: pixel underline, hard override of streamlit default */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 2px; border-bottom: 2px solid {INK};
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {INK} !important;
    font-family: 'Silkscreen', monospace !important;
    font-weight: 400 !important; font-size: 12px !important;
    letter-spacing: 0.1em; text-transform: uppercase;
    border-radius: 0;
    padding: 10px 14px;
  }}
  .stTabs [aria-selected="true"] {{
    background: {INK} !important;
    color: {PAPER} !important;
  }}
  .stTabs [aria-selected="true"] * {{ color: {PAPER} !important; }}
  /* kill the red streamlit underline */
  .stTabs [data-baseweb="tab-highlight"] {{ background: {INK} !important; display: none !important; }}
  .stTabs [data-baseweb="tab-border"] {{ background: {INK} !important; }}

  /* Big number */
  .gm-big {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 900; font-size: 38px; line-height: 1; color: {INK};
  }}
  .gm-big small {{ font-family: 'VT323', monospace; font-size: 18px; color: {MUTED}; font-weight: 500; }}

  /* How-it-works: flat card */
  .gm-how {{
    margin-top: 28px; padding: 22px 24px;
    background: {PAPER_2}; border-radius: 4px;
    border-left: 3px solid {ORANGE};
  }}
  .gm-how ol {{ margin: 10px 0 0 18px; line-height: 1.9; color: {INK}; font-size: 14px; }}
  .gm-how b {{ color: {ORANGE}; }}

  /* Data frame: clean frame, no shadow */
  .stDataFrame {{
    border: 1.5px solid {INK} !important;
    border-radius: 4px !important;
    overflow: hidden;
  }}

  /* Plotly transparent bg */
  .js-plotly-plot .plotly {{ background: transparent !important; }}

  /* Scanline / pixel texture on hero (subtle) */
  .gm-hero-wrap {{ position: relative; }}

  /* Team card: pixel jersey + team info */
  .gm-teamcard {{
    display: flex; align-items: center; gap: 12px;
    padding: 10px 12px; margin: 2px 0 8px 0;
    background: {PAPER};
    border: 1.5px solid {INK};
    border-radius: 4px;
    border-left: 6px solid var(--team-primary, {INK});
  }}
  .gm-teamcard-jersey {{ flex-shrink: 0; }}
  .gm-teamcard-info {{ flex: 1; min-width: 0; }}
  .gm-teamcard-name {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 700; font-size: 15px; color: {INK};
    line-height: 1.1;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .gm-teamcard-meta {{
    font-family: 'VT323', monospace; font-size: 14px;
    color: {MUTED}; letter-spacing: 0.04em; margin-top: 2px;
  }}
  .gm-teamcard-meta b {{ color: {INK}; font-weight: 500; }}

  /* Sidebar label: pixel */
  .gm-sidelabel {{
    font-family: 'Silkscreen', monospace;
    font-size: 10px; letter-spacing: 0.18em; color: {MUTED};
    text-transform: uppercase; margin: 14px 0 4px 0;
  }}

  /* Team picker popover panel */
  .gm-pickpanel-head {{
    font-family: 'Silkscreen', monospace;
    font-size: 11px; letter-spacing: 0.16em; color: {MUTED};
    text-transform: uppercase; margin: 2px 0 10px 0;
  }}
  .gm-pick-tile {{
    display: flex; align-items: center; justify-content: center;
    padding: 6px 4px 2px 4px;
    background: {PAPER};
    border: 1.5px solid {RULE};
    border-bottom: 3px solid var(--tile-accent, {INK});
    border-radius: 4px;
    transition: transform 0.08s, border-color 0.08s;
  }}
  .gm-pick-tile:hover {{ transform: translateY(-1px); border-color: {INK}; }}
  .gm-pick-tile.is-selected {{
    border: 2px solid {INK};
    border-bottom: 3px solid var(--tile-accent, {INK});
    box-shadow: 2px 2px 0 {INK};
  }}
  .gm-pick-jersey {{ line-height: 0; }}

  /* Popover trigger (Change team):small and quiet so the card is the focus */
  [data-testid="stPopover"] > div > button {{
    background: {PAPER} !important;
    color: {INK} !important;
    border: 1.5px solid {INK} !important;
    border-radius: 4px !important;
    padding: 6px 10px !important;
    font-family: 'Silkscreen', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    box-shadow: none !important;
    margin-top: 4px !important;
  }}
  [data-testid="stPopover"] > div > button:hover {{
    background: {PAPER_2} !important;
  }}
  /* Team-pick tiny buttons inside the picker (popover or expander):very compact */
  [data-testid="stPopoverBody"] .stButton > button,
  [data-testid="stExpander"] .stButton > button {{
    padding: 4px 2px !important;
    font-size: 10px !important;
    letter-spacing: 0.04em !important;
    box-shadow: none !important;
    background: {PAPER} !important;
    color: {INK} !important;
    border: 1.5px solid {INK} !important;
    margin-top: 2px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    min-width: 0 !important;
  }}
  [data-testid="stExpander"] .stButton > button > div,
  [data-testid="stExpander"] .stButton > button p {{
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: clip !important;
  }}
  [data-testid="stPopoverBody"] .stButton > button:hover,
  [data-testid="stExpander"] .stButton > button:hover {{
    background: {INK} !important;
    color: {PAPER} !important;
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading NBA data…")
def _load_all():
    df = load_dataset(2025)
    teams = get_all_teams(2025)
    return df, teams


@st.cache_resource(show_spinner="Training valuation model…")
def _get_model() -> PlayerValuationModel:
    m = PlayerValuationModel()
    m.fit()
    return m


def _roster(df: pd.DataFrame, abbr: str) -> List[PlayerRecord]:
    sub = df[df["team"] == abbr]
    return [
        PlayerRecord(
            player_id=int(r["player_id"]), name=str(r["name"]), team=str(r["team"]),
            position=str(r["position"]), age=int(r["age"]), salary=float(r["salary"]),
            bpm=float(r["bpm"]), vorp=float(r["vorp"]), ts_pct=float(r["ts_pct"]),
            has_ntc=bool(r["has_ntc"]), months_since_signing=int(r["months_since_signing"]),
        )
        for _, r in sub.iterrows()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Player row renderer: now with pixel sprite
# ─────────────────────────────────────────────────────────────────────────────

def player_row(p: PlayerRecord) -> str:
    tags = ""
    if p.has_ntc:
        tags += '<span class="gm-tag gm-tag-ntc">NTC</span>'
    if p.is_recently_signed():
        tags += '<span class="gm-tag gm-tag-recent">New</span>'
    return (
        f'<div class="gm-player-line">'
        f'  <div class="gm-sprite-wrap">{pixel_sprite(p)}</div>'
        f'  <div class="gm-player-info">'
        f'    <div class="gm-player-name">{p.name}{tags}</div>'
        f'    <div class="gm-player-meta">{p.position} · {p.age}y · BPM {p.bpm:+.1f} · VORP {p.vorp:.1f}</div>'
        f'  </div>'
        f'  <div class="gm-player-sal">${p.salary/1e6:,.1f}M</div>'
        f'</div>'
    )


def strip_cell(label: str, value: str, note: str = "", tone: str = "") -> str:
    tone_cls = f"gm-note {tone}".strip()
    note_html = f'<div class="{tone_cls}">{note}</div>' if note else ""
    return (
        f'<div>'
        f'<div class="gm-strip-label">{label}</div>'
        f'<div class="gm-strip-value">{value}</div>'
        f'{note_html}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

df, teams = _load_all()
model = _get_model()

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────

# hero mascot: a suited GM holding a clipboard
_hero_sprite = gm_sprite(px=5)
st.markdown(
    f'<div class="gm-hero-wrap" style="display:flex; align-items:center; gap:20px; margin-bottom:6px;">'
    f'  <div class="gm-hero-mascot">{_hero_sprite}</div>'
    f'  <div>'
    f'    <div class="gm-eyebrow">NBA TRADE DESK · CIS 1921</div>'
    f'    <div class="gm-title">GM Mode</div>'
    f'    <div class="gm-sub">{len(df)} players · {len(teams)} teams · live rosters via nba_api + salaries via Basketball-Reference.</div>'
    f'  </div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _player_option_label(p) -> str:
    """Compact option label for the player multiselect: jersey + name + salary."""
    num = p.jersey_num if p.jersey_num else "--"
    return f"#{num:>2}  {p.position}  {p.name}  (${p.salary/1e6:.1f}M)"


def team_picker(key: str, label: str, teams_list, default_idx: int):
    """Popover picker: trigger shows the selected jersey + team name; panel shows a 5×6 grid of clickable jerseys."""
    state_key = f"picked_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = teams_list[default_idx][0]

    current_abbr = st.session_state[state_key]
    current_name = next((n for a, n, _ in teams_list if a == current_abbr), teams_list[0][1])
    current_count = next((c for a, n, c in teams_list if a == current_abbr), 0)
    primary, _ = TEAM_COLORS.get(current_abbr, ("#333333", "#DDDDDD"))

    st.markdown(f'<div class="gm-sidelabel">{label}</div>', unsafe_allow_html=True)

    # Render a preview card just above the popover trigger so the logo is visible
    st.markdown(
        f'<div class="gm-teamcard" style="--team-primary:{primary};">'
        f'  <div class="gm-teamcard-jersey">{team_logo(current_abbr, px=6)}</div>'
        f'  <div class="gm-teamcard-info">'
        f'    <div class="gm-teamcard-name">{current_name}</div>'
        f'    <div class="gm-teamcard-meta"><b>{current_count}</b> players · tap ⇩ to change</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.expander(f"Change {label}", expanded=False):
        st.markdown('<div class="gm-pickpanel-head">Pick a team</div>', unsafe_allow_html=True)
        # 6 rows × 5 cols = 30 teams
        sorted_teams = sorted(teams_list, key=lambda x: x[0])  # alphabetical by abbr
        for row_i in range(6):
            cols = st.columns(5, gap="small")
            for col_i, col in enumerate(cols):
                idx = row_i * 5 + col_i
                if idx >= len(sorted_teams):
                    continue
                abbr, tname, tcount = sorted_teams[idx]
                tp, _ = TEAM_COLORS.get(abbr, ("#333333", "#DDDDDD"))
                selected = (abbr == current_abbr)
                with col:
                    st.markdown(
                        f'<div class="gm-pick-tile {"is-selected" if selected else ""}" '
                        f'style="--tile-accent:{tp};">'
                        f'<div class="gm-pick-jersey">{team_logo(abbr, px=4)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(abbr, key=f"{state_key}_{abbr}", use_container_width=True):
                        st.session_state[state_key] = abbr
                        st.rerun()

    return current_abbr, current_name


with st.sidebar:
    st.markdown(f'<div class="gm-eyebrow" style="margin:4px 0 10px 0;">Trade setup</div>', unsafe_allow_html=True)

    # ── Team pickers (popover grid of pixel jerseys) ──────────
    default_a = next((i for i, (a, n, _) in enumerate(teams) if "Lakers" in n), 0)
    default_b = next((i for i, (a, n, _) in enumerate(teams) if "Thunder" in n), 1 if len(teams) > 1 else 0)

    team_a, team_a_label = team_picker("team_a", "Team A", teams, default_a)
    roster_a = _roster(df, team_a)

    team_b, team_b_label = team_picker("team_b", "Team B", teams, default_b)
    roster_b = _roster(df, team_b)

    # ── Player pickers (sorted by salary, biggest contracts first) ─
    sorted_a = sorted(roster_a, key=lambda p: -p.salary)
    sorted_b = sorted(roster_b, key=lambda p: -p.salary)
    label_to_player_a = {_player_option_label(p): p for p in sorted_a}
    label_to_player_b = {_player_option_label(p): p for p in sorted_b}

    st.markdown(f'<div class="gm-sidelabel">{team_a} sends →</div>', unsafe_allow_html=True)
    default_send_a = [list(label_to_player_a.keys())[0]] if label_to_player_a else []
    sel_a = st.multiselect("   ", list(label_to_player_a.keys()), default=default_send_a,
                           label_visibility="collapsed", key="sel_a")
    outgoing = [label_to_player_a[k] for k in sel_a]

    st.markdown(f'<div class="gm-sidelabel">← {team_b} sends</div>', unsafe_allow_html=True)
    default_send_b = [list(label_to_player_b.keys())[0]] if label_to_player_b else []
    sel_b = st.multiselect("    ", list(label_to_player_b.keys()), default=default_send_b,
                           label_visibility="collapsed", key="sel_b")
    incoming = [label_to_player_b[k] for k in sel_b]

    with st.expander("Rules", expanded=False):
        enforce_cap    = st.checkbox("Hard cap", value=True)
        enforce_ntc    = st.checkbox("No-trade clauses", value=True)
        enforce_recent = st.checkbox("Recently-signed rule", value=True)
        cap_limit      = st.number_input("Hard cap ($M)", min_value=50.0, max_value=400.0,
                                         value=165.0, step=1.0)

    st.markdown(" ")
    run_clicked = st.button("▶  Run trade", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Summary strip
# ─────────────────────────────────────────────────────────────────────────────

out_sal = sum(p.salary for p in outgoing)
in_sal  = sum(p.salary for p in incoming)
if out_sal > 0 and in_sal > 0:
    # CBA rule applies per team: incoming <= outgoing * 1.25 + $100K.
    # Both teams must pass; the tighter side is whoever sends less.
    cap_a_receives = out_sal * 1.25 + 100_000
    cap_b_receives = in_sal  * 1.25 + 100_000
    match_ok = (in_sal <= cap_a_receives) and (out_sal <= cap_b_receives)
    ratio = max(out_sal, in_sal) / min(out_sal, in_sal)
else:
    ratio, match_ok = 0.0, False

st.markdown(
    f'<div class="gm-strip">'
    f'{strip_cell(f"{team_a} sends", f"${out_sal/1e6:,.1f}M", f"{len(outgoing)} player{'s' if len(outgoing)!=1 else ''}")}'
    f'{strip_cell(f"{team_b} sends", f"${in_sal/1e6:,.1f}M", f"{len(incoming)} player{'s' if len(incoming)!=1 else ''}")}'
    f'{strip_cell("Salary gap", f"{ratio*100:.0f}%", "within 125% + $100K" if match_ok else "exceeds 125% + $100K", "ok" if match_ok else "bad")}'
    f'</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Rosters: pixel sprites appear next to every player
# ─────────────────────────────────────────────────────────────────────────────

lcol, rcol = st.columns(2)
with lcol:
    st.markdown(f'<div class="gm-rhead">{team_a} sends</div>', unsafe_allow_html=True)
    if outgoing:
        st.markdown("".join(player_row(p) for p in outgoing), unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color:{MUTED}; padding:14px 0; font-style:italic; font-size:13px;">Select players in the sidebar.</div>', unsafe_allow_html=True)
with rcol:
    st.markdown(f'<div class="gm-rhead">{team_b} sends</div>', unsafe_allow_html=True)
    if incoming:
        st.markdown("".join(player_row(p) for p in incoming), unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color:{MUTED}; padding:14px 0; font-style:italic; font-size:13px;">Select players in the sidebar.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

if run_clicked:
    if not outgoing or not incoming:
        st.warning("Pick at least one player on each side first.")
        st.stop()

    # little "running" indicator with a walking pixel
    running_sprite = ""
    if outgoing:
        running_sprite = pixel_sprite(outgoing[0], px=4)
    anim_slot = st.empty()
    anim_slot.markdown(
        f'<div class="gm-running">{running_sprite}<span>Crunching the trade</span></div>',
        unsafe_allow_html=True,
    )

    cfg = ConstraintsConfig(
        enforce_hard_cap=enforce_cap,
        enforce_no_trade_clauses=enforce_ntc,
        enforce_recently_signed=enforce_recent,
        hard_cap_threshold=cap_limit * 1_000_000,
    )

    # Layer 1:SAT
    sat = SATFeasibilityChecker(cfg)
    t0 = time.perf_counter()
    sat_result = sat.check(
        roster_a=roster_a, roster_b=roster_b,
        candidates_from_a=outgoing, candidates_from_b=incoming,
    )
    sat_ms = (time.perf_counter() - t0) * 1000

    cap_a = cfg.salary_cap(out_sal)
    cap_b = cfg.salary_cap(in_sal)
    sal_ok = (in_sal <= cap_a) and (out_sal <= cap_b)
    proposed_ids = {p.player_id for p in outgoing + incoming}
    blocked_ids  = sat_result.forced_out & proposed_ids
    overall_ok   = sat_result.feasible and sal_ok and not blocked_ids

    ctx_a = TeamContext(team_abbr=team_a, rebuild_score=0.3)
    ctx_b = TeamContext(team_abbr=team_b, rebuild_score=0.7)
    for p in outgoing:
        p.valuation = model.predict(p, ctx_b)
    for p in incoming:
        p.valuation = model.predict(p, ctx_a)
    out_val = sum(p.valuation for p in outgoing)
    in_val  = sum(p.valuation for p in incoming)

    time.sleep(0.4)
    anim_slot.empty()

    # Verdict
    if overall_ok:
        st.markdown(
            f'<div class="gm-verdict gm-verdict-ok">'
            f'  <div class="gm-verdict-dot"></div>'
            f'  <div>'
            f'    <div class="gm-verdict-title">Trade is feasible</div>'
            f'    <div class="gm-verdict-sub">SAT satisfied in {sat_ms:.1f} ms · salary match within CBA.</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        reasons = list(sat_result.violations)
        if not sal_ok:
            reasons.append(f"Salary match: ${in_sal/1e6:.1f}M vs cap ${cap_a/1e6:.2f}M / ${cap_b/1e6:.2f}M.")
        if blocked_ids:
            blocked_names = [p.name for p in outgoing + incoming if p.player_id in blocked_ids]
            reasons.append("Blocked: " + ", ".join(blocked_names))
        shown = reasons[:3] if reasons else ["violates at least one CBA rule"]
        st.markdown(
            f'<div class="gm-verdict gm-verdict-bad">'
            f'  <div class="gm-verdict-dot"></div>'
            f'  <div>'
            f'    <div class="gm-verdict-title">Trade is blocked</div>'
            f'    <div class="gm-verdict-sub">{" · ".join(shown)}</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Tabs
    tab_overview, tab_mip, tab_details = st.tabs(["Overview", "MIP optimizer", "Under the hood"])

    with tab_overview:
        val_delta = in_val - out_val
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="gm-strip-label">Value to {team_a}</div>'
                f'<div class="gm-big">{in_val:.2f}</div>'
                f'<div class="gm-note {"ok" if val_delta >= 0 else "bad"}">Δ {val_delta:+.2f} vs outgoing</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="gm-strip-label">Value to {team_b}</div>'
                f'<div class="gm-big">{out_val:.2f}</div>'
                f'<div class="gm-note">receiving perspective</div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="gm-strip-label">SAT solve</div>'
                f'<div class="gm-big">{sat_ms:.1f}<small> ms</small></div>'
                f'<div class="gm-note">CDCL · {len(sat_result.forced_out)} locked</div>',
                unsafe_allow_html=True,
            )

        fig = go.Figure()
        fig.add_bar(
            x=[p.name for p in outgoing],
            y=[p.valuation for p in outgoing],
            name=f"{team_a} → {team_b}", marker_color="#B9B3A0",
        )
        fig.add_bar(
            x=[p.name for p in incoming],
            y=[p.valuation for p in incoming],
            name=f"{team_b} → {team_a}", marker_color=ORANGE,
        )
        fig.update_layout(
            template="simple_white",
            plot_bgcolor=PAPER, paper_bgcolor=PAPER,
            font=dict(family="Inter, sans-serif", color=INK, size=12),
            height=300, margin=dict(l=10, r=10, t=30, b=10),
            barmode="group",
            yaxis_title="GBT valuation",
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_mip:
        st.markdown(f'<div class="gm-strip-label">Layer 2 · mixed-integer program</div>', unsafe_allow_html=True)
        st.caption("OR-Tools CP-SAT and PuLP+CBC both search the candidate pool for the max-value deal that fits the cap.")

        with st.spinner("Two solvers racing…"):
            cand_a = sorted(roster_a, key=lambda p: -p.salary)[:6]
            cand_b = sorted(roster_b, key=lambda p: -p.salary)[:6]
            for p in outgoing:
                if p not in cand_a: cand_a.append(p)
            for p in incoming:
                if p not in cand_b: cand_b.append(p)
            for p in cand_a: p.valuation = model.predict(p, ctx_b)
            for p in cand_b: p.valuation = model.predict(p, ctx_a)

            sat_for_mip = sat.check(
                roster_a=roster_a, roster_b=roster_b,
                candidates_from_a=cand_a, candidates_from_b=cand_b,
            )
            ortools_res, pulp_res = solve_both(
                candidates_from_a=cand_a, candidates_from_b=cand_b,
                roster_a=roster_a, roster_b=roster_b,
                sat_result=sat_for_mip, config=cfg,
                team_a=team_a, team_b=team_b, time_limit_s=10.0,
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="gm-strip-label">OR-Tools CP-SAT</div>'
                f'<div class="gm-big">{ortools_res.solve_time_ms:.1f}<small> ms</small></div>'
                f'<div class="gm-note">obj {ortools_res.objective_value:+.3f}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="gm-strip-label">PuLP + CBC</div>'
                f'<div class="gm-big">{pulp_res.solve_time_ms:.1f}<small> ms</small></div>'
                f'<div class="gm-note">obj {pulp_res.objective_value:+.3f}</div>',
                unsafe_allow_html=True,
            )
        with c3:
            speedup = pulp_res.solve_time_ms / max(ortools_res.solve_time_ms, 0.001)
            agree = abs(ortools_res.objective_value - pulp_res.objective_value) < 0.01
            st.markdown(
                f'<div class="gm-strip-label">Speedup</div>'
                f'<div class="gm-big">{speedup:.1f}×</div>'
                f'<div class="gm-note {"ok" if agree else "bad"}">{"solvers agree" if agree else "disagree"}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="gm-rhead">MIP-optimal package</div>', unsafe_allow_html=True)
        mpa, mpb = st.columns(2)
        with mpa:
            st.markdown(f'<div class="gm-strip-label">{team_a} sends</div>', unsafe_allow_html=True)
            if ortools_res.players_traded_from_a:
                st.markdown("".join(player_row(p) for p in ortools_res.players_traded_from_a), unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:{MUTED};font-style:italic;font-size:13px;">none</div>', unsafe_allow_html=True)
        with mpb:
            st.markdown(f'<div class="gm-strip-label">{team_b} sends</div>', unsafe_allow_html=True)
            if ortools_res.players_traded_from_b:
                st.markdown("".join(player_row(p) for p in ortools_res.players_traded_from_b), unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:{MUTED};font-style:italic;font-size:13px;">none</div>', unsafe_allow_html=True)

    with tab_details:
        st.markdown(f'<div class="gm-strip-label">SAT check</div>', unsafe_allow_html=True)
        if sat_result.violations:
            for r in sat_result.violations:
                st.markdown(f"- {r}")
        else:
            st.markdown("No NTC / recently-signed / roster-size violations.")

        st.markdown(f'<div class="gm-strip-label" style="margin-top:18px;">GBT features</div>', unsafe_allow_html=True)
        feat_df = pd.DataFrame([
            {
                "name": p.name, "team": p.team, "pos": p.position, "age": p.age,
                "salary ($M)": round(p.salary / 1e6, 2),
                "BPM": round(p.bpm, 2), "VORP": round(p.vorp, 2), "TS%": round(p.ts_pct, 3),
                "valuation": round(p.valuation, 3),
            }
            for p in outgoing + incoming
        ])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

else:
    st.markdown(
        f'<div class="gm-how">'
        f'<div class="gm-eyebrow">How it works</div>'
        f'<p style="font-size:14px; margin:6px 0 4px 0; color:{INK};">'
        f'Pick two teams and the players moving each way, then hit <b>Evaluate</b>. Three layers run:'
        f'</p>'
        f'<ol>'
        f'<li><b>SAT feasibility</b>:a CDCL solver (pysat) checks the boolean CBA rules.</li>'
        f'<li><b>MIP optimization</b>:OR-Tools CP-SAT races PuLP+CBC to find the max-value deal under the hard cap.</li>'
        f'<li><b>GBT valuation</b>:a gradient-boosted regressor scores every player in the receiving team\'s context.</li>'
        f'</ol>'
        f'</div>',
        unsafe_allow_html=True,
    )
