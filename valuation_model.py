"""
valuation_model.py
------------------
Gradient-Boosted Tree (GBT) player valuation model.

The model takes per-player features (age, salary, BPM, VORP, TS%, positional
fit, team rebuild score) and outputs a scalar *valuation score* that represents
how valuable a player is in a given team context.  This score is what the MIP
layer maximises.

Because real historical trade data is not freely available through a public
API, the model is trained on *synthetic* data generated to approximate
realistic NBA player distributions.  The synthetic labels are a deterministic
function of the features plus Gaussian noise, so the model learns sensible
feature importances even without real outcomes.

Public API
----------
PlayerValuationModel.fit(players)        – train on a list of PlayerRecords
PlayerValuationModel.predict(player, team_context) → float
PlayerValuationModel.batch_predict(players, team_context) → dict[int, float]
generate_synthetic_training_data(n)     – standalone helper
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[valuation_model] scikit-learn not installed – using linear fallback.")

from data_fetcher import PlayerRecord


# ─────────────────────────────────────────────────────────────────────────────
# Team context
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TeamContext:
    """
    Encodes the receiving team's situation so the model can compute
    how well an incoming player fits.

    Attributes
    ----------
    team_abbr : str
        3-letter team abbreviation.
    rebuild_score : float in [0, 1]
        0 = full contender, 1 = full rebuild.
        Affects whether young cheap players score higher than expensive vets.
    positional_needs : dict[str, float]
        Keys are positions ("PG", "SG", "SF", "PF", "C").
        Values in [0, 1]: 1 = desperate need, 0 = no need.
    cap_space : float
        Current cap space in USD (may be negative if over cap).
    """
    team_abbr: str
    rebuild_score: float = 0.5
    positional_needs: dict[str, float] = None  # filled in __post_init__
    cap_space: float = 0.0

    def __post_init__(self):
        if self.positional_needs is None:
            # Default: equal need at every position
            self.positional_needs = {pos: 0.5 for pos in ("PG", "SG", "SF", "PF", "C")}

    def positional_fit(self, player_position: str) -> float:
        """Return the team's positional need for the given position [0, 1]."""
        return self.positional_needs.get(player_position, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

_POSITION_INDEX = {"PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4}
_SALARY_SCALE   = 1e7    # scale salaries to ~units of $10M


def build_feature_vector(
    player: PlayerRecord,
    team_context: TeamContext,
) -> list[float]:
    """
    Convert a PlayerRecord + TeamContext into a fixed-length feature vector.

    Feature layout (9 features):
      0  age                       (raw, e.g. 25)
      1  salary_scaled             (salary / $10M)
      2  bpm                       (Box Plus/Minus, typically -5 to +10)
      3  vorp                      (Value Over Replacement, typically 0 to 8)
      4  ts_pct                    (True Shooting %, typically 0.45 to 0.70)
      5  positional_fit            (team's need at player's position, 0–1)
      6  rebuild_score             (team rebuild score, 0=contender, 1=rebuild)
      7  age_rebuild_interaction   (age * rebuild_score – young players valued
                                    more by rebuilding teams)
      8  salary_per_vorp           (salary efficiency proxy; capped to avoid /0)
    """
    salary_scaled = player.salary / _SALARY_SCALE
    pos_fit       = team_context.positional_fit(player.position)
    rebuild       = team_context.rebuild_score

    # Avoid division by zero for VORP
    vorp_safe = max(player.vorp, 0.1)
    sal_per_vorp = salary_scaled / vorp_safe  # lower is more efficient

    return [
        float(player.age),
        salary_scaled,
        float(player.bpm),
        float(player.vorp),
        float(player.ts_pct),
        pos_fit,
        rebuild,
        float(player.age) * rebuild,   # interaction term
        sal_per_vorp,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic training data generation
# ─────────────────────────────────────────────────────────────────────────────

def _true_valuation(features: list[float]) -> float:
    """
    The latent (ground-truth) valuation function that our GBT will approximate.

    This is a hand-crafted formula mimicking how NBA teams actually value
    players: BPM and VORP most important, age and salary efficiency matter,
    positional fit adds context-specific value.
    """
    age, sal, bpm, vorp, ts, pos_fit, rebuild, age_reb, sal_per_vorp = features

    # Core production value
    production = 2.5 * vorp + 1.5 * bpm + 5.0 * (ts - 0.55)

    # Age premium/discount: peak ~26–28, discount for very young or very old
    age_factor = -0.15 * (age - 27) ** 2 / 10.0   # quadratic penalty

    # Salary efficiency: more valuable if producing a lot per dollar
    efficiency = -0.3 * sal_per_vorp  # lower sal_per_vorp → higher value

    # Positional fit bonus
    fit_bonus = 1.5 * pos_fit

    # Rebuild discount for expensive vets on rebuilding teams
    rebuild_penalty = -rebuild * sal * 0.5

    raw = production + age_factor + efficiency + fit_bonus + rebuild_penalty
    return float(raw)


def generate_synthetic_training_data(
    n_samples: int = 2000,
    seed: int = 42,
) -> tuple[list[list[float]], list[float]]:
    """
    Generate n_samples synthetic (features, valuation) training pairs.

    Returns
    -------
    X : list of feature vectors
    y : list of valuation scores
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    X, y = [], []
    positions = ["PG", "SG", "SF", "PF", "C"]

    for _ in range(n_samples):
        # Sample a synthetic player
        age     = rng.randint(19, 38)
        salary  = rng.uniform(1_000_000, 45_000_000)
        bpm     = np_rng.normal(0.0, 3.0)
        vorp    = max(0.0, np_rng.normal(1.5, 1.5))
        ts_pct  = np_rng.normal(0.555, 0.04)

        # Sample a synthetic team context
        pos_fit  = rng.uniform(0.0, 1.0)
        rebuild  = rng.uniform(0.0, 1.0)
        sal_sc   = salary / _SALARY_SCALE
        vorp_s   = max(vorp, 0.1)

        features = [
            float(age), sal_sc, float(bpm), float(vorp), float(ts_pct),
            pos_fit, rebuild, float(age) * rebuild, sal_sc / vorp_s,
        ]

        label = _true_valuation(features) + np_rng.normal(0, 0.3)
        X.append(features)
        y.append(label)

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Valuation model
# ─────────────────────────────────────────────────────────────────────────────

class PlayerValuationModel:
    """
    Gradient-Boosted Tree wrapper that predicts player valuation scores.

    Workflow:
      1. Call fit() once (or let predict() auto-fit on first use).
      2. Call predict(player, team_context) for individual predictions.
      3. Call batch_predict(players, team_context) for a full trade's players.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        seed: int = 42,
    ):
        self._fitted = False
        self._seed   = seed

        if SKLEARN_AVAILABLE:
            self._model = GradientBoostingRegressor(
                n_estimators  = n_estimators,
                max_depth     = max_depth,
                learning_rate = learning_rate,
                subsample     = 0.8,              # stochastic GBT
                random_state  = seed,
            )
            self._scaler = StandardScaler()
        else:
            self._model  = None
            self._scaler = None

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, players: list[PlayerRecord] | None = None) -> "PlayerValuationModel":
        """
        Train the GBT on synthetic data (optionally seeded with real players).

        If `players` is provided the real player feature vectors are appended
        to the synthetic set before fitting.  This allows the model to
        calibrate to the current season's distribution.
        """
        print("[valuation_model] Generating synthetic training data …")
        X_synth, y_synth = generate_synthetic_training_data(n_samples=2000,
                                                             seed=self._seed)

        X, y = list(X_synth), list(y_synth)

        # Append real player data with a neutral team context if provided
        if players:
            neutral_ctx = TeamContext(team_abbr="???")
            for p in players:
                fv = build_feature_vector(p, neutral_ctx)
                label = _true_valuation(fv)
                X.append(fv)
                y.append(label)

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=float)

        if SKLEARN_AVAILABLE:
            # Standardise features (GBT is not scale-sensitive but this helps
            # numerical stability with the linear fallback)
            X_arr = self._scaler.fit_transform(X_arr)
            self._model.fit(X_arr, y_arr)
            fi = self._model.feature_importances_
            print(
                f"[valuation_model] GBT trained on {len(y)} samples. "
                f"Feature importances: age={fi[0]:.2f}, sal={fi[1]:.2f}, "
                f"bpm={fi[2]:.2f}, vorp={fi[3]:.2f}, ts={fi[4]:.2f}, "
                f"pos_fit={fi[5]:.2f}, rebuild={fi[6]:.2f}"
            )
        else:
            # Fallback: store X, y for nearest-neighbour or direct formula
            self._X = X_arr
            self._y = y_arr
            print("[valuation_model] Using analytical fallback (no sklearn).")

        self._fitted = True
        return self

    def _auto_fit(self):
        """Auto-train with synthetic data if not already fitted."""
        if not self._fitted:
            self.fit()

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, player: PlayerRecord, team_context: TeamContext) -> float:
        """
        Return the valuation score for `player` in `team_context`.

        Higher is better.  Scores are roughly centred around 0 with typical
        range [-5, +15] for NBA players.
        """
        self._auto_fit()
        fv = build_feature_vector(player, team_context)

        if SKLEARN_AVAILABLE:
            X = self._scaler.transform([fv])
            return float(self._model.predict(X)[0])
        else:
            # Analytical fallback: evaluate the true-valuation formula directly
            return _true_valuation(fv)

    def batch_predict(
        self,
        players: list[PlayerRecord],
        team_context: TeamContext,
    ) -> dict[int, float]:
        """
        Predict valuations for all players in `players` given `team_context`.

        Returns a mapping player_id → valuation_score.
        Also writes the score back onto each PlayerRecord.valuation.
        """
        self._auto_fit()
        result: dict[int, float] = {}
        for p in players:
            score = self.predict(p, team_context)
            p.valuation = score
            result[p.player_id] = score
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: default team contexts for the demo
# ─────────────────────────────────────────────────────────────────────────────

LAL_CONTEXT = TeamContext(
    team_abbr="LAL",
    rebuild_score=0.1,                  # contender
    positional_needs={"PG": 0.3, "SG": 0.4, "SF": 0.2, "PF": 0.6, "C": 0.9},
    cap_space=-20_000_000,              # $20M over cap
)

BKN_CONTEXT = TeamContext(
    team_abbr="BKN",
    rebuild_score=0.8,                  # rebuilding
    positional_needs={"PG": 0.7, "SG": 0.5, "SF": 0.3, "PF": 0.8, "C": 0.6},
    cap_space=5_000_000,
)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_fetcher import _DEMO_PLAYERS

    model = PlayerValuationModel()
    model.fit()

    ad    = _DEMO_PLAYERS["anthony_davis"]
    simmo = _DEMO_PLAYERS["ben_simmons"]

    # Value Anthony Davis from the Nets' perspective (they'd be receiving him)
    val_ad_bkn   = model.predict(ad,    BKN_CONTEXT)
    # Value Ben Simmons from the Lakers' perspective
    val_sim_lal  = model.predict(simmo, LAL_CONTEXT)

    print(f"Anthony Davis  valuation for BKN: {val_ad_bkn:+.3f}")
    print(f"Ben Simmons    valuation for LAL: {val_sim_lal:+.3f}")
    print(f"Net delta (BKN gains - BKN loses): "
          f"{val_ad_bkn - model.predict(simmo, BKN_CONTEXT):+.3f}")
