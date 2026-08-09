"""Scoring rules for a predictive *distribution*, not just a point forecast.

RMSE and MAE only see the middle of the forecast. The reason to fit a Bayesian
model at all is the rest of it, so the honest report has to score the shape:
CRPS for sharpness-given-calibration, interval coverage for whether the stated
uncertainty is real, and PIT for where it is wrong if it is.
"""

from __future__ import annotations

import numpy as np


def rmse(y, pred) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae(y, pred) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(np.mean(np.abs(y - pred)))


def crps_from_samples(draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score per observation.

    ``draws`` is (n_draws, n_obs). Uses the order-statistic identity

        CRPS = (2/n^2) * sum_i (x_(i) - y) * (n * 1{y < x_(i)} - i + 0.5)

    which is O(n log n) in the number of draws rather than the O(n^2) of
    evaluating E|X - X'| directly, and exact rather than binned.

    CRPS collapses to MAE when the forecast is a point mass, so it is directly
    comparable against a deterministic model's MAE - a distribution has to earn
    its width.
    """
    x = np.sort(np.asarray(draws, float), axis=0)
    y = np.asarray(y, float)
    n = x.shape[0]
    i = np.arange(1, n + 1)[:, None]
    return (2.0 / n ** 2) * np.sum(
        (x - y) * (n * (y < x) - i + 0.5), axis=0
    )


def interval_coverage(draws: np.ndarray, y: np.ndarray, level: float) -> float:
    """Share of outcomes inside the central predictive interval at ``level``."""
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
    return float(np.mean((y >= lo) & (y <= hi)))


def coverage_given_played(
    draws: np.ndarray, games_draws: np.ndarray, y, y_games, level: float
) -> float:
    """Interval coverage among players who did play, using only playing draws.

    The unconditional predictive is bimodal - a spike at zero for the quarter of
    players who leave the league, and a lump wherever a healthy season would
    land. A central interval across a gap like that is wide by construction, so
    unconditional coverage overshoots its nominal level even when the model is
    right. Conditioning both sides on playing separates "is the scoring model
    calibrated" from "is the dropout rate calibrated", which are different
    questions with different fixes.
    """
    y, y_games = np.asarray(y, float), np.asarray(y_games, float)
    played = y_games > 0
    cond = np.where(games_draws > 0, draws, np.nan)[:, played]
    lo, hi = np.nanquantile(cond, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
    return float(np.mean((y[played] >= lo) & (y[played] <= hi)))


def conditional_on_playing(
    draws: np.ndarray, games_draws: np.ndarray, n_out: int | None = None, seed: int = 0
):
    """Re-draw the predictive conditional on the player being in the league.

    Whether a player is in the league next season is mostly *not* a forecasting
    problem for the person using this: retirements, releases and unsigned free
    agents are known in August, and a drafter simply does not draft them. Scoring
    the model against a coin flip it never had to call makes the headline error
    look worse than the thing being asked of it, and it hides movement in the
    part that does matter - points, given he is on the field.

    So this keeps only the draws where the simulated player played, and resamples
    them with replacement to a fixed width so every metric downstream works
    unchanged. What survives is the model's answer to "how many points, given he
    is in the league", which is the question a drafter actually poses.

    Rows where no draw has him playing keep their unconditional draws; there is
    nothing to condition on. Returns ``(draws, mask_of_usable_rows)``.
    """
    draws = np.asarray(draws, float)
    playing = np.asarray(games_draws) > 0
    n_draws, n_rows = draws.shape
    n_out = n_draws if n_out is None else n_out

    rng = np.random.default_rng(seed)
    out = np.empty((n_out, n_rows))
    usable = playing.any(axis=0)

    for j in range(n_rows):
        pool = draws[playing[:, j], j] if usable[j] else draws[:, j]
        out[:, j] = rng.choice(pool, size=n_out, replace=True)
    return out, usable


def pit(draws: np.ndarray, y: np.ndarray, rng=None) -> np.ndarray:
    """Randomised probability integral transform.

    If the forecast distribution is right, these are uniform on [0,1]. The
    randomisation between P(X < y) and P(X <= y) matters here because the
    predictive has a real atom at zero - a quarter of players score exactly
    nothing - and without it every such player would pile up at PIT = 0 and
    look like a calibration failure that isn't one.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    y = np.asarray(y, float)
    below = np.mean(draws < y, axis=0)
    at = np.mean(draws == y, axis=0)
    return below + rng.random(len(y)) * at


def summarise(draws: np.ndarray, y: np.ndarray, point: str = "mean") -> dict:
    """Full scorecard for a sampled predictive distribution."""
    y = np.asarray(y, float)
    pred = draws.mean(axis=0) if point == "mean" else np.median(draws, axis=0)
    order = np.argsort(np.argsort(pred))
    truth = np.argsort(np.argsort(y))
    return {
        "n": len(y),
        "rmse": rmse(y, pred),
        "mae": mae(y, np.median(draws, axis=0)),
        "crps": float(np.mean(crps_from_samples(draws, y))),
        "cov50": interval_coverage(draws, y, 0.50),
        "cov80": interval_coverage(draws, y, 0.80),
        "cov90": interval_coverage(draws, y, 0.90),
        "spearman": float(np.corrcoef(order, truth)[0, 1]),
        "bias": float(pred.mean() - y.mean()),
    }


def summarise_point(pred: np.ndarray, y: np.ndarray) -> dict:
    """Scorecard for a deterministic forecast, for baseline comparison.

    CRPS is reported as MAE, which is what CRPS equals for a point mass - so
    the column stays comparable rather than being left blank.
    """
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    order = np.argsort(np.argsort(pred))
    truth = np.argsort(np.argsort(y))
    return {
        "n": len(y),
        "rmse": rmse(y, pred),
        "mae": mae(y, pred),
        "crps": mae(y, pred),
        "cov50": np.nan,
        "cov80": np.nan,
        "cov90": np.nan,
        "spearman": float(np.corrcoef(order, truth)[0, 1]),
        "bias": float(pred.mean() - y.mean()),
    }
