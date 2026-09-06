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


TOP_K = 24

# The board sizes actually reported. 24 is the historical elite diagnostic - one
# starting lineup's worth of players per fold, and small enough that a single
# season is an anecdote. 100 is the population the owner drafts from: roughly the
# rows a 12-team league consumes in the first eight rounds. Reporting both is the
# point. The aggregate scorecard averages elite accuracy against ~450 players per
# season who are never drafted, and a change that improves the aggregate by
# getting better at the 30-point tail is worth nothing here; a change that wins
# the top 100 while wrecking the aggregate is visible because the aggregate is
# still printed next to it.
TOP_KS = (24, 100)


def top_of_board(pred: np.ndarray, y: np.ndarray, draws=None, k: int = TOP_K) -> dict:
    """Bias and CRPS over the top ``k`` rows, selected two different ways.

    The two selections answer different questions and only one of them can see
    the failure this project exists to fix.

    ``by_proj`` takes the k rows the model ranked highest. It asks *when I say
    elite, is he* - a precision question. A model that shrinks the top of the
    board hard can still score well here, because it is graded on the players it
    was already confident about and its errors on them are symmetric.

    ``by_actual`` takes the k rows that actually finished highest, whether the
    model ranked them there or not. It asks *when he was elite, did I say so* -
    a recall question. Systematic under-projection of the top shows up here as a
    large negative bias and nowhere else: those rows are a fixed 24 per season
    regardless of what the model believed, so a model cannot improve this number
    by declining to call anyone elite. Aggregate CRPS is dominated by the ~500
    ordinary rows and will happily trade this away.

    Bias is signed ``projection - actual``, so **negative means under-projected**.
    ``bias_pct`` divides by the mean actual, which is what makes the number
    comparable across folds whose top 24 sit at different levels.

    ``draws`` is optional; without it CRPS falls back to MAE, which is what CRPS
    equals for a point mass, so a deterministic baseline stays comparable.
    """
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    out = {}
    for label, key in (("proj", pred), ("actual", y)):
        # Ties broken by the other quantity is not worth the complexity: at k=24
        # out of ~500 rows, exact ties in either points or projected points are
        # vanishingly rare and never at the boundary.
        idx = np.argsort(-key, kind="stable")[:k]
        c = (
            float(np.mean(crps_from_samples(draws[:, idx], y[idx])))
            if draws is not None
            else mae(y[idx], pred[idx])
        )
        out[f"{label}_n"] = int(len(idx))
        out[f"{label}_mean_proj"] = float(pred[idx].mean())
        out[f"{label}_mean_actual"] = float(y[idx].mean())
        out[f"{label}_bias"] = float(pred[idx].mean() - y[idx].mean())
        out[f"{label}_bias_pct"] = float(
            100.0 * (pred[idx].mean() - y[idx].mean()) / y[idx].mean()
        )
        out[f"{label}_crps"] = c
        out[f"{label}_rmse"] = rmse(y[idx], pred[idx])
        out[f"{label}_cov80"] = (
            interval_coverage(draws[:, idx], y[idx], 0.80)
            if draws is not None
            else float("nan")
        )
        out[f"{label}_idx"] = idx
    return out


def self_consistent_top(pred: np.ndarray, draws: np.ndarray, k: int = TOP_K,
                        reps: int = 500, seed: int = 0) -> dict:
    """What a correctly-calibrated version of *this* model would score on top-k.

    Neither top-of-board bias has a target of zero. Both selections condition on
    a quantity correlated with the error, so even a perfect forecaster shows
    bias - negative when selecting by actual, positive when selecting by
    projection - and the magnitude is monotone in the predictive *dispersion*,
    not in the accuracy. Comparing a model's raw figure to a baseline's raw
    figure therefore rewards over-dispersion, which is how this project got the
    metric wrong once already.

    The reference that needs no refit: treat each posterior predictive draw as
    one synthetic season ``Y*``, re-select the top k by ``Y*``, and average
    ``mean(pred) - mean(Y*)`` over draws. Under the model's own assumptions that
    is exactly the number the observed statistic is drawn from.

    **Observed minus self-consistent is the shrinkage signal**: negative means
    the model really does under-project the top, positive means it does not.
    """
    draws = np.asarray(draws, float)
    pred = np.asarray(pred, float)
    n_draws = draws.shape[0]
    rng = np.random.default_rng(seed)
    sel = rng.choice(n_draws, size=min(reps, n_draws), replace=False)

    ip = np.argsort(-pred, kind="stable")[:k]
    sc_a = np.empty(len(sel))
    sc_p = np.empty(len(sel))
    sc_a_lvl = np.empty(len(sel))
    for j, r in enumerate(sel):
        ystar = draws[r]
        ja = np.argsort(-ystar, kind="stable")[:k]
        sc_a[j] = pred[ja].mean() - ystar[ja].mean()
        sc_a_lvl[j] = ystar[ja].mean()
        sc_p[j] = pred[ip].mean() - ystar[ip].mean()
    return {
        "sc_actual_bias": float(sc_a.mean()),
        "sc_actual_bias_pct": float(100.0 * sc_a.mean() / sc_a_lvl.mean()),
        "sc_actual_lo": float(np.quantile(sc_a, 0.05)),
        "sc_actual_hi": float(np.quantile(sc_a, 0.95)),
        "sc_proj_bias": float(sc_p.mean()),
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
