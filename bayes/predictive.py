"""Fit both halves of the model on seasons up to a cutoff and project forward.

The product is a set of draws from the posterior predictive distribution of
next season's PPR total for every player who appeared in the cutoff season -
not a number, a distribution. Everything downstream (point projections,
floors and ceilings, "probability he beats 200 points") is a summary of these
draws.

The two halves are fit separately and joined by simulation. That is a
deliberate modularisation, not an oversight: feeding the production model's
uncertainty into the availability model would let a player's *games missed*
feed back and revise the estimate of how good he is, and injury-shortened
seasons would then quietly drag down ability estimates. Cutting the feedback
keeps each half answering its own question. The cost is that the availability
model treats the ability estimate it is handed as known, which understates its
uncertainty slightly - small next to the spread the games distribution
contributes anyway.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bayes.availability import design_matrix, fit_availability, predict_games
from bayes.data import (
    PRESEASON_COLUMNS,
    PPG_FLOOR,
    add_missed_seasons,
    attach_established,
    attach_next_season,
    measurement_noise,
    season_length,
)
from bayes.production import (
    POS_INDEX,
    build_arrays,
    filtered_states,
    fit_production,
)


@dataclass
class Projection:
    """Predictive draws plus the rows they describe."""

    rows: pd.DataFrame           # one row per player projected
    draws: np.ndarray            # [n_draws, n_rows] next-season PPR totals
    games_draws: np.ndarray      # [n_draws, n_rows] next-season games played
    ability: np.ndarray          # [n_draws, n_rows] latent sqrt(PPG) ability
    production_mcmc: object = None
    availability_mcmc: object = None
    spline_spec: dict = None     # aging-curve basis, needed to redraw the curve
    # Wall-clock seconds for each stage of the fold, so a run reports where the
    # time went rather than only how long it took.
    timings: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Point projection and predictive quantiles, per player."""
        d = self.draws
        out = self.rows.copy()
        out["proj_mean"] = d.mean(axis=0)
        out["proj_median"] = np.median(d, axis=0)
        for q, name in [(0.05, "p05"), (0.25, "p25"), (0.75, "p75"), (0.95, "p95")]:
            out[name] = np.quantile(d, q, axis=0)
        out["exp_games"] = self.games_draws.mean(axis=0)
        out["p_misses_season"] = (self.games_draws == 0).mean(axis=0)
        out["proj_ppg"] = np.where(
            out["exp_games"] > 0, out["proj_mean"] / out["exp_games"].clip(lower=1e-9), 0.0
        )
        return out.sort_values("proj_mean", ascending=False).reset_index(drop=True)


def _skill_at_rows(rows, players, seasons, level, state):
    """Filtered ability (posterior mean) for each row, at the row's own season.

    ``level`` and ``state`` are (draws, players, seasons); this pulls the entry
    matching each row's player and season and averages over draws.
    """
    p_index = {a: i for i, a in enumerate(players)}
    s_index = {s: i for i, s in enumerate(seasons)}
    pi = rows["athlete_id"].map(p_index).to_numpy()
    si = rows["season"].map(s_index).to_numpy()
    theta = np.asarray(level)[:, pi, si] + np.asarray(state)[:, pi, si]
    return theta.mean(axis=0)


def fit_fold(
    panel: pd.DataFrame,
    cutoff: int,
    use_inputs: bool = True,
    num_warmup: int = 800,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    progress: bool = False,
    roster_ids: set | None = None,
    inference_production: str = "nuts",
    inference_availability: str = "nuts",
) -> Projection:
    """Train on seasons <= ``cutoff``, project season ``cutoff + 1``.

    ``inference_production`` and ``inference_availability`` are independent on
    purpose. The two halves are different shapes - the production posterior is a
    marginal over ~56 hyperparameters with the abilities integrated out by the
    Kalman filter, while the availability half carries non-centred scales that
    can pile up against zero - so there is no reason to assume a Gaussian
    approximation is equally good for both. Being able to run ``laplace`` on one
    and ``nuts`` on the other is the point.

    Nothing from season ``cutoff + 1`` or later touches either fit: the panel is
    truncated first, the aging spline is built on the truncated ages, and the
    availability model only sees transitions whose outcome had already happened
    by the cutoff.

    ``roster_ids`` extends the projection set to players who missed the cutoff
    season entirely but are on a roster now - see :func:`add_missed_seasons`. It
    comes from a snapshot with no history, so it belongs to a live projection
    and never to a backtest fold, which is why it defaults to off.
    """
    # Refit the measurement-variance law on the training seasons only. It is a
    # minor nuisance parameter, but a backtest is only worth running if nothing
    # from the future reaches it.
    timings: dict = {}
    t_prep = time.perf_counter()
    train = measurement_noise(panel[panel["season"] <= cutoff].copy())

    # Missed seasons become explicit rows, and this happens *after* truncation
    # on purpose: `add_missed_seasons` bounds an interior gap by appearances in
    # the frame it is handed, so truncating first is what keeps a fold from
    # learning that a player came back in a season it has not reached.
    #
    # The new rows need next-season labels too - a zero-game season is a
    # perfectly good starting point for a transition, and teaching the
    # availability model what follows one is most of the reason these rows
    # exist.
    #
    # They are *filled in*, never rebuilt. Recomputing them here would silently
    # destroy the labels on the cutoff season itself: `attach_next_season` only
    # resolves rows before the last season it can see, and the frame it sees
    # here stops at the cutoff - so every scored row would come back NaN, which
    # is exactly what it did. The caller attaches labels while it can still see
    # one season past the cutoff, and those are the authority wherever they
    # exist. Only the gap rows arrive missing, and only those get filled.
    train = add_missed_seasons(train, roster_ids=roster_ids)
    filled = attach_next_season(
        train.drop(columns=["next_games", "next_fp_ppr"])
    )
    for col in ("next_games", "next_fp_ppr"):
        train[col] = train[col].fillna(filled[col])
    # The preseason columns have to be carried across too. Rows invented by
    # `add_missed_seasons` do not exist in the panel the caller attached context
    # to - a player who sat out last season has no row to attach it to - so they
    # arrive with no roster status at all. Left alone they read as "unknown",
    # which `design_matrix` treats as on-a-roster, and the projection then hands
    # a real number to players nobody employs: on the 2026 board that was
    # Brandon Aiyuk at 33 points and Joe Mixon at 31, both unsigned.
    for col in PRESEASON_COLUMNS:
        if col in filled.columns:
            train[col] = (
                train[col].fillna(filled[col]) if col in train.columns else filled[col]
            )

    # Established status, from prior seasons only and from this fold's frame
    # only - so the median it thresholds against contains nothing the fold has
    # not reached, the same rule the measurement-variance law follows. Computed
    # after the gap rows land so a missed season inherits a status rather than
    # resetting one.
    train = attach_established(train)

    # The grid runs one season past the cutoff. That extra column carries no
    # observation, so it adds nothing to the likelihood - but the filter
    # propagates through it, which means its filtered state *is* the one-step
    # -ahead forecast, aged correctly and with process noise already added.
    seasons = np.arange(train["season"].min(), cutoff + 2)
    arr = build_arrays(train, seasons=seasons, use_inputs=use_inputs)
    timings["prep_s"] = time.perf_counter() - t_prep

    # Both halves must return the same number of draws, because the projection
    # pairs them one-for-one rather than crossing them. NUTS returns
    # ``num_samples * num_chains``; Laplace returns whatever it is asked for and
    # its draws are nearly free, so it is asked for the same total. Without this
    # a mixed run (laplace production, NUTS availability) fails the shape assert
    # below - and a matched pair of counts is also what keeps a Laplace fold
    # comparable to a NUTS fold at the same Monte Carlo resolution.
    n_draws = num_samples * num_chains

    t0 = time.perf_counter()
    prod = fit_production(
        arr,
        use_inputs=use_inputs,
        num_warmup=num_warmup,
        num_samples=n_draws if inference_production == "laplace" else num_samples,
        num_chains=num_chains,
        seed=seed,
        progress=progress,
        inference=inference_production,
    )
    timings["production_fit_s"] = time.perf_counter() - t0
    post = prod.get_samples()
    t0 = time.perf_counter()
    state, state_var, level = filtered_states(arr, post, use_inputs=use_inputs)
    state = np.asarray(state)
    state_var = np.asarray(state_var)
    level = np.asarray(level)
    timings["filtered_states_s"] = time.perf_counter() - t0

    # ------------------------------------------------ availability training set
    # Every transition whose outcome is known by the cutoff. The skill covariate
    # is the filtered estimate at the row's own season, which by construction
    # used no season after it.
    avail_rows = train[train["season"] < cutoff].copy()
    t0 = time.perf_counter()
    avail_skill = _skill_at_rows(avail_rows, arr.players, seasons, level, state)
    X_train, ref = design_matrix(avail_rows, avail_skill)
    timings["design_train_s"] = time.perf_counter() - t0
    pos_train = avail_rows["pos"].map(POS_INDEX).to_numpy()

    train_seasons = np.sort(avail_rows["season"].unique())
    season_idx = np.searchsorted(train_seasons, avail_rows["season"].to_numpy())

    # Slots above the first game, for the season each row's outcome lands in.
    # 14 before 2021 and 15 after: a beta-binomial with the wrong ceiling would
    # read every complete 2016-2020 season as one game short and push `p` down
    # for the whole era.
    n_slots_train = (
        avail_rows["season"].add(1).map(season_length).to_numpy(float) - 1.0
    )

    t0 = time.perf_counter()
    avail = fit_availability(
        X_train,
        pos_train,
        season_idx,
        len(train_seasons),
        avail_rows["next_games"].to_numpy(),
        n_slots_train,
        num_warmup=num_warmup,
        num_samples=n_draws if inference_availability == "laplace" else num_samples,
        num_chains=num_chains,
        seed=seed + 1,
        progress=progress,
        inference=inference_availability,
    )
    timings["availability_fit_s"] = time.perf_counter() - t0
    avail_post = avail.get_samples()

    # ------------------------------------------------------------ project rows
    rows = train[train["season"] == cutoff].copy().reset_index(drop=True)
    p_index = {a: i for i, a in enumerate(arr.players)}
    pi = rows["athlete_id"].map(p_index).to_numpy()

    # Last grid column: the one-step-ahead predictive for ability in cutoff + 1.
    theta_mean = level[:, pi, -1] + state[:, pi, -1]
    theta_var = state_var[:, pi, -1]

    # The availability model's skill covariate must mean the same thing here as
    # it did in training: the ability filtered at the row's *own* season. The
    # forecast for the following season is a different quantity - it is already
    # shrunk toward the cohort by rho and re-aged - and feeding it in here would
    # hand the fitted coefficients a compressed version of the variable they
    # were fit on, flattening exactly the dropout gradient the skill terms exist
    # to capture.
    t0 = time.perf_counter()
    pred_skill = _skill_at_rows(rows, arr.players, seasons, level, state)
    X_pred, _ = design_matrix(rows, pred_skill, ref=ref)
    timings["design_pred_s"] = time.perf_counter() - t0
    pos_pred = rows["pos"].map(POS_INDEX).to_numpy()

    key = jax.random.PRNGKey(seed + 2)
    k_games, k_theta, k_obs = jax.random.split(key, 3)

    target_games = season_length(cutoff + 1)
    t0 = time.perf_counter()
    games = np.asarray(
        predict_games(avail_post, X_pred, pos_pred, k_games, target_games - 1)
    )
    timings["predict_games_s"] = time.perf_counter() - t0

    draws, rows_n = theta_mean.shape
    # Match the availability draws to the production draws one-for-one. Both
    # posteriors have the same number of samples by construction; pairing them
    # rather than crossing them keeps the joint draw count manageable and the
    # dependence structure (none, by the modularisation above) explicit.
    assert games.shape == (draws, rows_n), (games.shape, (draws, rows_n))

    ability = np.asarray(
        theta_mean
        + np.sqrt(np.maximum(theta_var, 0.0))
        * np.asarray(jax.random.normal(k_theta, (draws, rows_n)))
    )

    t0 = time.perf_counter()
    total = _realise_season(
        ability,
        games,
        rows,
        np.asarray(post["kappa"]),
        np.asarray(post["lam"]),
        pos_pred,
        k_obs,
        target_games,
    )
    timings["realise_s"] = time.perf_counter() - t0

    return Projection(
        rows=rows,
        draws=total,
        games_draws=games,
        ability=ability,
        production_mcmc=prod,
        availability_mcmc=avail,
        spline_spec=arr.spline_spec,
        timings=timings,
    )


def _realise_season(ability, games, rows, kappa, lam, pos_idx, key, season_games):
    """Turn latent ability and a games count into a realised season total.

    Two things happen here that a "projected points per game times projected
    games" calculation misses.

    First, the games actually drawn feed back into the per-game rate through
    the missed-time offset. A player who lands on 7 games in a given draw is
    not his healthy self for those 7 - and because that pulls both factors of
    the product down together, the average of the product ends up correctly
    above the product of the averages.

    Second, a player's realised season is his ability plus a season's worth of
    luck, and a 5-game season is far luckier - in both directions - than a
    17-game one. The same variance law the panel was built with supplies that,
    so a projection for a player expected to miss time comes out correctly
    wide rather than merely low.
    """
    var_a = rows["var_a"].to_numpy()[None, :]
    var_b = rows["var_b"].to_numpy()[None, :]

    g = np.maximum(games, 1)                               # avoid 0-division; masked below
    # `season_games` is the length of the season being projected, which is what
    # "a full season" means for this draw - not the length of the seasons the
    # model was fit on.
    ability = ability + lam[:, pos_idx] * (g - season_games) / season_games

    mu = np.maximum(ability, 0.0) ** 2                     # latent points per game

    within = (var_a + var_b * mu) / (4.0 * g * np.maximum(mu, PPG_FLOOR))
    z_var = kappa[:, pos_idx] * within

    noise = np.asarray(jax.random.normal(key, ability.shape))
    z_obs = ability + np.sqrt(np.maximum(z_var, 0.0)) * noise
    ppg = np.maximum(z_obs, 0.0) ** 2

    return np.where(games > 0, games * ppg, 0.0)
