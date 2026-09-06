"""Availability model: how many games the player is on the field for.

Season points are games times points per game, and for fantasy purposes the
games half is not a nuisance term - it is most of the downside risk. A quarter
of the players in any season are out of the league the next one, and a
sizeable share of the rest miss time. A model that projects only per-game
ability and multiplies by 17 will be wrong in one direction for nearly
everybody.

Two questions, so two parts (a hurdle):

* **Is he in the league at all?** A Bernoulli on "zero games next season".
  A player who disappears is not injured for a whole season; he is a different
  outcome, and the spike at zero is far too sharp for any count distribution to
  reproduce on its own.
* **Given he plays, how much?** Beta-binomial over the season's remaining
  slots. The extra dispersion over a binomial matters because missed games
  arrive in runs - a torn ACL costs eight straight weeks, not eight coin flips.
  The number of slots is a property of the season being predicted (15 games
  before 2021, 16 from 2021 on, after the final week is cut), so it is passed in
  per row rather than baked in as a constant.

Both parts share the same covariates, and their coefficients are partially
pooled across positions: each position gets its own value, drawn from a common
distribution, so quarterbacks (463 rows) are informed by the pooled estimate
where their own data runs thin while receivers (1,301 rows) largely speak for
themselves.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from bayes.laplace import fit_laplace

from bayes.production import N_POS

# Covariate names, in the order the design matrix is built.
#
# skill_sq and skill x games are here because the linear version was visibly
# wrong in backtest: it under-predicted dropout for below-average players
# (decile 2 predicted a 34% chance of vanishing against an actual 46%) while
# over-predicting it for stars (4% against 1%). Falling out of the league is
# not linear in ability - there is a cliff near replacement level and almost
# nothing above it - and the interaction lets a full season mean something
# different for a starter than for a player who was active but barely used.
#
# off_roster is the one covariate that does not come from a box score. It is
# whether the player was released, retired or unsigned at the start of the
# season being predicted - a fact settled at final cuts and known to anyone
# drafting, but invisible to a panel built from games played. Measured on the
# dev folds, the incumbent model over-projects retired players by 55.6 points
# and every off-roster status by at least 19, while under-projecting active
# players by 13.7: the mass it spends on players who will not appear is taken
# from the ones who will. See bayes.data.attach_preseason_context.
#
# depth_rank is where he sits on his team's chart going into the season. It
# survives the test that matters: conditioning on prior-season usage tertiles,
# the residual still slopes from +24 at rank 1 to -44 at rank 3 among
# high-usage players, so it is not a restatement of last year's workload.
#
# Its coverage does move with the era - the upstream format changed for 2024 -
# but the gradient sits almost entirely below the part of the board anyone
# drafts: 3.3 points of coverage difference across the top 24 and 3.6 across
# 25-50, against 17.6 for players outside the top 200. `depth_missing` carries
# whatever remains, so the era effect loads on the indicator rather than
# bending the rank coefficient.
FEATURES = ("age", "age_sq", "skill", "skill_sq", "skill_x_games", "games_frac",
            "rookie", "off_roster", "depth_r2", "depth_r3plus", "depth_missing")

# Indicators: centring one would make the intercept describe a player who does
# not exist, and scaling one changes what a unit of its coefficient means.
INDICATORS = ("rookie", "off_roster", "depth_r2", "depth_r3plus", "depth_missing")

# Depth rank enters as levels against a rank-1 baseline, not as a number.
# A linear term was tried first and mispriced the middle of the chart: being
# listed at all carries a large positive effect, and a straight line in rank is
# not steep enough to take it back from the twos and threes. On fold 2020 it
# turned a group the incumbent had almost exactly right (rank 2: 53.35
# projected against 53.41 actual) into an 8.6-point over-projection, and pushed
# rank-3 expected games *up* to 9.99 against an actual 7.56.
#
# The gaps are not evenly spaced - rank 1 is far clear of rank 2, while 2 and 3
# sit close together - so three indicators let the data set the spacing where
# one slope could not. Ranks past third are pooled: they all mean "buried", and
# the tail is too thin to give its own level.
DEPTH_CAP = 3.0

# off_roster and depth_missing are *nested*, not merely correlated: measured on
# the 2016-2022 training rows, no player is off a roster and still listed on a
# chart (0 rows of 3,717). So the pair encodes an ordinal ladder rather than two
# independent flags, and the three rungs separate cleanly -
#
#   rostered, on a chart      11.55 games,  3.7% zero
#   rostered, not on a chart   3.21 games, 53.3% zero
#   off a roster               1.14 games, 84.5% zero
#
# Both stay identified because 610 rows are unlisted while still rostered. But
# `off_roster`'s coefficient is the *increment* on top of being unlisted, never
# the whole effect of being out of the league; reading it as the latter would
# understate the gap by roughly the depth_missing coefficient.


def design_matrix(rows, skill: np.ndarray, ref: dict | None = None):
    """Build the standardised design matrix and the reference used to scale it.

    ``rows`` is a slice of the panel carrying ``age``, ``games`` and
    ``seasons_seen``; ``skill`` is the filtered ability estimate at the row's
    own season. Returns ``(X, ref)``; pass the same ``ref`` back when building
    the matrix for prediction rows so the coefficients keep their meaning.
    """
    age = rows["age"].to_numpy(float)
    age = np.where(np.isnan(age), np.nanmedian(age), age)
    # Age of the season being predicted, not the season observed.
    age = age + 1.0

    # Share of *his own* season played. Pre-built on the panel because the
    # denominator differs by era; falling back to the row's season_games keeps
    # this usable on a frame that has not been through build_panel.
    games_frac = (
        rows["games_frac"].to_numpy(float)
        if "games_frac" in rows
        else rows["games"].to_numpy(float) / rows["season_games"].to_numpy(float)
    )

    # The nonlinear terms are built from a skill measure standardised against
    # the *training* rows, so that "skill squared" means the same thing at
    # prediction time. Squaring a raw ability score and standardising the
    # result afterwards would not be the same function.
    if ref is None:
        s_mu, s_sd = float(skill.mean()), float(skill.std()) or 1.0
    else:
        s_mu, s_sd = ref["skill_mu"], ref["skill_sd"]
    s = (skill - s_mu) / s_sd

    # Missing means "no roster information", not "unemployed" - defaulting an
    # unmatched player to off-roster would manufacture the signal this feature
    # exists to measure. A frame that never went through attach_preseason_context
    # therefore contributes a column of zeros and the coefficient simply has
    # nothing to fit, rather than the model silently reading every player as
    # rostered-and-known.
    off_roster = (
        (rows["next_on_roster"].to_numpy(dtype=float) == 0.0).astype(float)
        if "next_on_roster" in rows
        else np.zeros(len(rows))
    )

    # Mean-imputation plus an explicit missingness flag. Filling with the
    # training mean keeps an unlisted player from being scored as though he
    # were buried, which he may not be - a chart lists the top few at each
    # position and says nothing about the rest - while the flag lets the model
    # price "not listed" separately from any particular rank. The fill comes
    # from `ref` at prediction time so the column means the same thing there.
    depth = (
        rows["next_depth_rank"].to_numpy(dtype=float)
        if "next_depth_rank" in rows
        else np.full(len(rows), np.nan)
    )
    depth_missing = np.isnan(depth).astype(float)
    depth = np.clip(depth, 1.0, DEPTH_CAP)
    # Rank 1 is the baseline and is carried by the intercept; an unlisted player
    # is off all three rank levels and picked up by depth_missing alone, so the
    # four states stay distinguishable without a redundant fourth column.
    depth_r2 = (depth == 2.0).astype(float)
    depth_r3plus = (depth >= 3.0).astype(float)

    raw = np.column_stack(
        [
            age,
            age ** 2,
            s,
            s ** 2,
            s * games_frac,
            games_frac,
            (rows["seasons_seen"].to_numpy() == 0).astype(float),
            off_roster,
            depth_r2,
            depth_r3plus,
            depth_missing,
        ]
    )

    if ref is None:
        center = raw.mean(axis=0)
        scale = raw.std(axis=0)
        scale[scale == 0] = 1.0
        for name in INDICATORS:
            i = FEATURES.index(name)
            center[i], scale[i] = 0.0, 1.0
        ref = {"center": center, "scale": scale, "skill_mu": s_mu,
               "skill_sd": s_sd}

    return (raw - ref["center"]) / ref["scale"], ref


def _pooled_coefs(name: str, n_feat: int):
    """Position-level coefficients drawn around a shared mean.

    Non-centred: the raw offsets are standard normal and the scale multiplies
    them, which is the parameterisation NUTS can actually move through when a
    group's data is thin enough to pull its scale toward zero.
    """
    mu = numpyro.sample(f"{name}_mu", dist.Normal(0.0, 1.0).expand([n_feat]).to_event(1))
    tau = numpyro.sample(f"{name}_tau", dist.HalfNormal(0.5).expand([n_feat]).to_event(1))
    raw = numpyro.sample(
        f"{name}_raw", dist.Normal(0.0, 1.0).expand([N_POS, n_feat]).to_event(2)
    )
    return numpyro.deterministic(name, mu + tau * raw)


def availability_model(X, pos_idx, season_idx, n_seasons, n_slots, next_games=None):
    n_feat = X.shape[1]

    # ------------------------------------------------- part 1: out of the league
    zero_int = numpyro.sample(
        "zero_int", dist.Normal(-1.0, 1.5).expand([N_POS]).to_event(1)
    )
    zero_beta = _pooled_coefs("zero_beta", n_feat)

    # League-wide year effect on turnover. The share of players who vanish is
    # not a constant - it ran 0.25, 0.28, 0.28, 0.25, 0.32 over the observed
    # transitions - and no covariate available in August explains which kind of
    # year is coming. Modelling it as a random year effect means a projection
    # inherits that uncertainty instead of quietly asserting the training
    # average, which is the difference between an interval that covers and one
    # that does not.
    season_sd = numpyro.sample("season_sd", dist.HalfNormal(0.3))
    season_raw = numpyro.sample(
        "season_raw", dist.Normal(0.0, 1.0).expand([n_seasons]).to_event(1)
    )
    season_effect = numpyro.deterministic("season_effect", season_sd * season_raw)

    logit_zero = (
        zero_int[pos_idx]
        + jnp.sum(X * zero_beta[pos_idx], axis=-1)
        + season_effect[season_idx]
    )

    # -------------------------------------------- part 2: share of the season
    play_int = numpyro.sample(
        "play_int", dist.Normal(1.0, 1.5).expand([N_POS]).to_event(1)
    )
    play_beta = _pooled_coefs("play_beta", n_feat)
    logit_p = play_int[pos_idx] + jnp.sum(X * play_beta[pos_idx], axis=-1)
    p = jax.nn.sigmoid(logit_p)

    # Concentration of the beta-binomial. Small values mean games missed come
    # in clumps rather than independently week to week.
    conc = numpyro.sample(
        "conc", dist.LogNormal(1.0, 0.7).expand([N_POS]).to_event(1)
    )
    c = conc[pos_idx]

    if next_games is None:
        played = numpyro.sample("played", dist.Bernoulli(logits=-logit_zero))
        extra = numpyro.sample(
            "extra",
            dist.BetaBinomial(p * c, (1 - p) * c, n_slots),
        )
        numpyro.deterministic("next_games", played * (1 + extra))
        return

    played = (next_games > 0).astype(X.dtype)
    numpyro.sample("zero_obs", dist.Bernoulli(logits=-logit_zero), obs=played)

    # Conditional on playing at all, the games above the first - at most
    # `n_slots`, which is the length of the season being predicted minus one and
    # varies across the panel. Rows with zero games contribute nothing here:
    # masking them out is what makes this the *conditional* distribution rather
    # than a second look at the same zeros the hurdle already explained.
    with numpyro.handlers.mask(mask=played > 0):
        numpyro.sample(
            "games_obs",
            dist.BetaBinomial(p * c, (1 - p) * c, n_slots),
            obs=jnp.clip(next_games - 1, 0, n_slots),
        )


def fit_availability(
    X,
    pos_idx,
    season_idx,
    n_seasons,
    next_games,
    n_slots,
    num_warmup: int = 800,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 1,
    progress: bool = True,
    inference: str = "nuts",
):
    """Fit the hurdle + beta-binomial, by NUTS or by MAP + Laplace.

    This half is the riskier one for a Gaussian approximation: the pooled
    coefficients are non-centred, so their scales (`zero_beta_tau`,
    `play_beta_tau`) can have posteriors that pile up against zero, and a mode
    -based Gaussian has nothing to say about a density with no interior mode.
    `fit_laplace` reports the Hessian's smallest eigenvalue and the importance
    weight ESS for exactly this reason - read them before trusting the fit.
    """
    if inference == "laplace":
        return fit_laplace(
            availability_model,
            jnp.asarray(X), jnp.asarray(pos_idx), jnp.asarray(season_idx),
            n_seasons, jnp.asarray(np.asarray(n_slots, float)),
            next_games=jnp.asarray(np.asarray(next_games, float)),
            num_samples=num_samples, seed=seed,
        )
    if inference != "nuts":
        raise ValueError(f"unknown inference {inference!r}")
    kernel = NUTS(availability_model, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        # Vectorised, not sequential: the four chains run as one compiled
        # program instead of a Python loop over four. It is an execution
        # change and cannot move the target - verified against a 958s
        # sequential reference on fold 2020, where it reproduced E[games] to
        # -0.07% and P(zero) to +0.05pp, both inside the sampler's own
        # split-half noise (0.135% and 0.13pp), with a *higher* minimum
        # effective sample size (2022 against 1967) in 21 seconds.
        chain_method="vectorized",
        progress_bar=progress,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        jnp.asarray(X),
        jnp.asarray(pos_idx),
        jnp.asarray(season_idx),
        n_seasons,
        jnp.asarray(np.asarray(n_slots, float)),
        next_games=jnp.asarray(np.asarray(next_games, float)),
    )
    return mcmc


def predict_games(posterior: dict, X, pos_idx, key, n_slots):
    """Draw next-season games played, one draw per posterior sample per row.

    Returns an integer array of shape (draws, rows). Sampling rather than
    taking an expectation is the point: games and per-game scoring get
    multiplied together downstream, and a projection built from the product of
    two averages loses exactly the tail behaviour a fantasy manager is drafting
    for.

    The season being projected is one the model has never seen, so its year
    effect is drawn fresh from the estimated year-to-year distribution rather
    than set to zero. Setting it to zero would claim next season's league-wide
    turnover is already known to be average.
    """
    X = jnp.asarray(X)
    pos_idx = jnp.asarray(pos_idx)

    def one(zero_int, zero_beta, play_int, play_beta, conc, season_sd, k):
        k0, k1, k2 = jax.random.split(k, 3)
        new_season = season_sd * jax.random.normal(k0)
        lz = (
            zero_int[pos_idx]
            + jnp.sum(X * zero_beta[pos_idx], axis=-1)
            + new_season
        )
        lp = play_int[pos_idx] + jnp.sum(X * play_beta[pos_idx], axis=-1)
        p, c = jax.nn.sigmoid(lp), conc[pos_idx]
        plays = dist.Bernoulli(logits=-lz).sample(k1)
        extra = dist.BetaBinomial(p * c, (1 - p) * c, n_slots).sample(k2)
        return plays * (1 + extra)

    n = posterior["zero_int"].shape[0]
    keys = jax.random.split(key, n)
    # jit, not bare vmap. vmap alone builds the batched computation but never
    # compiles it, so every primitive is dispatched separately for each of the
    # thousands of draws; wrapping it hands XLA one program instead. Measured
    # cold in a fresh process on fold 2020, this is 10s against 565s for the
    # unwrapped version, and the draws are bit-identical - it is the same
    # arithmetic, executed once rather than interpreted.
    return jax.jit(jax.vmap(one))(
        posterior["zero_int"],
        posterior["zero_beta"],
        posterior["play_int"],
        posterior["play_beta"],
        posterior["conc"],
        posterior["season_sd"],
        keys,
    )
