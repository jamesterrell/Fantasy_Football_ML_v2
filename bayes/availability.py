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
* **Given he plays, how much?** Beta-binomial over the 17-game season. The
  extra dispersion over a binomial matters because missed games arrive in runs
   - a torn ACL costs eight straight weeks, not eight coin flips.

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

from bayes.data import SEASON_GAMES
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
FEATURES = ("age", "age_sq", "skill", "skill_sq", "skill_x_games", "games_frac", "rookie")


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

    games_frac = rows["games"].to_numpy(float) / SEASON_GAMES

    # The nonlinear terms are built from a skill measure standardised against
    # the *training* rows, so that "skill squared" means the same thing at
    # prediction time. Squaring a raw ability score and standardising the
    # result afterwards would not be the same function.
    if ref is None:
        s_mu, s_sd = float(skill.mean()), float(skill.std()) or 1.0
    else:
        s_mu, s_sd = ref["skill_mu"], ref["skill_sd"]
    s = (skill - s_mu) / s_sd

    raw = np.column_stack(
        [
            age,
            age ** 2,
            s,
            s ** 2,
            s * games_frac,
            games_frac,
            (rows["seasons_seen"].to_numpy() == 0).astype(float),
        ]
    )

    if ref is None:
        center = raw.mean(axis=0)
        scale = raw.std(axis=0)
        scale[scale == 0] = 1.0
        # The rookie flag is an indicator; centring it would make the intercept
        # mean something no player is.
        center[-1], scale[-1] = 0.0, 1.0
        ref = {"center": center, "scale": scale, "skill_mu": s_mu, "skill_sd": s_sd}

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


def availability_model(X, pos_idx, season_idx, n_seasons, next_games=None):
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
            dist.BetaBinomial(p * c, (1 - p) * c, SEASON_GAMES - 1),
        )
        numpyro.deterministic("next_games", played * (1 + extra))
        return

    played = (next_games > 0).astype(X.dtype)
    numpyro.sample("zero_obs", dist.Bernoulli(logits=-logit_zero), obs=played)

    # Conditional on playing at all, the remaining 0-16 games above the first.
    # Rows with zero games contribute nothing here: masking them out is what
    # makes this the *conditional* distribution rather than a second look at
    # the same zeros the hurdle already explained.
    with numpyro.handlers.mask(mask=played > 0):
        numpyro.sample(
            "games_obs",
            dist.BetaBinomial(p * c, (1 - p) * c, SEASON_GAMES - 1),
            obs=jnp.clip(next_games - 1, 0, SEASON_GAMES - 1),
        )


def fit_availability(
    X,
    pos_idx,
    season_idx,
    n_seasons,
    next_games,
    num_warmup: int = 800,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 1,
    progress: bool = True,
) -> MCMC:
    kernel = NUTS(availability_model, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=progress,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        jnp.asarray(X),
        jnp.asarray(pos_idx),
        jnp.asarray(season_idx),
        n_seasons,
        next_games=jnp.asarray(np.asarray(next_games, float)),
    )
    return mcmc


def predict_games(posterior: dict, X, pos_idx, key):
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
        extra = dist.BetaBinomial(p * c, (1 - p) * c, SEASON_GAMES - 1).sample(k2)
        return plays * (1 + extra)

    n = posterior["zero_int"].shape[0]
    keys = jax.random.split(key, n)
    return jax.vmap(one)(
        posterior["zero_int"],
        posterior["zero_beta"],
        posterior["play_int"],
        posterior["play_beta"],
        posterior["conc"],
        posterior["season_sd"],
        keys,
    )
