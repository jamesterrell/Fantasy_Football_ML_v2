"""Production model: how many points per game, given that the player plays.

The model is a hierarchical Gaussian state-space model on

    z = sqrt(points per game)

with one latent ability per player per season. Writing it as a state-space
model rather than a regression on last season's numbers buys three things that
matter for fantasy projection:

1. **Shrinkage proportional to sample size, for free.** A 3-game season is a
   loud measurement of ability and a 17-game season is a quiet one, and the
   panel supplies the noise level of each directly (``z_var``). The Kalman
   update weights every season by exactly that, so a back who averaged 18 PPG
   over four games is pulled toward his cohort far harder than one who did it
   over sixteen - without a hand-tuned "minimum games" rule.

2. **The whole career, not just last season.** Ability carries forward through
   an AR(1), so three consistent seasons speak louder than one, and a player
   who missed a year is carried across the gap with widened uncertainty rather
   than dropped or imputed.

3. **Honest uncertainty.** The filtered variance knows the difference between
   "reliably mediocre" and "barely observed", and that difference is what turns
   a point projection into a floor and a ceiling.

The latent abilities are marginalised out analytically by the Kalman filter, so
NUTS samples only the ~30 hyperparameters. That is what makes the model fit in
seconds rather than fighting 6,000 correlated latent variables.

Structure, for a player i of position p in season t:

    level     m[i,t] = base[p] + agecurve[p](age[i,t])
    state     z*[i,t] - m[i,t] = rho[p] * (z*[i,t-1] - m[i,t-1]) + noise
    observed  z[i,t] ~ Normal(z*[i,t], kappa[p] * z_var[i,t])

so the AR(1) acts on a player's deviation from what his position and age would
predict. The aging curve moves the whole cohort; the state carries where he
sits within it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import lax
from numpyro.infer import MCMC, NUTS

from bayes.data import POSITIONS
from bayes.laplace import fit_laplace
from bayes.spline import apply_spline_basis, natural_spline_basis

# Knot placement for the aging curve, in years. Fixed rather than fit from
# quantiles of whichever slice of seasons is in hand, so the same curve is
# comparable across backtest folds.
AGE_KNOTS = np.array([22.0, 24.5, 27.0, 30.0, 34.0])
AGE_CLIP = (21.0, 39.0)

N_POS = len(POSITIONS)
POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}


@dataclass
class PanelArrays:
    """The panel reshaped to a dense (player x season) grid for the filter."""

    players: np.ndarray          # [P] athlete_id
    pos_idx: np.ndarray          # [P] position index
    seasons: np.ndarray          # [T] season labels, ascending and contiguous
    z: np.ndarray                # [P,T] sqrt(points per game), 0 where unseen
    obs: np.ndarray              # [P,T] True where the player played
    zvar: np.ndarray             # [P,T] measurement variance of z, 1 where unseen
    games_frac: np.ndarray       # [P,T] share of the season played, 1 where unseen
    age_basis: np.ndarray        # [P,T,K] aging-curve basis, known every season
    start: np.ndarray            # [P,T] True at the player's first season seen
    est: np.ndarray              # [P,T] True where he counts as established
    ctrl: np.ndarray             # [P,T,C] inputs applied moving *into* season t
    spline_spec: dict = field(repr=False, default_factory=dict)
    ctrl_names: tuple = ()

    @property
    def n_players(self) -> int:
        return len(self.players)

    @property
    def n_seasons(self) -> int:
        return len(self.seasons)


def _impute_age(panel, seasons) -> np.ndarray:
    """Fill unknown ages with the position's median for that season."""
    age = panel["age"].to_numpy(float).copy()
    if np.isnan(age).any():
        med = panel.groupby("pos")["age"].transform("median").to_numpy(float)
        age = np.where(np.isnan(age), med, age)
    return np.clip(age, *AGE_CLIP)


# Season-level signals carried into the *next* season's transition. Both are
# things a single season total cannot say on its own.
CTRL_NAMES = ("usage", "late_form", "depth_rank", "moved", "qb_change",
              "rb_room")


def build_arrays(
    panel,
    seasons: np.ndarray | None = None,
    spline_spec: dict | None = None,
    use_inputs: bool = True,
) -> PanelArrays:
    """Reshape a season-grain panel into the dense grid the filter scans over.

    ``seasons`` covers the observed range plus, at prediction time, one season
    past it - the grid has to contain the season being forecast so the aging
    curve is evaluated at the age the player will actually be.
    """
    panel = panel.sort_values(["athlete_id", "season"]).reset_index(drop=True)

    if seasons is None:
        seasons = np.arange(panel["season"].min(), panel["season"].max() + 1)
    seasons = np.asarray(seasons)

    players = panel["athlete_id"].drop_duplicates().to_numpy()
    p_index = {a: i for i, a in enumerate(players)}
    s_index = {s: i for i, s in enumerate(seasons)}

    pos_by_player = (
        panel.drop_duplicates("athlete_id").set_index("athlete_id")["pos"]
    )
    pos_idx = np.array([POS_INDEX[pos_by_player[a]] for a in players])

    P, T = len(players), len(seasons)
    C = len(CTRL_NAMES)
    z = np.zeros((P, T))
    obs = np.zeros((P, T), bool)
    zvar = np.ones((P, T))
    # Full season where unseen, so the missed-time offset below is exactly zero
    # for every cell that carries no observation. Held as a *share* of the
    # season rather than a count, because the season is 15 games before 2021 and
    # 16 from 2021 on; a count would make "one game missed" a different quantity
    # in the two eras and the offset would absorb the difference as ability.
    games_frac = np.ones((P, T))
    ctrl = np.zeros((P, T, C))

    pi = panel["athlete_id"].map(p_index).to_numpy()
    si = panel["season"].map(s_index).to_numpy()

    # A row is an observation only if the player actually played that season.
    # Once missed seasons are represented as explicit zero-game rows, "has a
    # row" and "was observed" are different things, and conflating them would
    # feed the filter a scoring rate of zero as though it were measured.
    played = (
        panel["played"].to_numpy(bool)
        if "played" in panel
        else np.ones(len(panel), bool)
    )

    z[pi, si] = panel["z"].to_numpy()
    obs[pi, si] = played
    zvar[pi, si] = panel["z_var"].to_numpy()
    # Leave unobserved cells at a full season so the missed-time offset is
    # exactly zero there; the filter gates the level on `obs` anyway, so this
    # only guards against the value being read somewhere it should not be.
    games_frac[pi[played], si[played]] = panel["games_frac"].to_numpy()[played]

    # First season each player is seen. The filter restarts the state there
    # rather than carrying a state that does not yet exist.
    start = np.zeros((P, T), bool)
    first = np.full(P, -1)
    for p in range(P):
        seen = np.flatnonzero(obs[p])
        if len(seen):
            first[p] = seen[0]
            start[p, seen[0]] = True

    # ------------------------------------------------------- established status
    # Carried forward across the grid rather than read cell by cell. Two reasons
    # it has to be: a season the player missed has no row to read it from, and
    # the last grid column - the one whose filtered state *is* next season's
    # forecast - has no row at all. Leaving that column at False would forecast
    # every established player with the unestablished persistence, which is
    # precisely the shrinkage this split exists to stop.
    est = np.zeros((P, T), bool)
    if "established" in panel:
        est[pi, si] = panel["established"].to_numpy(bool)
    np.maximum.accumulate(est, axis=1, out=est)

    # ------------------------------------------------------------- aging curve
    # Age is known for every player in every season of the grid, including ones
    # he did not play, because it comes from a birth date rather than a box
    # score. That is what lets the model age a player across a missed season.
    # Held as (age - season) per player so it can be evaluated at any season on
    # the grid, including one the player has not played.
    birth_age = np.full(P, np.nan)
    imputed = _impute_age(panel, seasons)
    birth_age[pi] = imputed - seasons[si]
    ages = np.clip(birth_age[:, None] + seasons[None, :], *AGE_CLIP)

    flat = ages.reshape(-1)
    if spline_spec is None:
        basis, spline_spec = natural_spline_basis(flat, AGE_KNOTS)
    else:
        basis = apply_spline_basis(flat, spline_spec)
    age_basis = basis.reshape(P, T, -1)

    # --------------------------------------------------------------- inputs
    # Both channels are standardised within position and season, then carried
    # on the transition *out of* the season they were measured in - they are
    # known at the end of season t and inform where the player lands in t + 1.
    #
    #  usage:     opportunity per game. Volume repeats better than the
    #             efficiency layered on top of it, so a back who saw 15 carries
    #             a game on modest yardage is a different bet next season than
    #             one who matched his points on 6.
    #  late_form: how the closing stretch compared with the season as a whole,
    #             which is where a mid-season change of role shows up.
    #  depth_rank: where he is listed on the chart going into the season being
    #             moved into. Unlike the other two this is not a box-score
    #             quantity - it is the one thing here known *before* the season
    #             rather than measured during the previous one, which is what
    #             lets it speak to a role that is about to change.
    #
    # Depth rank earns its place almost entirely at quarterback. Measured on
    # the panel, the change in points per game runs +0.03 / -1.69 / -2.09 from
    # rank 1 to rank 3 for QBs, against a flat -0.52 / -0.51 / -0.48 for backs
    # and similarly flat lines for receivers and ends: a backup quarterback who
    # gets on the field is playing relief, while a backup back who plays simply
    # gets his touches. `gamma` is per position, so the model can price it that
    # way without being told to.
    #
    # There is no companion missingness channel. Being unlisted is mostly a
    # statement about whether he plays at all, which is the availability
    # model's question, and its effect on the *rate* does not even share a sign
    # across positions (+0.58 for QBs, -0.92 for receivers). Unlisted rows are
    # mean-imputed instead, which standardises to zero and lets the channel say
    # nothing about them.
    if use_inputs:
        def _standardise(values):
            s = pd.Series(values, index=panel.index)
            grp = s.groupby([panel["pos"], panel["season"]])
            mu, sd = grp.transform("mean"), grp.transform("std")
            out = np.where(sd > 0, (s - mu) / sd.where(sd > 0, 1.0), 0.0)
            # A missing input standardises to its own group's mean, i.e. zero.
            return np.nan_to_num(out, nan=0.0)

        # Capped for the same reason the availability model caps it: ranks past
        # fourth all mean "buried", and an uncapped tail would hand a few deep
        # reserves leverage over the coefficient.
        depth = (
            panel["next_depth_rank"].clip(upper=4.0)
            if "next_depth_rank" in panel
            else pd.Series(np.nan, index=panel.index)
        )

        #  snap_share: the share of his offence's snaps he was on the field for
        #             last season. Usage counts what he touched; this counts
        #             whether he was out there, and the two come apart exactly
        #             where it matters - a back on a third of the snaps who
        #             scored twice reads as a starter in a box score.
        #  team_vol:  the volume of the offence he is joining *next* season,
        #             measured last season. Passing pool for quarterbacks and
        #             receivers, rushing pool for backs. This is the column that
        #             knows a player changed teams: he inherits the new
        #             offence's plays, not his old one's.
        #  moved:     he is on a different team for the season being predicted.
        #  qb_change: the man throwing to him is different from the one who
        #             threw most for his team last season.
        #
        # Both are August facts from the roster and depth chart, and both are
        # binary because the continuous version of "what offence is he joining"
        # does not survive being lagged - see attach_situation for the numbers.
        # They are separate columns because they separate cleanly: with both in
        # one regression, receivers lose 1.23 ppg to the move and nothing to the
        # quarterback, while backs lose about a point to each.
        #  competition: how much of the ball is already claimed on the team he
        #             is joining - the summed prior workload of everyone else in
        #             his position group there. The only channel here that is
        #             about other people: a player's own history cannot contain
        #             the fact that his new team just signed someone ahead of
        #             him. Controlling for own prior rate, -0.046 ppg per rival
        #             target/game for receivers (t=-2.45) and ends (t=-2.63),
        #             -0.066 per rival carry/game for backs (t=-2.67).
        #  rb_room:   how good the *rest of his backfield* is - the summed prior
        #             ppg of the other backs on his August roster, zero for
        #             everyone else. Backs are graded on who is beside them, not
        #             on how many carries are nominally spoken for: with both
        #             terms in one regression, quality is -0.099 ppg per rival
        #             ppg (t=-2.62) and the carry count is +0.005 (t=+0.14).
        #             Out-of-sample it is worth -0.74% RMSE for backs, the same
        #             class as `usage` at -0.79%. It reorders the board where a
        #             volume term would not: ~19 backs a season move more than
        #             ten points, and in 2026 that is most of New Orleans.
        rb_room = panel["next_rb_rival_quality"] if "next_rb_rival_quality" in panel             else pd.Series(0.0, index=panel.index)

        # Competition was measured, wired in as a sixth channel, and taken back
        # out. It is the one quantity here a player's own history genuinely
        # cannot contain, and in a regression on raw ppg it is the strongest
        # thing found - but as a channel it moved fold-2021 CRPS from 22.050 to
        # 22.085, i.e. slightly the wrong way. Two candidate reasons, neither
        # tested: it reaches only 80% of skill rows so a fifth are imputed to
        # the mean, and `moved` may already carry most of it, since a player who
        # changes teams is the one whose competition changes. Worth revisiting
        # on four folds; `next_rival_load` stays on the panel for that.
        moved = panel["next_moved"] if "next_moved" in panel else pd.Series(
            np.nan, index=panel.index)
        qb_change = panel["next_qb_change"] if "next_qb_change" in panel else pd.Series(
            np.nan, index=panel.index)

        # snap_share and team_vol were built, wired in and measured here, and
        # then taken back out. Both remain on the panel; neither is a channel.
        #
        # Snap share is redundant. Within position it correlates 0.85-0.95 with
        # `usage` - QB 0.95, RB 0.91, WR 0.90, TE 0.85 - because attempts plus
        # carries plus targets already counts nearly the same thing. On fold
        # 2021 the pair moved pooled CRPS from 22.11 to 22.23, i.e. nothing.
        #
        # Team volume is *not* redundant (it correlates -0.05 with a player's
        # own usage) and is the only column here that knows a player changed
        # teams, so it is the one worth revisiting - but on a single fold it
        # earned nothing measurable, and an unvalidated feature does not belong
        # in a board someone drafts from. Re-test it on four folds at matched
        # chain settings before believing either result.

        channels = np.column_stack(
            [
                _standardise(np.sqrt(panel["opp_pg"].clip(lower=0))),
                _standardise(panel["late_form"].fillna(0.0)),
                _standardise(depth),
                _standardise(moved),
                _standardise(qb_change),
                _standardise(rb_room),
            ]
        )
        # The last grid column has no season after it to inform.
        moved = si + 1
        keep = moved < T
        ctrl[pi[keep], moved[keep], :] = channels[keep]

    return PanelArrays(
        ctrl_names=CTRL_NAMES,
        players=players,
        pos_idx=pos_idx,
        seasons=seasons,
        z=z,
        obs=obs,
        zvar=zvar,
        games_frac=games_frac,
        age_basis=age_basis,
        start=start,
        est=est,
        ctrl=ctrl,
        spline_spec=spline_spec,
    )


# --------------------------------------------------------------- Kalman filter


def kalman_filter(z, obs, v, m, ctrl, start, rho, sig2, su2):
    """Run the filter across seasons for every player at once.

    A player's deviation from his position-and-age cohort carries two pieces:

        deviation[i,t] = u[i] + w[i,t]

    ``u`` is **permanent** - the level he simply is, fixed for his career - and
    ``w`` is **transient**, an AR(1) that decays back to zero. The split is the
    difference between a model that can believe in a star and one that cannot.
    With a single AR(1) at rho = 0.68, four elite seasons still get dragged
    two-thirds of the way to the positional mean every year, because the model
    has no way to represent "this is his level" as distinct from "he is on a
    good run". Backtested, that showed up as the top projection decile coming
    in 12% under what those players actually scored.

    The transient starts in its stationary distribution, sig2 / (1 - rho^2),
    rather than getting its own free parameter - with at most six seasons per
    player, an initial variance and a permanent variance estimated separately
    would be trading off against each other on almost no information.

    All arrays are (player x season). ``su2`` is (player,) - it is the prior on
    a career-long quantity, fixed by definition - while ``rho`` and ``sig2`` are
    (player x season), because how durable a player's form is depends on whether
    he has established himself, and that changes during a career. See
    :func:`bayes.data.attach_established`.

    Returns the total log likelihood and the filtered mean and variance of the
    *sum* u + w at every season, which is the quantity the rest of the model
    cares about.

    Seasons a player did not play contribute no likelihood and no update: the
    state propagates, so his ability keeps aging and his uncertainty keeps
    widening while he is out. That is the correct treatment of an absence
    *given* the model is conditional on playing - whether the absence itself is
    informative is the availability model's question, not this one's.
    """
    n = rho.shape[0]

    def step(carry, xs):
        x, p = carry                      # x: [n,2], p: [n,2,2]
        z_t, obs_t, v_t, m_t, c_t, s_t, rho_t, sig2_t = xs

        # Stationary variance of the transient part, at this season's own
        # persistence and innovation.
        w_stat = sig2_t / jnp.maximum(1.0 - rho_t ** 2, 1e-6)

        # ---- predict. u is carried unchanged; w decays and takes the input.
        x_pred = jnp.stack([x[:, 0], rho_t * x[:, 1] + c_t], axis=-1)
        p_pred = jnp.stack(
            [
                jnp.stack([p[:, 0, 0], rho_t * p[:, 0, 1]], axis=-1),
                jnp.stack(
                    [rho_t * p[:, 1, 0], rho_t ** 2 * p[:, 1, 1] + sig2_t],
                    axis=-1,
                ),
            ],
            axis=-2,
        )

        # At a player's first season the state restarts from its prior: the
        # permanent piece unknown with variance su2, the transient stationary,
        # and the two independent.
        zeros = jnp.zeros((n, 2))
        p_start = jnp.stack(
            [
                jnp.stack([su2, jnp.zeros(n)], axis=-1),
                jnp.stack([jnp.zeros(n), w_stat], axis=-1),
            ],
            axis=-2,
        )
        s_col = s_t[:, None]
        x_pred = jnp.where(s_col, zeros, x_pred)
        p_pred = jnp.where(s_col[:, :, None], p_start, p_pred)

        # ---- update. The observation loads on u + w, so H = [1, 1] and the
        # innovation variance is the total of the state covariance plus noise.
        state_sum = x_pred[:, 0] + x_pred[:, 1]
        ph = p_pred.sum(axis=-1)                       # P @ [1,1]
        s = ph.sum(axis=-1) + v_t
        innov = jnp.where(obs_t, z_t - (m_t + state_sum), 0.0)
        ll = jnp.where(obs_t, -0.5 * (jnp.log(2 * jnp.pi * s) + innov ** 2 / s), 0.0)

        gain = jnp.where(obs_t[:, None], ph / s[:, None], 0.0)
        x_new = x_pred + gain * innov[:, None]
        p_new = p_pred - gain[:, :, None] * p_pred.sum(axis=-2)[:, None, :]

        total_mean = x_new[:, 0] + x_new[:, 1]
        total_var = p_new.sum(axis=(-1, -2))
        return (x_new, p_new), (ll.sum(), total_mean, total_var)

    # Carry entering the first column. `start` resets any player whose career
    # begins there, so this only has to be finite and sane; it uses the first
    # season's own persistence.
    w_stat0 = sig2[:, 0] / jnp.maximum(1.0 - rho[:, 0] ** 2, 1e-6)
    init = (
        jnp.zeros((n, 2)),
        jnp.stack(
            [
                jnp.stack([su2, jnp.zeros(n)], axis=-1),
                jnp.stack([jnp.zeros(n), w_stat0], axis=-1),
            ],
            axis=-2,
        ),
    )
    xs = (z.T, obs.T, v.T, m.T, ctrl.T, start.T, rho.T, sig2.T)
    _, (ll, xf, pf) = lax.scan(step, init, xs)
    return ll.sum(), xf.T, pf.T


# ------------------------------------------------------------------ the model


def _level(base, beta_age, age_basis, pos_idx):
    """Position level plus aging curve, for every player and season.

    This is ability at full health - the missed-time offset is applied only to
    observed seasons, in :func:`_missed_time_offset`, and deliberately not
    here, because the season being forecast has no games count yet.
    """
    return base[pos_idx][:, None] + jnp.einsum(
        "ptk,pk->pt", age_basis, beta_age[pos_idx]
    )


def _missed_time_offset(lam, games_frac, pos_idx):
    """How much a shortened season depresses the per-game rate observed in it.

    Within the same player, seasons where he plays less are also seasons where
    he scores less per game - the panel puts the slope at 0.048 on the sqrt
    scale per game, so eight games below his own norm costs 0.39, which for a
    12-points-per-game player is about two and a half points a game. Injuries
    do not politely wait for the whistle, roles get taken over, and players
    return at less than full strength.

    Two things follow from modelling it rather than ignoring it. Ability
    estimates stop reading an injury year as decline. And at projection time
    the sampled games count feeds the per-game rate, which puts back the
    positive covariance between games and scoring that sampling them
    independently destroys - worth about 5% on the top of the board, where
    treating E[games * rate] as E[games] * E[rate] loses the most.

    Centred at a full season, so ability means full-season ability.
    """
    return lam[pos_idx][:, None] * (games_frac - 1.0)


def production_model(arr: PanelArrays, use_inputs: bool = True):
    """NUTS target: hyperparameters only, latent abilities integrated out."""
    pos = jnp.asarray(arr.pos_idx)

    # Position level on the sqrt-points-per-game scale. The panel average is
    # about 2.1, so this prior is centred there and wide enough to be led.
    base = numpyro.sample("base", dist.Normal(2.1, 0.7).expand([N_POS]).to_event(1))
    n_basis = arr.age_basis.shape[-1]
    beta_age = numpyro.sample(
        "beta_age", dist.Normal(0.0, 0.3).expand([N_POS, n_basis]).to_event(2)
    )

    # Persistence and innovation of the *transient* part, split by whether the
    # player has established himself. Index 0 is not established, 1 is.
    #
    # The panel is unambiguous that these are two processes. Among players with
    # 14+ games in consecutive seasons - so measurement noise is matched - the
    # upper half of the skill range carries ability forward at 0.87 with a
    # residual spread of 0.51; the lower half manages 0.60 and 0.65. One
    # parameter has to average them, and the average is wrong for both.
    #
    # Beta(2,2) is deliberately vague: with a permanent effect carrying the
    # durable share, rho is free to be low where the data says it is low.
    rho = numpyro.sample(
        "rho", dist.Beta(2.0, 2.0).expand([N_POS, 2]).to_event(2)
    )
    # Year-to-year innovation in the transient part - form, role, health.
    sigma = numpyro.sample(
        "sigma", dist.HalfNormal(0.5).expand([N_POS, 2]).to_event(2)
    )
    # Spread of the permanent, career-long player effect.
    sigma_u = numpyro.sample(
        "sigma_u", dist.HalfNormal(1.0).expand([N_POS]).to_event(1)
    )
    # The panel hands the filter a measurement variance for every season from
    # within-season game-to-game spread. kappa lets the data rescale it; a
    # posterior near 1 says that derivation was about right, and anything else
    # is worth knowing.
    kappa = numpyro.sample(
        "kappa", dist.LogNormal(0.0, 0.4).expand([N_POS]).to_event(1)
    )
    # Per-game penalty for a shortened season, on the sqrt scale, per full
    # season of games missed. Prior allows either sign; the panel says it is
    # firmly positive.
    lam = numpyro.sample("lam", dist.Normal(0.0, 0.5).expand([N_POS]).to_event(1))

    m = _level(base, beta_age, jnp.asarray(arr.age_basis), pos)
    m_obs = m + _missed_time_offset(lam, jnp.asarray(arr.games_frac), pos)
    v = kappa[pos][:, None] * jnp.asarray(arr.zvar)

    if use_inputs:
        n_ctrl = arr.ctrl.shape[-1]
        gamma = numpyro.sample(
            "gamma", dist.Normal(0.0, 0.2).expand([N_POS, n_ctrl]).to_event(2)
        )
        ctrl = jnp.einsum("ptc,pc->pt", jnp.asarray(arr.ctrl), gamma[pos])
    else:
        ctrl = jnp.zeros_like(v)

    # Per-cell persistence and innovation: the player's position picks the row,
    # his established status that season picks the column.
    est = jnp.asarray(arr.est).astype(int)
    rho_pt = rho[pos[:, None], est]
    sig2_pt = sigma[pos[:, None], est] ** 2

    ll, _, _ = kalman_filter(
        jnp.asarray(arr.z),
        jnp.asarray(arr.obs),
        v,
        m_obs,
        ctrl,
        jnp.asarray(arr.start),
        rho_pt,
        sig2_pt,
        sigma_u[pos] ** 2,
    )
    numpyro.factor("kalman_ll", ll)


def fit_production(
    arr: PanelArrays,
    use_inputs: bool = True,
    num_warmup: int = 800,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    progress: bool = True,
    inference: str = "nuts",
):
    """Fit the hyperparameters, by NUTS or by MAP + Laplace.

    The latent abilities are marginalised either way - that is the Kalman
    filter's job and it is unaffected by which sampler sits on top. What
    ``inference`` chooses is only how the ~56 hyperparameters are explored. See
    `bayes/laplace.py` for what the Gaussian approximation asserts and how it
    reports its own failure.
    """
    if inference == "laplace":
        return fit_laplace(
            production_model, arr, use_inputs=use_inputs,
            num_samples=num_samples, seed=seed,
        )
    if inference != "nuts":
        raise ValueError(f"unknown inference {inference!r}")
    kernel = NUTS(production_model, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=progress,
    )
    mcmc.run(jax.random.PRNGKey(seed), arr, use_inputs=use_inputs)
    return mcmc


# ------------------------------------------------------- filtered state per draw


def filtered_states(arr: PanelArrays, posterior: dict, use_inputs: bool = True):
    """Filtered ability mean and variance at every season, for every draw.

    Returns ``(mean, var)`` each of shape (draws, players, seasons). These are
    *filtered*, not smoothed: the estimate at season t uses seasons up to t and
    no later. Anything downstream that feeds a season-t estimate into a
    season-t+1 forecast needs that guarantee.
    """
    pos = jnp.asarray(arr.pos_idx)
    z = jnp.asarray(arr.z)
    obs = jnp.asarray(arr.obs)
    zvar = jnp.asarray(arr.zvar)
    ab = jnp.asarray(arr.age_basis)
    start = jnp.asarray(arr.start)
    ctrl_raw = jnp.asarray(arr.ctrl)
    games_frac = jnp.asarray(arr.games_frac)
    est = jnp.asarray(arr.est).astype(int)

    def one(base, beta_age, rho, sigma, sigma_u, kappa, lam, gamma):
        m = _level(base, beta_age, ab, pos)
        m_obs = m + _missed_time_offset(lam, games_frac, pos)
        v = kappa[pos][:, None] * zvar
        ctrl = (
            jnp.einsum("ptc,pc->pt", ctrl_raw, gamma[pos])
            if use_inputs
            else jnp.zeros_like(v)
        )
        _, xf, pf = kalman_filter(
            z, obs, v, m_obs, ctrl, start,
            rho[pos[:, None], est], sigma[pos[:, None], est] ** 2,
            sigma_u[pos] ** 2,
        )
        # The level returned is full-health ability, without the missed-time
        # offset: the season being forecast has no games count yet, and the
        # offset is applied there from the games the availability model draws.
        return xf, pf, m

    n = posterior["base"].shape[0]
    gamma = posterior.get("gamma", jnp.zeros((n, N_POS, ctrl_raw.shape[-1])))
    return jax.vmap(one)(
        posterior["base"],
        posterior["beta_age"],
        posterior["rho"],
        posterior["sigma"],
        posterior["sigma_u"],
        posterior["kappa"],
        posterior["lam"],
        gamma,
    )
