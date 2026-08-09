"""Season-grain panel for the Bayesian projection model.

One row per (athlete, season) actually played. Every quantity here describes the
season in the row itself - nothing is shifted forward - so a consumer can slice
by season without smearing a future season's information into the past. The
next-season label is attached separately by :func:`attach_next_season`.

Seasons before 2020 are excluded: the database holds only a handful of players
per year before then (1-49 players, against ~550 from 2020 on), so those rows
are not a sample of the league, they are a sample of whoever happened to be
pulled first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from helpers.db_query import query_db

FIRST_SEASON = 2020
LAST_SEASON = 2025

POSITIONS = ("QB", "RB", "TE", "WR")

# ------------------------------------------------------------- the fantasy season
#
# Week 18 is excluded everywhere. Three reasons, and they point the same way:
#
# 1. No fantasy league plays it, so a week 18 performance is worth nothing to
#    the decision this model informs.
# 2. It is where resting shows up. Among top-decile scorers who played 15-16
#    games, week 18 is the single most-missed week by a wide margin - 38
#    absences against 26 for the next worst - and weeks 5-14 carry every
#    player's bye while weeks 17-18 carry none. A contender sitting his starters
#    is not an injury, but `games` cannot tell the difference, so the missed
#    -time penalty (`lam`) and the availability model were both fitting rest as
#    if it were unavailability.
# 3. It harmonises the panel. 2020 ran 16 games over 17 weeks; 2021 on run 17
#    over 18. Dropping week 18 makes every season 16 games, which removes a
#    real bug: with SEASON_GAMES = 17, all 104 players who played every 2020
#    game were scored as having missed a game, taking a lam * (16-17)/17 hit to
#    their per-game rate for a season they never missed.
LAST_FANTASY_WEEK = 17

# Games in a season, after the week 18 cut. A handful of rows still report one
# more than this because a mid-season trade landed two box scores in one week;
# those are clipped, so "share of the season available" means the same thing in
# every row.
SEASON_GAMES = 16

# Age is measured at Sept 1 of the row's own season - a fixed point just before
# week 1, so a player is straightforwardly a year older each row and two
# players in the same season stay comparable.
AGE_REF_MMDD = "-09-01"

# `athletes.experience_years` counts the upcoming season, and `athletes` is a
# point-in-time snapshot (taken 2026-07-28/08-03, before the 2026 season). So a
# player's debut is EXPERIENCE_REF - experience_years, and this constant is tied
# to *when the snapshot was taken*, not to LAST_SEASON. Re-pull `athletes` and it
# needs re-checking: the test is in `_debut_season`, and the check is whether the
# median error against observed debut is still zero for uncensored players.
EXPERIENCE_REF = 2027

# Prior weight, in games, for shrinking a player's own within-season variance
# toward the position-level variance-mean law. A 3-game season carries almost
# no information about its own variance; a 17-game one mostly speaks for itself.
VAR_PRIOR_GAMES = 6.0

# The delta-method variance of sqrt(mean) divides by the mean, which explodes as
# the mean approaches zero. Evaluating at max(mean, this) keeps a 0.2-point-per
# -game season from claiming infinite measurement noise while still marking it
# as far less informative than a productive one.
PPG_FLOOR = 1.0

# ---------------------------------------------------------------- draft universe
#
# A player is worth modelling only once he has shown he can post a season total
# worth a roster spot. Below this bar the question "how many points will he
# score" has no decision attached to it - he is not draftable at any point in
# any format - so fitting effort spent separating a 12-point season from a
# 30-point one is effort spent on a distinction nobody acts on.
#
# The bar is on the *season total*, and a player qualifies on his best season,
# not his average: one 50-point year is enough to make him a name worth
# projecting, and the sub-50 seasons of a player who cleared it stay in the
# panel because a decline from productive to nothing is exactly the trajectory
# the model needs to learn.
MIN_PEAK_FP = 50.0
PEAK_WINDOW = (2021, LAST_SEASON)

# How many seasons a rostered player may have missed and still be projected.
# One: he sat out last year and is on a roster now, which is a real draft
# question with a real answer. Two is not - nobody drafts a player who has not
# taken a snap since 2023 - and the players in that bucket are overwhelmingly
# free agents in name only. See `add_missed_seasons`.
MAX_MISSED_SEASONS = 1

# Seasons a player must already have played before he can count as established.
# One good season is a realisation; two is the start of a level. Set against a
# panel only six seasons deep, so raising it starves the established group.
ESTABLISHED_MIN_SEASONS = 2


def _season_totals() -> pd.DataFrame:
    """Per (athlete, season) totals, per-game rates, and within-season spread.

    Aggregation is over athlete_id and season only. Grouping on team as well
    would split a traded player into two rows and halve both his games and his
    season total, which is not what "his 2023" means.
    """
    return query_db(
        """
        SELECT
            athlete_id,
            season,
            MAX(display_name)        AS display_name,
            COUNT(*)                 AS games_raw,
            SUM(fp_ppr)              AS fp_ppr,
            AVG(fp_ppr)              AS ppg,
            -- Sample variance of per-game points within the season. This is the
            -- game-to-game noise the season average is measured through, and it
            -- is what makes a 17-game average worth more than a 4-game one.
            CASE WHEN COUNT(*) > 1
                 THEN (SUM(fp_ppr * fp_ppr) - SUM(fp_ppr) * SUM(fp_ppr) / COUNT(*))
                      / (COUNT(*) - 1)
            END                      AS fp_var,

            -- Opportunity. Volume is more repeatable season to season than the
            -- efficiency on top of it, so these are carried as their own signal
            -- rather than folded into points.
            SUM(COALESCE(passingAttempts, 0))  * 1.0 / COUNT(*) AS pass_att_pg,
            SUM(COALESCE(rushingAttempts, 0))  * 1.0 / COUNT(*) AS rush_att_pg,
            SUM(COALESCE(receivingTargets, 0)) * 1.0 / COUNT(*) AS targets_pg,

            SUM(COALESCE(passingYards, 0))     * 1.0 / COUNT(*) AS pass_yds_pg,
            SUM(COALESCE(rushingYards, 0))     * 1.0 / COUNT(*) AS rush_yds_pg,
            SUM(COALESCE(receivingYards, 0))   * 1.0 / COUNT(*) AS rec_yds_pg,
            SUM(COALESCE(passingTouchdowns, 0)
              + COALESCE(rushingTouchdowns, 0)
              + COALESCE(receivingTouchdowns, 0)) * 1.0 / COUNT(*) AS td_pg
        FROM v_player_games
        WHERE season_type = 2
          AND season BETWEEN ? AND ?
          AND week <= ?
        GROUP BY athlete_id, season
        """,
        params=(FIRST_SEASON, LAST_SEASON, LAST_FANTASY_WEEK),
    )


def _late_form() -> pd.DataFrame:
    """Per (athlete, season), how the last stretch of the season compared.

    A season compressed to one average forgets its shape, and the shape carries
    signal: a receiver who took over the slot in week 10 and a receiver who lost
    the job in week 10 can finish with identical season totals and are not the
    same bet next August. This is the sqrt-scale gap between the last 8 games
    and the season as a whole, so it is on the same scale as the model's state
    and means "he was playing this much above his own season line at the end".
    """
    games = query_db(
        """
        SELECT athlete_id, season, week, game_date, fp_ppr
        FROM v_player_games
        WHERE season_type = 2 AND season BETWEEN ? AND ? AND week <= ?
        """,
        params=(FIRST_SEASON, LAST_SEASON, LAST_FANTASY_WEEK),
    )
    games = games.sort_values(["athlete_id", "season", "week", "game_date"])
    tail = games.groupby(["athlete_id", "season"]).tail(8)
    late = (
        tail.groupby(["athlete_id", "season"])["fp_ppr"]
        .mean()
        .reset_index(name="late_ppg")
    )
    late["late_z"] = np.sqrt(late["late_ppg"].clip(lower=0.0))
    return late[["athlete_id", "season", "late_z"]]


def _debut_season(panel_first: pd.Series) -> pd.Series:
    """True first NFL season per player, from `athletes` plus the panel.

    `seasons_seen` used to be a cumcount inside the 2020-2025 window, which is
    left-censored at the start of it: every 2020 row read 0, whether the player
    was an actual rookie or a ten-year veteran. 257 of the 553 players first
    seen in 2020 were not rookies - some had debuted as far back as 1998 - so
    the `rookie` indicator in the availability model was firing on nearly half
    that cohort wrongly.

    `athletes.experience_years` fixes it, with two corrections.

    **The offset.** The field counts the *upcoming* season, so a player about to
    play his first year reads 1, not 0. Validated against the 633 players whose
    first panel season is 2021 or later - late enough that the window is not
    censoring them - `EXPERIENCE_REF - experience_years` lands exactly on the
    observed debut for 56% of them with a median error of zero, against 16% and
    a median error of -1 for the uncorrected version.

    **The floor.** The remainder is noisy in both directions, and one direction
    is impossible: ~200 players carry an implied debut *after* a season they
    demonstrably played. Experience can only ever push a debut earlier than the
    panel proves, so the two are combined with a min.
    """
    exp = query_db("SELECT athlete_id, experience_years FROM athletes")
    exp["athlete_id"] = exp["athlete_id"].astype(panel_first.index.dtype)
    exp = exp.dropna(subset=["experience_years"]).set_index("athlete_id")

    implied = EXPERIENCE_REF - exp["experience_years"]
    debut = panel_first.to_frame("panel").join(implied.rename("implied"))
    return debut.min(axis=1).astype(int)


def _positions() -> pd.DataFrame:
    """One position per player, for the whole panel.

    Taken as the modal position across his games rather than per season. The
    model groups players by position, and a player who is relisted mid-career
    should not jump between hierarchies - that would reset the very history the
    state-space model exists to accumulate.
    """
    per_game = query_db(
        """
        SELECT athlete_id, position_abbr, COUNT(*) AS n
        FROM v_player_games
        WHERE season_type = 2 AND season BETWEEN ? AND ? AND week <= ?
        GROUP BY athlete_id, position_abbr
        """,
        params=(FIRST_SEASON, LAST_SEASON, LAST_FANTASY_WEEK),
    )
    modal = (
        per_game.sort_values("n", ascending=False)
        .drop_duplicates("athlete_id")
        .loc[:, ["athlete_id", "position_abbr"]]
        .rename(columns={"position_abbr": "pos"})
    )
    return modal[modal["pos"].isin(POSITIONS)]


def _ages(df: pd.DataFrame) -> pd.Series:
    birth = query_db("SELECT athlete_id, birth_date FROM athletes")
    birth["birth_date"] = pd.to_datetime(birth["birth_date"], errors="coerce")
    birth["athlete_id"] = birth["athlete_id"].astype(df["athlete_id"].dtype)

    merged = df[["athlete_id", "season"]].merge(birth, on="athlete_id", how="left")
    ref = pd.to_datetime(merged["season"].astype(str) + AGE_REF_MMDD)
    # NaN where birth_date is unknown; the model treats age as a covariate and
    # an unknown one is imputed to the position median rather than to zero.
    return (ref - merged["birth_date"]).dt.days.to_numpy() / 365.25


def variance_law(df: pd.DataFrame) -> pd.DataFrame:
    """Fit Var(per-game points) ~ a + b * mean, per position.

    Empirically b is 3.4-5.6 across positions and a is small: game-to-game
    variance grows roughly in proportion to the scoring level, the signature of
    an overdispersed count process. That proportionality is exactly what the
    sqrt transform stabilises, and it supplies a variance estimate for players
    whose own season is too short to provide one.
    """
    rows = []
    for pos, g in df[df["games"] >= 6].groupby("pos"):
        mu, var = g["ppg"].to_numpy(), g["fp_var"].to_numpy()
        ok = np.isfinite(mu) & np.isfinite(var)
        design = np.c_[np.ones(ok.sum()), mu[ok]]
        a, b = np.linalg.lstsq(design, var[ok], rcond=None)[0]
        # A negative intercept is a fitting artifact at the low end, not a
        # claim that variance can be negative.
        rows.append({"pos": pos, "var_a": max(a, 0.0), "var_b": b})
    return pd.DataFrame(rows)


def build_panel() -> pd.DataFrame:
    """Return the season-grain panel, one row per athlete-season played."""
    df = _season_totals().merge(_positions(), on="athlete_id", how="inner")

    df["games"] = df["games_raw"].clip(upper=SEASON_GAMES)
    # Recompute the rate on the clipped denominator so games * ppg reproduces
    # the season total for the 18-game rows too.
    df["ppg"] = df["fp_ppr"] / df["games"]
    df["age"] = _ages(df)

    df = measurement_noise(df)

    df["opp_pg"] = df["pass_att_pg"] + df["rush_att_pg"] + df["targets_pg"]

    df = df.merge(_late_form(), on=["athlete_id", "season"], how="left")
    # How the closing stretch compared to the season as a whole. Zero for a
    # season short enough that "the last 8 games" is the whole thing, which is
    # correct: there is no trajectory to read.
    df["late_form"] = (df["late_z"] - df["z"]).where(df["games"] > 8, 0.0)

    # Every row built from box scores is a season the player actually played.
    # `add_missed_seasons` appends rows where he did not, and those carry no
    # observation of scoring rate - the distinction the filter needs, and one
    # "has a row" can no longer carry once absences are represented explicitly.
    df["played"] = True

    keep = [
        "athlete_id", "display_name", "pos", "season", "played", "games",
        "fp_ppr", "ppg", "fp_var", "z", "z_var", "age", "opp_pg", "late_form",
        "pass_att_pg", "rush_att_pg", "targets_pg", "pass_yds_pg",
        "rush_yds_pg", "rec_yds_pg", "td_pg", "var_a", "var_b",
    ]
    df = df[keep].sort_values(["athlete_id", "season"]).reset_index(drop=True)

    # Seasons of NFL experience, counted from the player's real debut rather
    # than from the start of this window, so `seasons_seen == 0` means an actual
    # rookie season in every year of the panel - which is what the availability
    # model's rookie indicator is asking.
    debut = _debut_season(df.groupby("athlete_id")["season"].min())
    df["seasons_seen"] = df["season"] - df["athlete_id"].map(debut)

    return df


def measurement_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``z``, its measurement variance, and the variance law behind it.

    Split out from :func:`build_panel` so a backtest fold can recompute it on
    its own training seasons. The law is a mild nuisance parameter, but fitting
    it once over every season and then using it to score a held-out one would
    let information from the future in through the back door, and the whole
    point of the backtest is that nothing does.
    """
    df = df.copy()
    df = df.drop(columns=[c for c in ("var_a", "var_b") if c in df], errors="ignore")
    df = df.merge(variance_law(df), on="pos", how="left")

    # ---------------------------------------------------------- observation
    #
    # z = sqrt(points per game) is the scale the model works on. The sqrt is not
    # cosmetic: within-season variance is close to proportional to the mean, so
    # sqrt flattens it, and the residual spread of next-season z against this
    # season's z is near-constant across the skill range (0.73/0.73/0.53 by
    # tercile) where on the raw points scale it triples (2.6/3.5/3.7). A model
    # with one noise parameter needs the scale on which one noise parameter is
    # true.
    #
    # Negative season totals (53 rows, all short seasons of a QB throwing
    # interceptions or a back losing fumbles) clip to zero: sqrt has no opinion
    # about them, and the clip is recorded in the observation variance below as
    # a maximally uninformative measurement.
    df["z"] = np.sqrt(df["ppg"].clip(lower=0.0))

    # Within-season variance, shrunk toward the position law. A one-game season
    # has no variance of its own and takes the law outright.
    law = df["var_a"] + df["var_b"] * df["ppg"].clip(lower=0.0)
    own_n = (df["games"] - 1).clip(lower=0)
    w = own_n / (own_n + VAR_PRIOR_GAMES)
    fp_var = df["fp_var"].fillna(0.0)
    var_shrunk = w * fp_var + (1 - w) * law

    # Delta method: Var(sqrt(m)) ~ Var(m) / (4m), with Var(m) = s^2 / games.
    # This is the variance the season average is *measured* through, and it is
    # known rather than estimated - a 3-game sample is loud in a way the model
    # should not have to infer.
    df["z_var"] = var_shrunk / (4.0 * df["games"] * df["ppg"].clip(lower=PPG_FLOOR))

    return df


def rostered_players() -> set:
    """Players on an NFL roster or unsigned free agents, as of the snapshot.

    `athletes.active` is 1 only for players currently under contract, so free
    agents - who in early August are still very much draftable - read 0. Both
    are included; what is excluded is retired and out-of-league.

    This is a **point-in-time snapshot with no history** (`updated_at` runs
    2026-07-28 to 08-03). It is therefore usable for deciding who to project and
    never for deciding who to train on: applied to a past fold it would select
    the population on who survived, which is the exact bias the box-score player
    enumeration exists to avoid.
    """
    rows = query_db(
        "SELECT athlete_id FROM athletes WHERE active = 1 OR status = 'Free Agent'"
    )
    return set(rows["athlete_id"])


def add_missed_seasons(
    panel: pd.DataFrame,
    roster_ids: set | None = None,
    max_missed: int = MAX_MISSED_SEASONS,
) -> pd.DataFrame:
    """Append explicit zero-game rows for seasons a player missed entirely.

    Without these, a missed season is simply an absent row, and the two things
    that follow from it both go unmodelled: the availability model never sees a
    player *return* from a lost year (every training row is a season played, so
    `games_frac` never reaches 0 and the coefficient is extrapolating when it
    matters most), and a player who missed the most recent season cannot be
    projected at all, because the projection set is "everyone with a row in the
    cutoff season".

    Two kinds of gap, and they differ in what they are entitled to know.

    **Interior** - a season between two the player appeared in, bounded by
    appearances *inside this frame*. Purely historical. Because the bound comes
    from the frame it is given, calling this after truncating to a fold's
    training seasons keeps it causally exact: a 2023 gap row only appears once
    the player has been seen again by the cutoff, which is what a drafter would
    have known.

    **Trailing** - seasons after a player's last appearance, added only for
    ``roster_ids``, and only when he has missed no more than ``max_missed``.
    This needs the roster snapshot, so it is projection-only; passing it in a
    backtest would assert survival the fold cannot know.

    The recency bound is doing real work. ``status = 'Free Agent'`` covers 515
    of 1,493 athletes - it means "not currently signed", not "between jobs" -
    so an unbounded rule adds 528 players, 418 of whom last played in 2020-2023
    and are out of the league in everything but the label. Bounding at one
    missed season leaves 110: the players who sat out last year and could
    plausibly be drafted this one, Deshaun Watson and Joe Mixon and Brandon
    Aiyuk among them. A player who has missed two straight seasons is not a
    draft consideration, and inventing two years of zero-game rows to say so
    would tell the availability model something it already knows.

    The rows carry ``played = False`` and no scoring observation. The Kalman
    filter already propagates a player's ability across a season it cannot
    observe, widening the variance as it goes, so production is unaffected by
    construction - these rows exist for the availability half and the projection
    set.
    """
    if panel.empty:
        return panel

    last_season = int(panel["season"].max())
    seen = panel.groupby("athlete_id")["season"].agg(["min", "max"])
    played_at = set(zip(panel["athlete_id"], panel["season"]))

    wanted: list[tuple] = []
    for athlete_id, (first, last) in seen.iterrows():
        recent = last_season - int(last) <= max_missed
        stop = (
            last_season
            if roster_ids and athlete_id in roster_ids and recent
            else last
        )
        wanted += [
            (athlete_id, s)
            for s in range(int(first) + 1, int(stop) + 1)
            if (athlete_id, s) not in played_at
        ]
    if not wanted:
        return panel

    gaps = pd.DataFrame(wanted, columns=["athlete_id", "season"])

    # Identity carries over from the player; everything measured is zero or
    # absent. `z` and `z_var` are placeholders - `played = False` means nothing
    # downstream reads them - but they must be finite, because 0 games would
    # otherwise divide to inf in the delta-method variance.
    ident = panel.drop_duplicates("athlete_id").set_index("athlete_id")
    for col in ("display_name", "pos", "var_a", "var_b"):
        gaps[col] = gaps["athlete_id"].map(ident[col])

    gaps["played"] = False
    for col in ("games", "fp_ppr", "ppg", "opp_pg", "late_form", "z",
                "pass_att_pg", "rush_att_pg", "targets_pg", "pass_yds_pg",
                "rush_yds_pg", "rec_yds_pg", "td_pg"):
        gaps[col] = 0.0
    gaps["fp_var"] = np.nan
    gaps["z_var"] = 1.0

    # Age comes from a birth date, so it is known for a season he sat out - the
    # whole reason the aging curve can carry a player across one.
    gaps["age"] = _ages(gaps)
    gaps["seasons_seen"] = gaps["season"] - gaps["athlete_id"].map(
        _debut_season(panel.groupby("athlete_id")["season"].min())
    )

    # Align to whatever the caller's frame carries. Anything the gap rows have
    # no value for - next-season labels, most likely, which the caller rebuilds
    # afterwards - comes through as NaN rather than raising.
    gaps = gaps.reindex(columns=panel.columns)
    gaps["played"] = False

    out = pd.concat([panel, gaps], ignore_index=True)
    return out.sort_values(["athlete_id", "season"]).reset_index(drop=True)


def attach_established(df: pd.DataFrame) -> pd.DataFrame:
    """Flag each player-season as an established player, from prior seasons only.

    The panel says elite players are a different process, not a tail of the same
    one. Restricted to pairs where both seasons ran 14+ games - so measurement
    noise is small and near-identical between groups (mean z_var 0.066 vs
    0.069) - the upper half of the skill range carries its sqrt-scale ability
    forward at a slope of 0.87 against 0.60 for the lower half, and does it with
    22% less residual spread (0.51 against 0.65). It holds in all four
    positions.

    A single `rho` and `sigma` per position cannot express that. It fits a
    compromise, and the compromise is wrong in both directions for exactly the
    players a drafter cares about: too much regression toward the cohort, and
    intervals too wide. This flag is what lets the two halves have their own.

    **Everything here looks backwards.** The mean is over seasons strictly
    before the row's own, the count is of seasons already played, and the
    threshold is a median over the frame it is handed - which, called on a
    fold's training seasons, contains nothing the fold has not reached. A flag
    built from a player's own future would hand the model the answer.

    Note this is deliberately *not* the 50-point peak filter. That one removed
    players from the fit and measured worse: it moved the position baselines and
    took away the low end that identifies the dropout cliff. Nobody is removed
    here. They are just no longer forced to share a variance parameter.
    """
    df = df.sort_values(["athlete_id", "season"]).reset_index(drop=True)

    played = df["played"] if "played" in df else pd.Series(True, index=df.index)
    z_played = df["z"].where(played)

    grp = df.groupby("athlete_id")["season"]  # groupby key only; ops below use z
    by_player = z_played.groupby(df["athlete_id"])

    # Expanding mean and count over seasons *before* this one. `shift(1)` is the
    # whole causal argument: without it a player's own season sets the flag that
    # governs how that season is filtered.
    prior_mean = by_player.apply(lambda s: s.expanding().mean().shift(1))
    prior_n = by_player.apply(lambda s: s.notna().cumsum().shift(1))
    prior_mean = prior_mean.reset_index(level=0, drop=True).sort_index()
    prior_n = prior_n.reset_index(level=0, drop=True).sort_index()

    # Threshold per position, from the frame in hand.
    bar = df.loc[played].groupby("pos")["z"].median()

    # Experience is the player's real career length, not his tenure inside this
    # window, so a veteran already 6 years into the league in 2021 counts as
    # one. Counting panel seasons instead left 2020 and 2021 with *no*
    # established players at all and cost a year of the transitions this split
    # has to be identified from - which, on six seasons, is not affordable.
    #
    # The panel still has to supply at least one prior season, because the
    # threshold is on observed scoring and there is nothing to average
    # otherwise. That is what keeps 2020 empty, correctly: nobody in it has a
    # prior season on record.
    df["established"] = (
        (prior_n.fillna(0) >= 1)
        & (df["seasons_seen"] >= ESTABLISHED_MIN_SEASONS)
        & (prior_mean >= df["pos"].map(bar))
    ).fillna(False).to_numpy()

    del grp
    return df


def qualifying_players(
    panel: pd.DataFrame,
    min_peak: float = MIN_PEAK_FP,
    window: tuple[int, int] = PEAK_WINDOW,
    through: int | None = None,
) -> set:
    """Athlete ids whose best season in the window cleared ``min_peak``.

    ``through`` caps the seasons allowed to establish qualification, and it is
    the difference between a defensible backtest and a flattering one. Asking
    "did he ever post a 50-point season in 2021-2025" is a question only
    answerable in 2026: applied to a fold that predicts 2022 it would keep the
    players who were *about to* break out and discard the ones who were about
    to wash out, handing the model a universe selected on the very outcomes it
    is being scored against. Passing ``through=cutoff`` restricts the evidence
    to seasons that had already happened, which is the same rule a drafter
    could have applied at the time.

    Qualification is a property of the player, not the season, so a qualifying
    player keeps every row he has - including the seasons where he scored
    nothing, which are real outcomes for a name that was on draft boards.
    """
    lo, hi = window
    if through is not None:
        hi = min(hi, through)
    seen = panel[panel["season"].between(lo, hi)]
    peak = seen.groupby("athlete_id")["fp_ppr"].max()
    return set(peak.index[peak >= min_peak])


def apply_peak_filter(
    panel: pd.DataFrame,
    min_peak: float = MIN_PEAK_FP,
    window: tuple[int, int] = PEAK_WINDOW,
    through: int | None = None,
) -> pd.DataFrame:
    """Drop every player who never cleared ``min_peak`` in a single season."""
    keep = qualifying_players(panel, min_peak=min_peak, window=window, through=through)
    return panel[panel["athlete_id"].isin(keep)].reset_index(drop=True)


def attach_next_season(panel: pd.DataFrame) -> pd.DataFrame:
    """Add next-season games and points to every row that can have them.

    A player with no row in season t+1 did not play in it, which is a real
    fantasy outcome worth zero points - not a missing value. The exception is
    the last season in the panel, where absence means "not played yet"; those
    labels stay NaN and those rows are the projection set.
    """
    nxt = panel[["athlete_id", "season", "games", "fp_ppr"]].rename(
        columns={"games": "next_games", "fp_ppr": "next_fp_ppr"}
    )
    nxt["season"] -= 1

    out = panel.merge(nxt, on=["athlete_id", "season"], how="left")

    resolved = out["season"] < panel["season"].max()
    out.loc[resolved, ["next_games", "next_fp_ppr"]] = (
        out.loc[resolved, ["next_games", "next_fp_ppr"]].fillna(0.0)
    )
    return out


if __name__ == "__main__":
    p = attach_next_season(build_panel())
    pd.set_option("display.width", 220)
    print(f"{len(p):,} player-seasons, {p.athlete_id.nunique():,} players")
    print(p.groupby("season").agg(
        rows=("athlete_id", "size"),
        mean_pts=("fp_ppr", "mean"),
        mean_games=("games", "mean"),
        mean_z=("z", "mean"),
        med_zvar=("z_var", "median"),
    ).round(3))
    print("\nz_var by games played:")
    print(p.groupby(pd.cut(p["games"], [0, 3, 6, 10, 14, 17]), observed=True)["z_var"]
          .describe()[["count", "25%", "50%", "75%"]].round(3))
