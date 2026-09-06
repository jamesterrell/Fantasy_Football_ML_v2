"""Season-grain panel for the Bayesian projection model.

One row per (athlete, season) actually played. Every quantity here describes the
season in the row itself - nothing is shifted forward - so a consumer can slice
by season without smearing a future season's information into the past. The
next-season label is attached separately by :func:`attach_next_season`.

Seasons before 2016 are excluded: the database holds 1-7 players per year
before then, against 508-608 from 2016 on, so those rows are not a sample of
the league, they are a sample of whoever happened to be pulled first.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from helpers.db_query import query_db

FIRST_SEASON = 2016
LAST_SEASON = 2025

POSITIONS = ("QB", "RB", "TE", "WR")

# ------------------------------------------------------------- the fantasy season
#
# The **final scheduled week of every season is dropped**, whichever week that
# is: week 17 in 2016-2020, week 18 from 2021 on. Two reasons, and they point
# the same way:
#
# 1. No fantasy league plays it, so a performance there is worth nothing to the
#    decision this model informs.
# 2. It is where resting shows up. Among top-decile scorers who played 15-16
#    games, the final week is the single most-missed week by a wide margin - 38
#    absences against 26 for the next worst - and the middle weeks carry every
#    player's bye while the last two carry none. A contender sitting his
#    starters is not an injury, but `games` cannot tell the difference, so the
#    missed-time penalty (`lam`) and the availability model were both fitting
#    rest as if it were unavailability.
#
# The cut is derived from each season's schedule, never hardcoded. The old
# `LAST_FANTASY_WEEK = 17` implemented the rule for 2021-2025 only: before 2021
# week 18 does not exist, so nothing was dropped and the rest week survived for
# half the panel, contaminating `lam` and the availability model in an
# era-correlated way.
#
# What the cut does *not* do is harmonise the panel to 16 games. Every team
# plays in the final week of every season (16 games in it, all 32 teams), so
# removing it costs each team one game: 2016-2020 ran 16 games over 17 weeks
# and leave 15; 2021-2025 ran 17 over 18 and leave 16. Season length is
# therefore a property of the season - see `season_length` - carried on the
# panel as `season_games`, and every place that asks "what share of the season
# was he available for" divides by the row's own season rather than by a
# constant. A single 16 would score all ~500 players who played every game in
# 2016-2020 as having missed one: the exact bug the SEASON_GAMES = 17 era hit
# in 2020, re-created in the other direction.

# Length of the season being *projected* (2026: 17 fantasy weeks, one game per
# team dropped with the final week, so 16). It is the right denominator for a
# forward-looking draw and the wrong one for a historical row, which uses its
# own `season_games`.
SEASON_GAMES = 16


@lru_cache(maxsize=1)
def _schedule() -> pd.DataFrame:
    """Per season: the last scheduled week, and games per team once it is cut.

    Read from `games` (the schedule) rather than from `v_player_games`, so a
    season with no box score in its final week would still be cut correctly.
    Season length is the *median* team's game count in the retained weeks, not
    the max: 2017 carries one duplicated team-game in the schedule table and the
    median is unmoved by it, where the max would silently lengthen the season.
    """
    g = query_db(
        "SELECT season, week, home_team_id, away_team_id FROM games "
        "WHERE season_type = 2 AND season BETWEEN ? AND ?",
        params=(FIRST_SEASON, LAST_SEASON),
    )
    cut = g.groupby("season")["week"].max() - 1
    long = pd.concat(
        [
            g[["season", "week", "home_team_id"]].rename(columns={"home_team_id": "t"}),
            g[["season", "week", "away_team_id"]].rename(columns={"away_team_id": "t"}),
        ]
    )
    kept = long[long["week"] <= long["season"].map(cut)]
    length = kept.groupby(["season", "t"]).size().groupby("season").median()
    return pd.DataFrame({"last_week": cut, "season_games": length.astype(int)})


def last_fantasy_week(season: int) -> int:
    """Last week of ``season`` that counts, i.e. one before the last scheduled."""
    return int(_schedule().loc[season, "last_week"])


def season_length(season: int) -> int:
    """Games a team plays in ``season`` once its final week is dropped.

    A season the schedule does not cover is one that has not been played, and
    the only such season anything asks about is the one being projected; it
    takes SEASON_GAMES. Raising instead would mean the projection could not be
    made at all, which is the wrong answer to "how long is next year".
    """
    sched = _schedule()
    if season not in sched.index:
        return SEASON_GAMES
    return int(sched.loc[season, "season_games"])


def _week_cut_sql(col: str = "season") -> str:
    """SQL predicate keeping only weeks that count, per season.

    Inlined as a CASE rather than passed as a parameter because the bound
    differs by season; a single `week <= ?` is what made the old cut a no-op
    before 2021.
    """
    arms = " ".join(
        f"WHEN {s} THEN {w}" for s, w in _schedule()["last_week"].items()
    )
    return f"week <= CASE {col} {arms} ELSE 0 END"

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
# no information about its own variance; a full one mostly speaks for itself.
# Unchanged by the widened panel: this weight is about games *within* a season,
# and a season is still 15-16 games. What the extra seasons do change is the
# thing being shrunk toward - `variance_law` is now fit on ~4,900 player-seasons
# instead of ~2,600, so the law itself is better estimated, which argues for
# leaving the weight where it is rather than lowering it.
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
#
# The bar is on points, and points are era-dependent in two ways that pull in
# opposite directions: pre-2021 seasons are one game shorter (~6% fewer points
# at the same rate) and older seasons under-report marginal stat lines, so a
# fixed 50 admits slightly fewer 2016-2020 players than 2021-2025 ones. It is
# left at 50 anyway, because it is a *decision* threshold - below it nobody is
# drafted in any format - not an estimate calibrated to a distribution. It is
# also inert in the backtest, which runs no peak filter; it only shapes the
# projection universe.
MIN_PEAK_FP = 50.0

# The window over which a player may have earned his 50-point season. Was
# hardcoded (2021, LAST_SEASON), which under a 2020-start panel meant "all but
# the first season" and under a 2016-start panel would mean "the last five",
# silently discarding half the evidence the panel now holds. It is the panel, so
# it is written as the panel. `through=` still restricts it to seasons that had
# already happened at a fold's cutoff, which is what keeps it causal.
PEAK_WINDOW = (FIRST_SEASON, LAST_SEASON)

# How many seasons a rostered player may have missed and still be projected.
# One: he sat out last year and is on a roster now, which is a real draft
# question with a real answer. Two is not - nobody drafts a player who has not
# taken a snap since 2023 - and the players in that bucket are overwhelmingly
# free agents in name only. See `add_missed_seasons`.
MAX_MISSED_SEASONS = 1

# Seasons a player must already have played before he can count as established.
# One good season is a realisation; two is the start of a level.
#
# The old comment justified 2 by the panel being six seasons deep, so that
# raising it would starve the established group. That reason has expired - ten
# seasons make 3 or 4 affordable - but the value stays at 2, because the reason
# to expire is not a reason to change: 2 is the substantive claim (a level needs
# a repeat), and the constant is what selects which rows get their own `rho` and
# `sigma`. Raising it is now a *testable* model-structure change with a real
# alternative behind it, and it belongs in a run that measures it against
# top-of-board bias, not in a data-layer widening.
ESTABLISHED_MIN_SEASONS = 2


def _season_totals() -> pd.DataFrame:
    """Per (athlete, season) totals, per-game rates, and within-season spread.

    Aggregation is over athlete_id and season only. Grouping on team as well
    would split a traded player into two rows and halve both his games and his
    season total, which is not what "his 2023" means.
    """
    return query_db(
        f"""
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
          AND {_week_cut_sql()}
        GROUP BY athlete_id, season
        """,
        params=(FIRST_SEASON, LAST_SEASON),
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
        f"""
        SELECT athlete_id, season, week, game_date, fp_ppr
        FROM v_player_games
        WHERE season_type = 2 AND season BETWEEN ? AND ? AND {_week_cut_sql()}
        """,
        params=(FIRST_SEASON, LAST_SEASON),
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
    play his first year reads 1, not 0.

    **The floor.** The remainder is noisy in both directions, and one direction
    is impossible: a player carrying an implied debut *after* a season he
    demonstrably played. Experience can only ever push a debut earlier than the
    panel proves, so the two are combined with a min.

    **Re-validated on the 2016-start panel, and the old test statement no longer
    holds as written.** Over all 1,122 players first seen in 2017 or later - the
    cohort the window is not censoring - the median error against observed debut
    is **+1**, not zero, and 57% carry an impossible (post-dated) debut. Split by
    whether the snapshot still counts the player, the reason is immediate:

    * `active = 1` (578 players): median error **0**, exact match 74%. This is
      the test the constant was set by, and EXPERIENCE_REF = 2027 passes it -
      better than it did on the old panel (58% exact, 593 players).
    * `active = 0` (544 players): median error **+4**, exact match 3%, 96%
      impossible. `experience_years` is frozen at the last season the player was
      active, so for anyone out of the league it counts to his final year rather
      than to 2026 and carries no information about when he started.

    So the constant is right and the *criterion* was under-specified: it is
    "median error zero among players the snapshot still counts". Widening to
    2016 did not break the debut inference, it enlarged the retired share of the
    panel from a fifth to a half, which is where the field was always useless.

    The min floor absorbs it - a frozen count is always too late, so it always
    loses to the panel - at the price of pinning a retired veteran's debut to his
    first *panel* season. 1,083 players end up pinned that way; their median age
    in that season is 23.3, so most are genuine rookies, but 111 are older than
    25.5 and 25 older than 27 and are near-certainly mislabelled as rookies. That
    is ~2% of panel rows carrying a wrong `seasons_seen == 0`, which reaches the
    availability model's rookie indicator and the established-player gate. An age
    gate on the rookie flag is the obvious cheap fix and has not been made here.
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
        f"""
        SELECT athlete_id, position_abbr, COUNT(*) AS n
        FROM v_player_games
        WHERE season_type = 2 AND season BETWEEN ? AND ? AND {_week_cut_sql()}
        GROUP BY athlete_id, position_abbr
        """,
        params=(FIRST_SEASON, LAST_SEASON),
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

    # Season length is the row's own season's, not a constant - see the
    # fantasy-season note at the top. A handful of rows report one game more
    # than the schedule allows because a mid-season trade landed two box scores
    # in one week; those are clipped, so "share of the season available" means
    # the same thing in every row and in every era.
    df["season_games"] = df["season"].map(_schedule()["season_games"]).astype(float)
    df["games"] = np.minimum(df["games_raw"], df["season_games"])
    # Recompute the rate on the clipped denominator so games * ppg reproduces
    # the season total for the clipped rows too.
    df["ppg"] = df["fp_ppr"] / df["games"]
    df["games_frac"] = df["games"] / df["season_games"]
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
        "season_games", "games_frac",
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
    gaps["season_games"] = gaps["season"].map(season_length).astype(float)
    for col in ("games", "games_frac", "fp_ppr", "ppg", "opp_pg", "late_form", "z",
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
    # window, so a veteran already 6 years into the league counts as one.
    # Counting panel seasons instead left the first two seasons of the panel with
    # *no* established players at all and cost a year of the transitions this
    # split has to be identified from.
    #
    # The panel still has to supply at least one prior season, because the
    # threshold is on observed scoring and there is nothing to average
    # otherwise. That is what keeps the panel's first season empty, correctly:
    # nobody in it has a prior season on record.
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
    "did he ever post a 50-point season in 2016-2025" is a question only
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


# Roster and depth-chart state at the start of a season, built by the database
# project's `ffdb preseason` loader. Keyed on the season it describes, so the
# row attached to a panel row is the one for `season + 1` - the state of the
# world just before the outcome that row is trying to predict.
PRESEASON_COLUMNS = ("next_on_roster", "next_status", "next_depth_rank")


def _preseason() -> pd.DataFrame:
    """Preseason roster state per athlete-season, or empty if never loaded.

    The table lives in the same database but is loaded separately and from a
    different source (nflverse, because ESPN keeps no roster history). Treating
    its absence as fatal would break every existing workflow the moment this
    file is pulled without the loader having been run, so a missing table
    degrades to "no context" rather than an exception.
    """
    try:
        pre = query_db(
            "SELECT season, athlete_id, status, on_roster, depth_rank, team "
            "FROM preseason_roster"
        )
    except Exception:  # table not created yet
        return pd.DataFrame(
            columns=["season", "athlete_id", "status", "on_roster", "depth_rank",
                     "team"]
        )
    pre["athlete_id"] = pre["athlete_id"].astype(str)
    return pre


def attach_preseason_context(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach next season's roster status and depth-chart rank to every row.

    These describe the season being *predicted*, not the season being observed,
    which is what makes them usable: whether a player is on a roster in August,
    and where he sits on a depth chart, is settled before the season starts and
    is exactly what a drafter knows and the box scores do not say.

    Measured on the dev folds, the model's residual by status runs from -55.6
    points for retired players to +13.7 for active ones, and the slope in depth
    rank survives conditioning on prior-season usage - so neither is a
    restatement of a feature the panel already has.

    Two cautions, both deliberate:

    * `next_on_roster` is the defensible flag. It turns on the roster
      transactions - released, retired, unsigned - that are settled at final
      cuts, a little before drafts. `next_status` is exposed raw, but its `INA`
      level is a game-day inactive declaration and is genuinely later than draft
      day; using that level as a predictor would be reading the future.
    * For 2016-2024 the snapshot is regular-season week 1, the earliest the
      source publishes, which is after final cuts. The 2025 format snapshots
      from early August and is clean. Backtest gains on the older folds should
      therefore be read as an upper bound.

    Rows with no match keep NaN/1: absent from the roster table is "unknown",
    and defaulting an unknown player to off-roster would invent the very signal
    this is supposed to measure.
    """
    pre = _preseason()
    if pre.empty:
        out = panel.copy()
        out["next_on_roster"] = np.nan
        out["next_status"] = pd.NA
        out["next_depth_rank"] = np.nan
        out["next_team"] = pd.NA
        return out

    pre = pre.rename(
        columns={
            "on_roster": "next_on_roster",
            "status": "next_status",
            "depth_rank": "next_depth_rank",
            "team": "next_team",
        }
    )
    # The table is keyed by the season it describes; the panel row it belongs to
    # is the one a season earlier, whose label is that season's outcome.
    pre["season"] = pre["season"] - 1

    out = panel.copy()
    out["athlete_id"] = out["athlete_id"].astype(str)
    return out.merge(pre, on=["athlete_id", "season"], how="left")


# nflverse and ESPN disagree on exactly two team codes. Left unmapped, the Rams
# and Commanders silently drop out of every team-context join - a null feature
# for two teams' worth of players, raising nothing.
NFLVERSE_TO_ESPN_TEAM = {"LA": "LAR", "WAS": "WSH"}

# Columns the preseason/role context contributes to the panel.
CONTEXT_COLUMNS = ("snap_pct", "next_team_vol", "next_moved",
                   "next_qb_change", "next_rival_load",
                   "next_rb_rival_quality")


def _snap_share() -> pd.DataFrame:
    """Share of his offense's snaps a player was on the field for, per season.

    Box-score volume says what he did; this says how much he was out there to
    do it. Averaged over the games he appeared in, so it is a role measure and
    not a second helping of missed time - the availability model owns that.
    """
    try:
        snaps = query_db(
            "SELECT season, athlete_id, offense_pct AS snap_pct FROM player_snaps"
        )
    except Exception:
        return pd.DataFrame(columns=["season", "athlete_id", "snap_pct"])
    snaps["athlete_id"] = snaps["athlete_id"].astype(str)
    return snaps


def _team_volume() -> pd.DataFrame:
    """Per team-season offensive volume, split into passing and rushing.

    Volume is the part of an offence that carries year to year - who calls the
    plays changes, how many plays there are changes far less - so last season's
    volume is a usable stand-in for the situation a player is walking into.
    """
    vol = query_db(
        f"""
        SELECT season, team_abbr,
               SUM(COALESCE(passingAttempts, 0))
                 + SUM(COALESCE(receivingTargets, 0)) AS team_pass_vol,
               SUM(COALESCE(rushingAttempts, 0))      AS team_rush_vol
        FROM v_player_games
        WHERE season_type = 2 AND {_week_cut_sql()}
        GROUP BY season, team_abbr
        """
    )
    return vol.dropna(subset=["team_abbr"])


def _preseason_qb1() -> pd.DataFrame:
    """Each team's preseason QB1 per season, from the depth chart."""
    try:
        qb = query_db(
            "SELECT season, team, athlete_id AS qb1 FROM preseason_roster "
            "WHERE position = 'QB' AND depth_rank = 1 AND team IS NOT NULL"
        )
    except Exception:
        return pd.DataFrame(columns=["season", "team", "qb1"])
    qb = qb.drop_duplicates(["season", "team"])
    qb["team"] = qb["team"].map(lambda t: NFLVERSE_TO_ESPN_TEAM.get(t, t))
    return qb


def _starting_qb() -> pd.DataFrame:
    """Who actually threw most for each team in each season."""
    qb = query_db(
        f"""
        SELECT season, team_abbr, athlete_id, SUM(COALESCE(passingAttempts, 0)) AS att
        FROM v_player_games
        WHERE season_type = 2 AND position_abbr = 'QB' {"AND " + _week_cut_sql()}
        GROUP BY season, team_abbr, athlete_id
        """
    )
    qb = qb[qb["att"] >= 100]
    return (qb.sort_values("att", ascending=False)
              .drop_duplicates(["season", "team_abbr"])
              .rename(columns={"athlete_id": "qb_prior"}))


def _prior_workload() -> pd.DataFrame:
    """Per-game targets and carries for every skill player-season."""
    w = query_db(
        f"""
        SELECT athlete_id, season, COUNT(*) AS g,
               SUM(COALESCE(receivingTargets, 0)) AS tgt,
               SUM(COALESCE(rushingAttempts, 0))  AS car,
               SUM(fp_ppr)                        AS fp
        FROM v_player_games
        WHERE season_type = 2 AND position_abbr IN ('WR','TE','RB')
              AND {_week_cut_sql()}
        GROUP BY athlete_id, season
        """
    )
    w["athlete_id"] = w["athlete_id"].astype(str)
    w["tgt_pg"] = w["tgt"] / w["g"].clip(lower=1)
    w["car_pg"] = w["car"] / w["g"].clip(lower=1)
    w["ppg"] = w["fp"] / w["g"].clip(lower=1)
    return w[["athlete_id", "season", "tgt_pg", "car_pg", "ppg"]]


def attach_competition(panel: pd.DataFrame) -> pd.DataFrame:
    """How much of the ball is already spoken for on the team he is joining.

    There is one football. A team's touches are close to fixed, so a player's
    share is a claim against his own teammates, and signing a receiver takes
    targets from the receivers already there. Nothing in a player's own history
    can see that coming - which is what makes this different from every other
    feature here.

    Measured as the summed prior-season workload of everyone *else* on his
    next-season roster: targets per game for pass-catchers, carries per game for
    backs, since those are the pools each actually competes in. Controlling for
    his own prior rate, across 2016-2024:

        receivers  -0.046 ppg per rival target/game   (t=-2.45)
        ends       -0.046                             (t=-2.63)
        backs      -0.066 ppg per rival carry/game    (t=-2.67)

    A receiver joining a room with twenty more targets a game already claimed
    loses about nine tenths of a point per game.

    Two limits worth stating. Rookies carry no prior workload, so a team that
    drafted a receiver in April looks *less* crowded than it is - the same
    blindness that makes second-year players look like reaches. And the roster
    is the August one, so a trade in September is invisible.
    """
    out = panel.copy()
    out["athlete_id"] = out["athlete_id"].astype(str)
    try:
        roster = query_db(
            "SELECT season, team, athlete_id, position FROM preseason_roster "
            "WHERE position IN ('WR','TE','RB') AND on_roster = 1 AND team IS NOT NULL"
        )
    except Exception:
        out["next_rival_load"] = np.nan
        return out
    if roster.empty:
        out["next_rival_load"] = np.nan
        return out

    roster["athlete_id"] = roster["athlete_id"].astype(str)
    roster["team"] = roster["team"].map(lambda t: NFLVERSE_TO_ESPN_TEAM.get(t, t))
    # The roster is for season S; the workload that values it is season S-1, and
    # the panel row it attaches to is the one for S-1 as well.
    roster["season"] = roster["season"] - 1

    work = _prior_workload()
    r = roster.merge(work, on=["athlete_id", "season"], how="left").fillna(
        {"tgt_pg": 0.0, "car_pg": 0.0, "ppg": 0.0})
    # Backfields are graded on *who*, not on how many carries. Measured across
    # 2016-2023 transitions, with both terms in one regression, quality carries
    # the whole effect for backs (-0.099 ppg per rival ppg, t=-2.62) while the
    # carry count adds nothing (+0.005, t=+0.14). For receivers it reverses -
    # there the target count is what matters - which is why only the backfield
    # version is built: the receiver version never moves anyone more than about
    # sixteen points, and the backfield one moves ~19 backs a season by more
    # than ten.
    rb_only = r["position"].to_numpy() == "RB"
    r["_rb_ppg"] = np.where(rb_only, r["ppg"].to_numpy(), 0.0)
    totals = r.groupby(["season", "team"])[["tgt_pg", "car_pg", "_rb_ppg"]].transform("sum")
    # Everyone else's load: the room's total minus his own contribution to it.
    r["rival_tgt"] = totals["tgt_pg"] - r["tgt_pg"]
    r["rival_car"] = totals["car_pg"] - r["car_pg"]
    r["rival_rb_quality"] = np.where(
        rb_only, totals["_rb_ppg"].to_numpy() - r["_rb_ppg"].to_numpy(), np.nan)

    out = out.merge(
        r[["season", "team", "athlete_id", "rival_tgt", "rival_car",
           "rival_rb_quality"]].rename(columns={"team": "_dest"}),
        on=["athlete_id", "season"], how="left",
    )
    dest = out["next_team"].map(lambda t: NFLVERSE_TO_ESPN_TEAM.get(t, t))         if "next_team" in out.columns else pd.Series(pd.NA, index=out.index)
    # Only trust the join where the roster row is the team he is actually on.
    ok = out["_dest"].notna() & (out["_dest"] == dest)
    out["next_rival_load"] = np.where(
        ok, np.where(out["pos"].to_numpy() == "RB",
                     out["rival_car"].to_numpy(), out["rival_tgt"].to_numpy()),
        np.nan)
    # Zero, not NaN, for everyone who is not a back: this channel says "how good
    # is the rest of your backfield", a question that does not arise for a
    # receiver, and zero is the neutral value once standardised.
    out["next_rb_rival_quality"] = np.where(
        ok & (out["pos"].to_numpy() == "RB"),
        out["rival_rb_quality"].to_numpy(), 0.0)
    return out.drop(columns=["_dest", "rival_tgt", "rival_car", "rival_rb_quality"])


def attach_situation(panel: pd.DataFrame) -> pd.DataFrame:
    """Two things August knows and a box score cannot: is he moving, and is the
    man throwing to him changing.

    Both are measured against the season being *predicted* and both come from
    the preseason snapshot, so neither reads the outcome.

    Measured across 2016-2024 transitions, controlling for the player's own
    prior rate and with both terms in the same regression so they do not stand
    in for each other:

        receivers   moved -1.23 ppg (t=-3.90)   QB change -0.17 (t=-0.61)
        backs       moved -1.04 ppg (t=-2.23)   QB change -0.80 (t=-1.97)
        ends        moved -0.56 ppg (t=-1.89)   QB change -0.11 (t=-0.40)

    So they are not the same variable wearing two hats: 96.6% of movers also
    change quarterback, but only a third of stayers do, and it is that third
    that identifies the second term. For receivers the cost is in the move
    itself - scheme, role, target competition - and who throws matters little on
    top. For backs both cost about the same, which is the defensive-attention
    story: a quarterback who cannot threaten downfield brings the safeties up.

    What is deliberately *not* here is any continuous measure of the offence he
    is joining. Those work only contemporaneously. A team's passing yards per
    game correlates just +0.44 with its own next season (20% of variance), while
    a player's own rate correlates +0.79 with his (62%) - so last year's team
    statistics are a far worse guide to next year's situation than the player's
    own history already is, and adding them contributes noise. Forecasting the
    situation needs forward-looking information this project does not have.
    """
    out = panel.copy()
    if "next_team" not in out.columns:
        out["next_moved"] = np.nan
        out["next_qb_change"] = np.nan
        return out

    dest = out["next_team"].map(lambda t: NFLVERSE_TO_ESPN_TEAM.get(t, t))
    # Team he played for this season, by appearances.
    here = out["team_abbr"] if "team_abbr" in out.columns else None
    if here is None:
        played = query_db(
            f"""
            SELECT athlete_id, season, team_abbr, COUNT(*) AS n
            FROM v_player_games
            WHERE season_type = 2 {"AND " + _week_cut_sql()}
            GROUP BY athlete_id, season, team_abbr
            """
        )
        played["athlete_id"] = played["athlete_id"].astype(str)
        played = (played.sort_values("n", ascending=False)
                        .drop_duplicates(["athlete_id", "season"]))
        here = (out[["athlete_id", "season"]]
                .merge(played, on=["athlete_id", "season"], how="left")["team_abbr"])

    out["next_moved"] = np.where(
        dest.isna() | pd.Series(here).isna(), np.nan,
        (dest.to_numpy() != pd.Series(here).to_numpy()).astype(float))

    qb1 = _preseason_qb1()
    prior = _starting_qb()
    if qb1.empty or prior.empty:
        out["next_qb_change"] = np.nan
        return out

    nxt_qb = (pd.DataFrame({"season": out["season"] + 1, "team": dest})
              .merge(qb1, on=["season", "team"], how="left")["qb1"])
    cur_qb = (pd.DataFrame({"season": out["season"], "team_abbr": pd.Series(here)})
              .merge(prior[["season", "team_abbr", "qb_prior"]],
                     on=["season", "team_abbr"], how="left")["qb_prior"])
    both = nxt_qb.notna().to_numpy() & cur_qb.notna().to_numpy()
    out["next_qb_change"] = np.where(
        both, (nxt_qb.astype(str).to_numpy() != cur_qb.astype(str).to_numpy()).astype(float),
        np.nan)
    return out


def attach_context(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach snap share, and the offence the player is joining next season.

    Two different times are involved and mixing them up would leak. `snap_pct`
    describes the row's *own* season - how he was used in the year we observed.
    `next_team_vol` describes the team he is on for the season being *predicted*,
    measured in the season we observed: he is walking into that offence, and
    what it did last year is knowable in August whereas what it will do is not.

    Volume is passing for quarterbacks and receivers, rushing for backs, because
    those are the pools each is actually drawing from. The production model's
    coefficients are per position anyway, so one column carrying the relevant
    pool costs a parameter where two columns would cost two.
    """
    out = panel.copy()
    out["athlete_id"] = out["athlete_id"].astype(str)

    snaps = _snap_share()
    out = (out.merge(snaps, on=["athlete_id", "season"], how="left")
           if not snaps.empty else out.assign(snap_pct=np.nan))

    vol = _team_volume()
    if vol.empty or "next_team" not in out.columns:
        out["next_team_vol"] = np.nan
        return out

    # The team he plays for next season, in ESPN's vocabulary; his volume figure
    # is that team's, in the season this row observes.
    team = out["next_team"].map(lambda t: NFLVERSE_TO_ESPN_TEAM.get(t, t))
    key = pd.DataFrame({"season": out["season"], "team_abbr": team})
    joined = key.merge(vol, on=["season", "team_abbr"], how="left")
    is_rb = out["pos"].to_numpy() == "RB"
    out["next_team_vol"] = np.where(
        is_rb, joined["team_rush_vol"].to_numpy(), joined["team_pass_vol"].to_numpy()
    )
    return out


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
    # Attached here rather than by a separate call so every entry point - the
    # backtest, a fold, and the live projection - picks it up from the one
    # place next-season columns are already assembled.
    return attach_competition(
        attach_situation(attach_context(attach_preseason_context(out)))
    )


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
    print(p.groupby(pd.cut(p["games"], [0, 3, 6, 10, 14, 16]), observed=True)["z_var"]
          .describe()[["count", "25%", "50%", "75%"]].round(3))
