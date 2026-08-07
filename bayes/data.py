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

# Games in an NFL regular season. 2020 was 16; a handful of rows report 18
# because of a mid-season trade landing two box scores in one week. Both are
# clipped to 17 so "share of the season available" means the same thing in
# every row.
SEASON_GAMES = 17

# Age is measured at Sept 1 of the row's own season - a fixed point just before
# week 1, so a player is straightforwardly a year older each row and two
# players in the same season stay comparable.
AGE_REF_MMDD = "-09-01"

# Prior weight, in games, for shrinking a player's own within-season variance
# toward the position-level variance-mean law. A 3-game season carries almost
# no information about its own variance; a 17-game one mostly speaks for itself.
VAR_PRIOR_GAMES = 6.0

# The delta-method variance of sqrt(mean) divides by the mean, which explodes as
# the mean approaches zero. Evaluating at max(mean, this) keeps a 0.2-point-per
# -game season from claiming infinite measurement noise while still marking it
# as far less informative than a productive one.
PPG_FLOOR = 1.0


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
        """
        SELECT athlete_id, season, week, game_date, fp_ppr
        FROM v_player_games
        WHERE season_type = 2 AND season BETWEEN ? AND ?
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
        WHERE season_type = 2 AND season BETWEEN ? AND ?
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

    keep = [
        "athlete_id", "display_name", "pos", "season", "games", "fp_ppr", "ppg",
        "fp_var", "z", "z_var", "age", "opp_pg", "late_form", "pass_att_pg",
        "rush_att_pg", "targets_pg", "pass_yds_pg", "rush_yds_pg", "rec_yds_pg",
        "td_pg", "var_a", "var_b",
    ]
    df = df[keep].sort_values(["athlete_id", "season"]).reset_index(drop=True)

    # Seasons of experience visible in this window. Left-censored for the 2020
    # cohort - a 2020 row shows 0 whether the player was a rookie or a
    # ten-year veteran - so it is only ever used alongside age, which is not.
    df["seasons_seen"] = df.groupby("athlete_id").cumcount()

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
