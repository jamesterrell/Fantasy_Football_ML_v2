from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from helpers.db_query import query_db

matplotlib.use("Agg")  # plots are written to file rather than shown

# --------------------------------------------------------------------- config

DB_PATH = Path(
    os.environ.get(
        "FF_DB_PATH",
        r"C:\Users\terre\OneDrive\Desktop\Projects\Fantasy_Football_Database\data\fantasy_football.db",
    )
)

TARGET = "fp_ppr"

FIRST_SEASON, LAST_SEASON = 2021, 2025

# One row per athlete_id x season. Counting stats are summed, "long" stats take
# the max, and rate stats are recomputed from the summed components rather than
# averaged - the mean of per-game yards-per-carry is not season yards-per-carry,
# because it weights a 3-carry game the same as a 25-carry one.
#
# Only athlete_id and season are grouped on. Adding team_id/team_abbr here would
# split a traded player into one row per team, which breaks the grain and would
# scatter his games_* columns below across two rows. display_name and
# position_abbr are aggregated rather than grouped on for the same reason - a
# player relisted at a different position mid-season would otherwise split too.
df = query_db(
    """
    SELECT
        athlete_id,
        season,
        MAX(display_name)               AS display_name,
        MAX(position_abbr)              AS position_abbr,
        COUNT(*)                        AS games,

        -- Per-game columns are SUM/COUNT(*) rather than AVG(). AVG() skips NULL
        -- rows, so a back with no receiving line in 6 of 17 games would be
        -- averaged over 11 - inflating his per-game receiving. Dividing by
        -- COUNT(*) keeps the denominator at games actually played.

        -- fantasy points
        SUM(fp_ppr)                     AS fp_ppr,

        -- passing
        SUM(passingAttempts)            AS passingAttempts,
        SUM(completions)                AS completions,
        SUM(passingYards)               AS passingYards,
        SUM(passingTouchdowns)          AS passingTouchdowns,
        SUM(interceptions)              AS interceptions,
        SUM(sacks)                      AS sacks,
        MAX(longPassing)                AS longPassing,
        100.0 * SUM(completions) / NULLIF(SUM(passingAttempts), 0)
                                        AS completionPct,
        SUM(passingYards) / NULLIF(SUM(passingAttempts), 0)
                                        AS yardsPerPassAttempt,
        SUM(passingAttempts)   * 1.0 / COUNT(*) AS passingAttempts_pg,
        SUM(completions)       * 1.0 / COUNT(*) AS completions_pg,
        SUM(passingYards)      * 1.0 / COUNT(*) AS passingYards_pg,
        SUM(passingTouchdowns) * 1.0 / COUNT(*) AS passingTouchdowns_pg,
        SUM(interceptions)     * 1.0 / COUNT(*) AS interceptions_pg,
        SUM(sacks)             * 1.0 / COUNT(*) AS sacks_pg,

        -- rushing
        SUM(rushingAttempts)            AS rushingAttempts,
        SUM(rushingYards)               AS rushingYards,
        SUM(rushingTouchdowns)          AS rushingTouchdowns,
        MAX(longRushing)                AS longRushing,
        SUM(rushingYards) / NULLIF(SUM(rushingAttempts), 0)
                                        AS yardsPerRushAttempt,
        SUM(rushingAttempts)   * 1.0 / COUNT(*) AS rushingAttempts_pg,
        SUM(rushingYards)      * 1.0 / COUNT(*) AS rushingYards_pg,
        SUM(rushingTouchdowns) * 1.0 / COUNT(*) AS rushingTouchdowns_pg,

        -- receiving
        SUM(receivingTargets)           AS receivingTargets,
        SUM(receptions)                 AS receptions,
        SUM(receivingYards)             AS receivingYards,
        SUM(receivingTouchdowns)        AS receivingTouchdowns,
        MAX(longReception)              AS longReception,
        SUM(receivingYards) / NULLIF(SUM(receptions), 0)
                                        AS yardsPerReception,
        SUM(receivingTargets)    * 1.0 / COUNT(*) AS receivingTargets_pg,
        SUM(receptions)          * 1.0 / COUNT(*) AS receptions_pg,
        SUM(receivingYards)      * 1.0 / COUNT(*) AS receivingYards_pg,
        SUM(receivingTouchdowns) * 1.0 / COUNT(*) AS receivingTouchdowns_pg,

        -- misc
        SUM(fumbles)                    AS fumbles,
        SUM(fumblesLost)                AS fumblesLost,
        SUM(fumbles)     * 1.0 / COUNT(*) AS fumbles_pg,
        SUM(fumblesLost) * 1.0 / COUNT(*) AS fumblesLost_pg

    FROM v_player_games
    WHERE season_type = 2
        AND season BETWEEN ? AND ?
    GROUP BY athlete_id, season
    ORDER BY athlete_id, season
    """,
    params=(FIRST_SEASON, LAST_SEASON),
)

print(df.head())


# --------------------------------------------------------------- next-year target
#
# Each row gets the PPR total that player put up the *following* season, so a
# 2024 row carries his 2025 points. Shifting the season back one and merging
# (rather than .shift()) means a player who missed a year gets no carry-over
# from two seasons later.
next_year = df[["athlete_id", "season", "fp_ppr"]].rename(
    columns={"fp_ppr": "next_fp_ppr"}
)
next_year["season"] -= 1

df = df.merge(next_year, on=["athlete_id", "season"], how="left")

# No next-season row means the player didn't appear in it -> 0 fantasy points.
# LAST_SEASON is excluded because its NaNs mean "hasn't happened yet", and
# zeroing those would label every 2025 player a bust.
has_next = df["season"] < LAST_SEASON
df.loc[has_next, "next_fp_ppr"] = df.loc[has_next, "next_fp_ppr"].fillna(0)

print(df[["athlete_id", "season", "fp_ppr", "next_fp_ppr"]].head(10))


# ------------------------------------------------------------------------- age
#
# Age at Sept 1 of the row's own season - a fixed point just before week 1, and
# the usual fantasy convention. Anchoring to the season rather than to any
# individual game means a player is straightforwardly a year older in his 2024
# row than his 2023 one, and two players in the same season stay comparable.
#
# Fractional years rather than whole, so the model can split the aging curve
# finely: a 27.9-year-old RB is not a 27-year-old RB.
AGE_REF_MMDD = "-09-01"

birth = query_db("SELECT athlete_id, birth_date FROM athletes")
birth["birth_date"] = pd.to_datetime(birth["birth_date"], errors="coerce")
# Match the key dtype so the merge doesn't silently drop every row.
birth["athlete_id"] = birth["athlete_id"].astype(df["athlete_id"].dtype)

df = df.merge(birth, on="athlete_id", how="left")
season_start = pd.to_datetime(df["season"].astype(str) + AGE_REF_MMDD)
# Left as NaN where birth_date is unknown - 0 would assert newborn, and CatBoost
# handles NaN in numeric features natively.
df["age"] = (season_start - df["birth_date"]).dt.days / 365.25
df = df.drop(columns=["birth_date"])

print(
    f"\nage: {df['age'].min():.1f}-{df['age'].max():.1f}, "
    f"mean {df['age'].mean():.1f}, {int(df['age'].isna().sum())} missing"
)



# train = df[df.season < 2024]
# validate = df[df.season == 2024]
# test = df[df.season == 2025]