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


# ------------------------------------------------------------------- modelling
#
# A row's features describe season N; the label is season N+1's PPR total. So
# the last labelled season is LAST_SEASON - 1: a 2024 row is scored against what
# actually happened in 2025, and 2025 rows are the *projection* set, with no
# label because 2026 hasn't been played.
#
# Split is by season, never randomly. A random split would put a player's 2022
# and 2023 rows on both sides, and those rows share most of their signal.
TEST_SEASON = LAST_SEASON - 1        # 2024 -> scored against 2025
VALID_SEASON = LAST_SEASON - 2       # 2023 -> early stopping
TRAIN_SEASONS = list(range(FIRST_SEASON, VALID_SEASON))  # 2021-2022
PROJECT_SEASON = LAST_SEASON         # 2025 -> unlabelled, predicts 2026

LABEL = "next_fp_ppr"

# athlete_id is deliberately not a feature. Target-encoding it lets CatBoost
# memorise a per-player scoring level, which is the same bet as a random
# intercept in a mixed model - and at the per-game grain it measurably hurt
# (see the note in train_catboost.py). season is held out too: the model would
# learn a level for 2021-2022 that means nothing when applied to 2025.
META_COLS = ["display_name", "season"]
CAT_COLS = ["athlete_id", "position_abbr"]

FEATURE_COLS = [c for c in df.columns if c not in META_COLS + [LABEL]]

# CatBoost rejects NaN in categorical features, unlike numeric ones.
df[CAT_COLS] = df[CAT_COLS].fillna("UNK").astype(str)

MODEL_PARAMS = {
    # RMSE is the honest default but not the best fit for this label: 29% of
    # rows are exactly 0 and the rest is a skewed positive tail. Swap in
    # "Tweedie:variance_power=1.5" to model that point mass directly - it takes
    # no other changes, since Tweedie is still a regression on the same target.
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 3000,
    "learning_rate": 0.03,
    # Shallower and more regularised than the per-game model: ~1,100 training
    # rows against 40-odd features is a much easier frame to overfit.
    "depth": 5,
    "l2_leaf_reg": 6.0,
    "random_seed": 42,
    "early_stopping_rounds": 200,
    "verbose": 500,
}

IMPORTANCE_PNG = Path(__file__).with_name("season_feature_importance.png")


def make_pool(frame: pd.DataFrame) -> Pool:
    return Pool(frame[FEATURE_COLS], frame[LABEL], cat_features=CAT_COLS)


def metrics(y, pred) -> tuple[float, float]:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return np.sqrt(np.mean((y - pred) ** 2)), np.mean(np.abs(y - pred))


train = df[df["season"].isin(TRAIN_SEASONS)]
valid = df[df["season"] == VALID_SEASON]
test = df[df["season"] == TEST_SEASON]
project = df[df["season"] == PROJECT_SEASON]

assert train[LABEL].notna().all() and valid[LABEL].notna().all()
assert test[LABEL].notna().all(), "test season must be labelled - check LAST_SEASON"
assert project[LABEL].isna().all(), "projection season should have no label"

print(f"\n{len(FEATURE_COLS)} features, label {LABEL}")
for name, part, seasons in [
    ("train", train, TRAIN_SEASONS),
    ("valid", valid, [VALID_SEASON]),
    ("test", test, [TEST_SEASON]),
    ("project", project, [PROJECT_SEASON]),
]:
    lbl = part[LABEL]
    tail = "unlabelled" if lbl.isna().all() else f"mean label {lbl.mean():6.1f}"
    print(f"  {name:8s} {len(part):5,d} rows  seasons {seasons}  {tail}")

model = CatBoostRegressor(**MODEL_PARAMS)
model.fit(make_pool(train), eval_set=make_pool(valid), use_best_model=True)
print("best iteration:", model.get_best_iteration())

pred = model.predict(make_pool(test))

# The bar to clear is not zero - it is "assume he repeats last season", which is
# already a strong projection at a year-over-year correlation of 0.76.
print(f"\n=== test: {TEST_SEASON} rows scored against {TEST_SEASON + 1} ===")
y_test = test[LABEL].to_numpy()
for label, p in [
    ("CatBoost", pred),
    ("repeat this season's fp_ppr", test["fp_ppr"].to_numpy()),
    ("constant = train mean", np.full(len(test), train[LABEL].mean())),
    ("constant = test mean (oracle)", np.full(len(test), y_test.mean())),
]:
    r, m = metrics(y_test, p)
    print(f"  {label:32s} RMSE {r:7.2f}  MAE {m:7.2f}")

print(f"\n  calibration -> actual {y_test.mean():.1f} | predicted {pred.mean():.1f}")

scored = test.copy()
scored["pred"] = pred

print("\n=== by position ===")
for pos, g in scored.groupby("position_abbr", observed=True):
    r, m = metrics(g[LABEL], g["pred"])
    print(
        f"  {pos:3s} n={len(g):4,d}  RMSE {r:7.2f}  MAE {m:7.2f}  "
        f"actual {g[LABEL].mean():6.1f}  pred {g['pred'].mean():6.1f}"
    )

# Zeros are 29% of the label and a different question from "how many points" -
# splitting them out shows whether the model is projecting the drop-offs at all,
# or just regressing everyone toward the middle.
print("\n=== by outcome: did they play the next season? ===")
for flag, g in scored.groupby(scored[LABEL] == 0):
    r, m = metrics(g[LABEL], g["pred"])
    name = "did NOT play" if flag else "played"
    print(
        f"  {name:12s} n={len(g):4,d}  actual {g[LABEL].mean():6.1f}  "
        f"pred {g['pred'].mean():6.1f}  RMSE {r:7.2f}  MAE {m:7.2f}"
    )


# ------------------------------------------------------- expanding-window check
#
# There are only four season boundaries in the data, so a single test season is
# one noisy number. Retraining on each boundary in turn - always predicting
# forward, never backward - gives four of them for the price of a few seconds.
#
# Iterations are fixed rather than early-stopped here: with the fold's own
# validation set doubling as its score, early stopping would tune on the thing
# being measured.
def expanding_window(fixed_iterations: int) -> pd.DataFrame:
    rows = []
    for holdout in range(FIRST_SEASON + 1, LAST_SEASON):
        tr = df[df["season"] < holdout]
        te = df[df["season"] == holdout]
        params = {**MODEL_PARAMS, "iterations": fixed_iterations, "verbose": 0}
        params.pop("early_stopping_rounds")
        m = CatBoostRegressor(**params).fit(make_pool(tr))
        r, mae = metrics(te[LABEL], m.predict(make_pool(te)))
        rb, maeb = metrics(te[LABEL], te["fp_ppr"])
        rows.append(
            {"holdout": holdout, "n_train": len(tr), "n_test": len(te),
             "rmse": r, "mae": mae, "baseline_rmse": rb, "baseline_mae": maeb}
        )
    return pd.DataFrame(rows)


cv = expanding_window(max(model.get_best_iteration(), 100))
print("\n=== expanding window (train on all prior seasons, test on holdout) ===")
print(cv.round(2).to_string(index=False))
print(
    f"  mean RMSE {cv['rmse'].mean():.2f} vs baseline {cv['baseline_rmse'].mean():.2f}"
)


# ------------------------------------------------------------------ projections

imp = (
    pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": model.get_feature_importance(make_pool(train))}
    )
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
print("\n=== top 15 features ===")
print(imp.head(15).to_string(index=False))

top = imp.head(25).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(top["feature"], top["importance"])
ax.set_title(f"Top 25 features - label: {LABEL}")
fig.tight_layout()
fig.savefig(IMPORTANCE_PNG, dpi=120)
plt.close(fig)
print(f"\nfeature importance plot -> {IMPORTANCE_PNG}")

# The actual product: what each 2025 player is projected to score in 2026.
projected = project[META_COLS + ["position_abbr", "games", "fp_ppr", "age"]].copy()
projected["projected_next"] = model.predict(
    Pool(project[FEATURE_COLS], cat_features=CAT_COLS)
)
projected = projected.sort_values("projected_next", ascending=False).reset_index(drop=True)

print(f"\n=== top 25 projections for {PROJECT_SEASON + 1} ===")
print(
    projected.head(25)
    .drop(columns=["season"])
    .to_string(index=False, formatters={
        "fp_ppr": "{:.1f}".format,
        "projected_next": "{:.1f}".format,
        "age": "{:.1f}".format,
    })
)