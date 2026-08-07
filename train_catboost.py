"""Predict per-game PPR fantasy points with CatBoost.

Train on 2021-2024, test on 2025. Every feature is knowable before kickoff:
prior-season player averages, prior-season opponent defense, age, and schedule
context. Same-game box score columns are dropped - they describe the game
being predicted.

Run:  python train_catboost.py
"""

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

# 2020 is loaded only to supply prior-season features for 2021 - it is never
# trained on, since 2019 is not in the database.
FEEDER_SEASON = 2021
TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON = 2025


# team_id / opponent_id are numeric-looking strings; as categories CatBoost
# treats them as labels instead of ordered numbers.
#
# athlete_id is deliberately NOT here. As a categorical CatBoost target-encodes
# it, which memorises a per-player scoring level: measured over 3 seeds it moved
# 2025 RMSE from 6.496 to 6.581, shifted predictions 1.1 points low, and cost
# 0.61 RMSE on the 21.5% of test rows whose player was unseen in training. The
# signal it was adding is career scoring level, which career_ppr_avg below
# carries explicitly and generalises to players the model has never seen.
CAT_COLS = ["home_away", "position_abbr", "team_id", "opponent_id"]

# Career-to-date features, built by build_career_features().
CAREER_COLS = ["career_ppr_avg", "career_games"]

# Age is measured at Sept 1 of the season - a fixed point just before week 1, and
# the usual fantasy convention. Using one date per season rather than each game's
# own date keeps two players in the same season directly comparable, and makes age
# a season-level trait rather than something that drifts 0.3 years by December.
AGE_REF_MMDD = "-09-01"

# Carried for slicing and inspection, not fed to the model.
META_COLS = ["display_name", "athlete_id", "season", "week"]


ID_COLS = [
    "display_name", "position_abbr", "athlete_id", "event_id", "team_id", "team_abbr",
    "opponent_id", "opponent_abbr", "home_away", "result", "game_date",
]
CALENDAR_COLS = ["season", "season_type", "week", "is_all_star"]

# Alternate scorings of the target - near-duplicates of fp_ppr, so they add nothing.
SKIP_STATS = ["fp_standard", "fp_half_ppr"]

# "games" is 17 for nearly every team, so it carries no signal.
DEF_ID_COLS = ["team_id", "team_abbr", "team_name", "season", "games"]

MODEL_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "early_stopping_rounds": 200,
    "verbose": 500,
}

IMPORTANCE_PNG = Path(__file__).with_name("feature_importance.png")


# ----------------------------------------------------------------------- load


def load_games(first_season: int, last_season: int) -> pd.DataFrame:
    frame = query_db(
        """
        SELECT *
        FROM v_player_games
        WHERE season_type = 2
          AND season BETWEEN ? AND ?
        ORDER BY athlete_id, season, week
        """,
        params=(first_season, last_season),
    )
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    return frame.drop(columns=["raw_stats", "loaded_at"])


def load_defense(first_season: int, last_season: int) -> pd.DataFrame:
    return query_db(
        "SELECT * FROM v_team_defense_seasons WHERE season BETWEEN ? AND ?",
        params=(first_season, last_season),
    )


def load_birth_dates() -> pd.DataFrame:
    """athlete_id -> birth_date, for the age feature.

    181 of the 1,493 rows in `athletes` have no birth_date, but none of them play
    in 2020-2025 - every one of the 1,186 players in the modelling frame is
    covered, so the NaN path below is defensive rather than load-bearing.
    """
    return query_db("SELECT athlete_id, birth_date FROM athletes")


# ------------------------------------------------------------------- features


def add_player_features(df: pd.DataFrame, first_train_season: int) -> pd.DataFrame:
    """Prior-season per-game averages, days rest, and the categorical casts."""
    stat_cols = [c for c in df.columns if c not in ID_COLS + CALENDAR_COLS + SKIP_STATS]

    # Per-player season averages, then shift the season forward so each row joins
    # to the season *before* it. Merging on season (rather than shifting rows)
    # means a player who missed a year gets no carry-over from two years back.
    season_avg = (
        df.groupby(["athlete_id", "season"])
        .agg(games=("event_id", "size"), **{c: (c, "mean") for c in stat_cols})
        .reset_index()
    )
    season_avg["season"] += 1
    season_avg = season_avg.rename(
        columns={**{c: f"prev_{c}_avg" for c in stat_cols}, "games": "prev_games"}
    )

    feat = df.merge(season_avg, on=["athlete_id", "season"], how="left")

    # The feeder season has no prior season of its own, so it drops out here.
    feat = feat[feat["season"] >= first_train_season].sort_values(
        ["athlete_id", "season", "week"]
    )

    # Days since that player's previous game, reset each season. The first
    # appearance of a season has no predecessor -> standard 7 days.
    feat["days_rest"] = feat.groupby(["athlete_id", "season"])["game_date"].diff().dt.days
    feat["days_rest"] = feat["days_rest"].fillna(7).astype(int)

    feat[CAT_COLS] = feat[CAT_COLS].astype("category")

    prev_cols = [c for c in feat.columns if c.startswith("prev_")]
    feat[prev_cols] = feat[prev_cols].fillna(0)
    return feat


def build_career_features(df: pd.DataFrame) -> pd.DataFrame:
    """Career scoring level to date, pooled over every strictly prior season.

    prev_fp_ppr_avg only sees the season immediately before, and is zeroed when
    that season is missing - so a player's longer track record is invisible to
    it, which is most of what separates a proven starter from a one-year sample.

    Built on the full frame including the feeder season, so a 2021 row sees 2020.
    Subtracting the current season from the cumulative sum is what keeps it
    strictly prior: a 2025 row is built from 2020-2024 and never from 2025.
    """
    season_tot = (
        df.groupby(["athlete_id", "season"])[TARGET]
        .agg(season_points="sum", season_games="size")
        .reset_index()
        .sort_values(["athlete_id", "season"])
    )
    grouped = season_tot.groupby("athlete_id")
    prior_points = grouped["season_points"].cumsum() - season_tot["season_points"]
    prior_games = grouped["season_games"].cumsum() - season_tot["season_games"]

    season_tot["career_ppr_avg"] = (prior_points / prior_games.replace(0, np.nan)).fillna(0)
    season_tot["career_games"] = prior_games
    return season_tot[["athlete_id", "season"] + CAREER_COLS]


def add_age_feature(feat: pd.DataFrame, birth: pd.DataFrame) -> pd.DataFrame:
    """Age in years at the start of the season the row belongs to.

    Constant across a player's games within a season by construction - see
    AGE_REF_MMDD. Fractional rather than whole years so the model can split on
    the aging curve finely; a 27.9-year-old RB is not a 27-year-old RB.

    Deliberately NOT zero-filled the way the prev_/opp_ blocks are: a missing
    birth date means unknown, and 0 would assert newborn. CatBoost handles NaN in
    numeric features natively, learning a side for it at each split, so it is
    left as NaN.
    """
    birth = birth.copy()
    birth["birth_date"] = pd.to_datetime(birth["birth_date"], errors="coerce")
    # Match the key dtype so the merge doesn't silently drop every row.
    birth["athlete_id"] = birth["athlete_id"].astype(feat["athlete_id"].dtype)

    feat = feat.merge(birth, on="athlete_id", how="left")

    season_start = pd.to_datetime(feat["season"].astype(str) + AGE_REF_MMDD)
    feat["age"] = (season_start - feat["birth_date"]).dt.days / 365.25
    return feat.drop(columns=["birth_date"])


def add_opponent_features(feat: pd.DataFrame, df_defense: pd.DataFrame) -> pd.DataFrame:
    """Prior-season defensive line for the opponent.

    Same shift-the-season trick as the player features: at prediction time the
    current season's defensive numbers don't exist yet, so last year's are the
    best available.
    """
    def_stat_cols = [c for c in df_defense.columns if c not in DEF_ID_COLS]

    opp_prev = df_defense[["team_id", "season"] + def_stat_cols].copy()
    opp_prev["season"] += 1
    opp_prev = opp_prev.rename(
        columns={**{c: f"opp_prev_{c}" for c in def_stat_cols}, "team_id": "opponent_id"}
    )
    # Match the key dtype so the merge doesn't downgrade opponent_id out of category.
    opp_prev["opponent_id"] = opp_prev["opponent_id"].astype(feat["opponent_id"].dtype)

    feat = feat.merge(opp_prev, on=["opponent_id", "season"], how="left")

    opp_cols = [c for c in feat.columns if c.startswith("opp_prev_")]
    feat[opp_cols] = feat[opp_cols].fillna(0)
    return feat


# ---------------------------------------------------------------------- split


def split(model_df: pd.DataFrame, feature_cols: list[str]):
    """Train 2021-2024, test 2025.

    This split was previously unusable. Prior seasons were backfilled from the
    2025 top-200 list, so 2022-2024 held only players still active in 2025 and
    averaged ~12 fp_ppr against 2025's 6.8; training across that boundary
    over-predicted by ~3 pts/game. The database now enumerates each season from
    its box scores (ffdb build-season), so 2020-2024 carry ~550 players each and
    the season means sit at 6.7-8.0 against 2025's 6.78 - one population.

    Early stopping needs data the model hasn't fit, and it should look like the
    test season, so the tail of 2024 is held out rather than a random slice.
    """
    in_train_seasons = model_df["season"].isin(TRAIN_SEASONS)
    is_valid = (model_df["season"] == 2024)

    train = model_df[in_train_seasons & ~is_valid]
    valid = model_df[is_valid]
    test = model_df[model_df["season"] == TEST_SEASON]

    def make_pool(x):
        return Pool(x[feature_cols], x[TARGET], cat_features=CAT_COLS)

    return (train, valid, test), tuple(make_pool(x) for x in (train, valid, test))


# ------------------------------------------------------------------ reporting


def metrics(y, pred) -> tuple[float, float]:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return np.sqrt(np.mean((y - pred) ** 2)), np.mean(np.abs(y - pred))


def report(model, train, valid, test, train_pool, valid_pool, test_pool) -> pd.DataFrame:
    print("\n=== fit ===")
    for name, x, pool in [
        ("train", train, train_pool),
        ("valid", valid, valid_pool),
        ("test", test, test_pool),
    ]:
        r, m = metrics(x[TARGET], model.predict(pool))
        print(f"{name:6s} n={len(x):6,d}  RMSE {r:6.3f}  MAE {m:6.3f}")

    # Baselines matter more than the raw RMSE: the naive "use last season's
    # average" model is the bar any of this has to clear. Do not compare against
    # the 6.413 from the old within-2025 split - that model had a shortcut this
    # one does not, and it was scored on 5 weeks rather than a full season.
    print(f"\n=== test ({TEST_SEASON}, full season) vs baselines ===")
    y_test = test[TARGET].to_numpy()
    pred = model.predict(test_pool)
    for label, p in [
        ("CatBoost", pred),
        ("prev_fp_ppr_avg passthrough", test["prev_fp_ppr_avg"].to_numpy()),
        ("constant = train mean", np.full(len(test), train[TARGET].mean())),
        ("constant = test mean (oracle)", np.full(len(test), y_test.mean())),
    ]:
        r, m = metrics(y_test, p)
        print(f"{label:32s} RMSE {r:6.3f}  MAE {m:6.3f}")

    print(f"\ncalibration -> actual {y_test.mean():.3f} | predicted {pred.mean():.3f}")

    scored = test.copy()
    scored["pred"] = pred

    print("\n=== by position ===")
    for pos, g in scored.groupby("position_abbr", observed=True):
        r, m = metrics(g[TARGET], g["pred"])
        print(
            f"{pos:3s} n={len(g):5,d}  RMSE {r:6.3f}  MAE {m:6.3f}  "
            f"actual {g[TARGET].mean():5.2f}  pred {g['pred'].mean():5.2f}"
        )

    # The honest check: how much of the win comes from players who simply have
    # no prior-season row (all-zero features, and reliably low scorers)?
    print("\n=== by prior-season availability ===")
    scored["has_prev"] = scored["prev_games"] > 0
    for flag, g in scored.groupby("has_prev"):
        r, m = metrics(g[TARGET], g["pred"])
        rb, mb = metrics(g[TARGET], g["prev_fp_ppr_avg"])
        print(
            f"has_prev={flag!s:5s} n={len(g):5,d}  actual {g[TARGET].mean():5.2f} | "
            f"CatBoost RMSE {r:6.3f} MAE {m:6.3f} | passthrough RMSE {rb:6.3f} MAE {mb:6.3f}"
        )
    return scored


def season_totals(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-player season totals: summed per-game predictions vs what happened.

    IMPORTANT - this is not a preseason projection. It sums over the games the
    player actually played, so it already knows who stayed healthy and who lost
    their job in week 4. A real preseason number would have to predict games
    played too, and that is where most season-long fantasy error actually lives.
    Read this as "given the games they played, how close was the per-game model
    in aggregate", not as "how good a draft projection is this".
    """
    totals = (
        scored.groupby(["athlete_id", "display_name", "position_abbr"], observed=True)
        .agg(
            games=("pred", "size"),
            actual=(TARGET, "sum"),
            projected=("pred", "sum"),
            baseline=("prev_fp_ppr_avg", "sum"),
        )
        .reset_index()
    )
    totals["error"] = totals["projected"] - totals["actual"]
    totals["abs_error"] = totals["error"].abs()
    # Undefined for a player who scored nothing all year, so left as NaN there.
    totals["pct_error"] = 100 * totals["error"] / totals["actual"].replace(0, np.nan)
    return totals.sort_values("actual", ascending=False).reset_index(drop=True)


def report_season_totals(scored: pd.DataFrame, min_points: float = 100.0) -> pd.DataFrame:
    totals = season_totals(scored)

    print(f"\n=== {TEST_SEASON} season totals: projected vs actual ===")
    print("Summed over games actually played, so this is not a preseason "
          "projection -\nit already knows who stayed healthy.")

    for label, sub in [
        (f"all {len(totals)} players", totals),
        (f"actual >= {min_points:.0f} pts", totals[totals["actual"] >= min_points]),
    ]:
        r, m = metrics(sub["actual"], sub["projected"])
        rb, mb = metrics(sub["actual"], sub["baseline"])
        med = sub["abs_error"].median()
        med_pct = sub["pct_error"].abs().median()
        print(
            f"\n{label:24s} n={len(sub):4,d}  mean actual {sub['actual'].mean():7.1f}\n"
            f"  model     RMSE {r:7.1f}  MAE {m:7.1f}  median |err| {med:6.1f} ({med_pct:5.1f}%)\n"
            f"  baseline  RMSE {rb:7.1f}  MAE {mb:7.1f}"
        )

    print("\n--- by position (actual >= "
          f"{min_points:.0f} pts) ---")
    big = totals[totals["actual"] >= min_points]
    print(f"{'pos':>4} {'n':>4} {'mean actual':>12} {'MAE':>8} {'median |err|':>13} {'median |%|':>11}")
    for pos, g in big.groupby("position_abbr", observed=True):
        _, m = metrics(g["actual"], g["projected"])
        print(
            f"{pos:>4} {len(g):>4} {g['actual'].mean():>12.1f} {m:>8.1f} "
            f"{g['abs_error'].median():>13.1f} {g['pct_error'].abs().median():>10.1f}%"
        )

    cols = ["display_name", "position_abbr", "games", "actual", "projected", "error", "pct_error"]
    fmt = {"actual": "{:.1f}", "projected": "{:.1f}", "error": "{:+.1f}", "pct_error": "{:+.1f}"}

    print("\n--- top 15 by actual points ---")
    print(totals.head(15)[cols].to_string(index=False, formatters={k: v.format for k, v in fmt.items()}))

    print(f"\n--- most over-projected (actual >= {min_points:.0f}) ---")
    print(big.nlargest(8, "error")[cols].to_string(index=False, formatters={k: v.format for k, v in fmt.items()}))

    print(f"\n--- most under-projected (actual >= {min_points:.0f}) ---")
    print(big.nsmallest(8, "error")[cols].to_string(index=False, formatters={k: v.format for k, v in fmt.items()}))

    return totals


def plot_importance(model, train_pool, feature_cols: list[str]) -> pd.DataFrame:
    imp = (
        pd.DataFrame(
            {"feature": feature_cols, "importance": model.get_feature_importance(train_pool)}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top = imp.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(f"Top 25 features - target: {TARGET}")
    fig.tight_layout()
    fig.savefig(IMPORTANCE_PNG, dpi=120)
    plt.close(fig)
    print(f"\nfeature importance plot -> {IMPORTANCE_PNG}")
    return imp


# ----------------------------------------------------------------------- main


def main() -> None:
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 200)

    assert DB_PATH.exists(), f"DB not found: {DB_PATH}"
    print(DB_PATH, f"{DB_PATH.stat().st_size / 1e6:.1f} MB")

    first_train_season = min(TRAIN_SEASONS)
    df = load_games(FEEDER_SEASON, TEST_SEASON)
    df_defense = load_defense(FEEDER_SEASON, TEST_SEASON)

    # Built from the full frame, before the feeder season is filtered out.
    career = build_career_features(df)

    feat = add_player_features(df, first_train_season)
    prev_cols = [c for c in feat.columns if c.startswith("prev_")]

    feat = add_opponent_features(feat, df_defense)
    opp_cols = [c for c in feat.columns if c.startswith("opp_prev_")]

    feat = feat.merge(career, on=["athlete_id", "season"], how="left")
    feat[CAREER_COLS] = feat[CAREER_COLS].fillna(0)

    feat = add_age_feature(feat, load_birth_dates())

    feature_cols = CAT_COLS + ["days_rest", "age"] + CAREER_COLS + prev_cols + opp_cols
    model_df = feat[META_COLS + feature_cols + [TARGET]].reset_index(drop=True)

    print(
        f"\n{model_df.shape[0]:,} rows x {model_df.shape[1]} cols "
        f"({len(feature_cols)} features + {len(META_COLS)} meta + target)"
    )
    print(f"  categorical      : {len(CAT_COLS)}  {CAT_COLS}")
    print(f"  career-to-date   : {len(CAREER_COLS)}  {CAREER_COLS}")
    print(f"  player prior-year: {len(prev_cols)}")
    print(f"  opponent defense : {len(opp_cols)}")
    age = model_df["age"]
    print(
        f"  age at Sept 1    : {age.min():.1f}-{age.max():.1f}, mean {age.mean():.1f}, "
        f"{int(age.isna().sum())} missing"
    )
    print("  NaNs:", int(model_df.isna().sum().sum()))

    (train, valid, test), (train_pool, valid_pool, test_pool) = split(model_df, feature_cols)

    for name, x in [("train", train), ("valid", valid), ("test", test)]:
        print(
            f"{name:6s} {len(x):6,d} rows  seasons {sorted(x['season'].unique())}  "
            f"mean {TARGET} {x[TARGET].mean():6.3f}"
        )

    # The level check that killed the old split: these means have to be comparable.
    # 2021 reads ~1 pt high because ESPN omits some scoreless appearances that
    # season (15.6% zero-point games vs ~22-26% either side, holding the player
    # set fixed) - the rows that are there are fine, the quiet games are missing.
    print(f"\nmean {TARGET} by season")
    for s, g in model_df.groupby("season"):
        print(f"  {s}  n={len(g):6,d}  {g[TARGET].mean():6.3f}")

    # Was the shortcut feature: rows with no prior-season row averaged 2.95 vs
    # 12.01 because the DB simply lacked those rows. The gap is now ~3, and the
    # same check on 2024 vs 2023 gives the same number, so what is left is
    # signal rather than artifact.
    cold = test["prev_games"] == 0
    print(
        f"\n{TEST_SEASON} rows with {TEST_SEASON - 1} history: "
        f"{(~cold).sum():,} ({(~cold).mean():.1%}), "
        f"mean {TARGET} {test.loc[~cold, TARGET].mean():.2f}"
    )
    print(
        f"{TEST_SEASON} rows without           : {cold.sum():,} ({cold.mean():.1%}), "
        f"mean {TARGET} {test.loc[cold, TARGET].mean():.2f}"
    )

    model = CatBoostRegressor(**MODEL_PARAMS)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    print("best iteration:", model.get_best_iteration())

    scored = report(model, train, valid, test, train_pool, valid_pool, test_pool)
    report_season_totals(scored)
    imp = plot_importance(model, train_pool, feature_cols)
    print("\n=== top 15 features ===")
    print(imp.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
