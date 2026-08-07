"""Expanding-window backtest of the Bayesian projection model.

Every fold trains on seasons up to a cutoff and is scored on what actually
happened the season after. Nothing is ever trained on a season later than the
one it predicts, and the split is by season rather than at random - a player's
2022 and 2023 rows share most of their signal, so a random split would let the
model see a version of the answer.

Run:  python run_backtest.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bayes.data import LAST_SEASON, attach_next_season, build_panel
from bayes.figures import diagnostics
from bayes.metrics import coverage_given_played, pit, summarise, summarise_point
from bayes.predictive import fit_fold

OUT = Path(__file__).parent / "artifacts"


def run(quick: bool = False, use_inputs: bool = True) -> pd.DataFrame:
    OUT.mkdir(exist_ok=True)
    panel = attach_next_season(build_panel())

    warmup, samples, chains = (300, 400, 2) if quick else (800, 1000, 4)

    # The first fold trains on a single season, so it has no transition to learn
    # availability from. Start where there is at least one.
    cutoffs = list(range(panel["season"].min() + 2, LAST_SEASON))

    records, folds = [], {}
    for cutoff in cutoffs:
        t0 = time.time()
        proj = fit_fold(
            panel,
            cutoff=cutoff,
            use_inputs=use_inputs,
            num_warmup=warmup,
            num_samples=samples,
            num_chains=chains,
            progress=False,
        )
        rows = proj.rows
        y = rows["next_fp_ppr"].to_numpy()

        entries = {
            "bayes": summarise(proj.draws, y),
            "repeat last season": summarise_point(rows["fp_ppr"].to_numpy(), y),
            "last season ppg x 17": summarise_point(
                rows["ppg"].to_numpy() * 17, y
            ),
            "position mean": summarise_point(
                rows.groupby("pos")["fp_ppr"].transform("mean").to_numpy(), y
            ),
        }
        for name, m in entries.items():
            records.append({"predicts": cutoff + 1, "model": name, **m})

        folds[cutoff] = {
            "rows": rows,
            "draws": proj.draws,
            "games": proj.games_draws,
            "production_summary": _param_summary(proj.production_mcmc),
        }
        b = entries["bayes"]
        print(
            f"predicts {cutoff + 1}: RMSE {b['rmse']:6.2f}  MAE {b['mae']:6.2f}  "
            f"CRPS {b['crps']:6.2f}  cov80 {b['cov80']:.2f}  | "
            f"repeat-last-season RMSE {entries['repeat last season']['rmse']:6.2f}   "
            f"({time.time() - t0:.0f}s)"
        )

    results = pd.DataFrame(records)
    results.to_csv(OUT / "backtest_results.csv", index=False)
    with open(OUT / "backtest_folds.pkl", "wb") as fh:
        pickle.dump(folds, fh)

    _report(results, folds)
    return results


def _param_summary(mcmc) -> pd.DataFrame:
    post = mcmc.get_samples()
    rows = {}
    for k, v in post.items():
        v = np.asarray(v)
        if v.ndim == 2:
            for j in range(v.shape[1]):
                rows[f"{k}[{j}]"] = (v[:, j].mean(), v[:, j].std())
        elif v.ndim == 1:
            rows[k] = (v.mean(), v.std())
    return pd.DataFrame(rows, index=["mean", "sd"]).T


def _report(results: pd.DataFrame, folds: dict) -> None:
    pd.set_option("display.width", 220)

    print("\n=== per fold ===")
    print(
        results.pivot(index="predicts", columns="model", values="rmse")
        .round(2)
        .to_string()
    )

    print("\n=== averaged over folds ===")
    agg = (
        results.groupby("model")[
            ["rmse", "mae", "crps", "cov50", "cov80", "cov90", "spearman", "bias"]
        ]
        .mean()
        .sort_values("crps")
    )
    print(agg.round(3).to_string())

    # Pooling every fold's rows gives one calibration check with real sample
    # size behind it rather than four noisy ones.
    all_draws = np.concatenate([f["draws"] for f in folds.values()], axis=1)
    all_y = np.concatenate(
        [f["rows"]["next_fp_ppr"].to_numpy() for f in folds.values()]
    )
    all_games = np.concatenate([f["games"] for f in folds.values()], axis=1)
    all_y_games = np.concatenate(
        [f["rows"]["next_games"].to_numpy() for f in folds.values()]
    )
    print("\n=== interval coverage ===")
    print("  level   all players   given he plays")
    for lv in (0.5, 0.8, 0.9):
        lo, hi = np.quantile(all_draws, [(1 - lv) / 2, 1 - (1 - lv) / 2], axis=0)
        uncond = np.mean((all_y >= lo) & (all_y <= hi))
        cond = coverage_given_played(all_draws, all_games, all_y, all_y_games, lv)
        print(f"  {lv:5.0%}   {uncond:11.3f}   {cond:14.3f}")
    print(
        "  (the unconditional predictive has an atom at zero, so a central\n"
        "   interval spanning the gap over-covers even when calibrated)"
    )

    u = pit(all_draws, all_y)
    counts, _ = np.histogram(u, bins=10, range=(0, 1))
    print(f"\n=== PIT over all {len(all_y):,} pooled predictions ===")
    print("uniform would be", f"{len(all_y) / 10:.0f} per decile")
    print(" ".join(f"{c:5d}" for c in counts))

    # Does the projection mean what it says? Bucketing by projection and
    # comparing to realised points catches shrinkage that is too strong or too
    # weak, which a single RMSE hides.
    pred = all_draws.mean(axis=0)
    bucket = pd.qcut(pred, 10, labels=False)
    cal = pd.DataFrame({"pred": pred, "actual": all_y, "b": bucket})
    print("\n=== calibration by projection decile ===")
    print(
        cal.groupby("b")
        .agg(n=("pred", "size"), mean_proj=("pred", "mean"), mean_actual=("actual", "mean"))
        .round(1)
        .to_string()
    )
    slope = np.polyfit(pred, all_y, 1)
    print(f"\nactual ~ a + b*projection:  b = {slope[0]:.3f} (1.0 = correctly shrunk)")

    (OUT / "backtest_summary.json").write_text(
        json.dumps(
            {
                "per_fold_rmse": results.pivot(
                    index="predicts", columns="model", values="rmse"
                ).round(3).to_dict(),
                "averaged": agg.round(4).to_dict(),
                "calibration_slope": float(slope[0]),
            },
            indent=2,
        )
    )
    fig = diagnostics(folds, OUT / "diagnostics.png")
    print(f"\nwrote {OUT / 'backtest_results.csv'}, {OUT / 'backtest_folds.pkl'}, {fig}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short chains, for iteration")
    ap.add_argument(
        "--no-inputs", action="store_true", help="drop the usage / late-form inputs"
    )
    args = ap.parse_args()
    run(quick=args.quick, use_inputs=not args.no_inputs)
