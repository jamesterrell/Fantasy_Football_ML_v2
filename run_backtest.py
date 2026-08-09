"""Expanding-window backtest of the Bayesian projection model.

Every fold trains on seasons up to a cutoff and is scored on what actually
happened the season after. Nothing is ever trained on a season later than the
one it predicts, and the split is by season rather than at random - a player's
2022 and 2023 rows share most of their signal, so a random split would let the
model see a version of the answer.

The model is fit on the **draftable universe**: players who have posted at
least one ``MIN_PEAK_FP``-point season. Because that restriction changes the
population being scored, comparing the filtered model against the old numbers
would compare two different questions. So each fold fits twice - once on the
filtered panel, once on everything - and scores both on the *same* filtered
rows. That isolates what dropping the undraftable players from training did,
which is the only thing the change controls.

Run:  python run_backtest.py [--quick] [--modes causal,window,off]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bayes.data import (
    LAST_SEASON,
    MIN_PEAK_FP,
    apply_peak_filter,
    attach_next_season,
    build_panel,
    qualifying_players,
)
from bayes.figures import diagnostics
from bayes.metrics import (
    conditional_on_playing,
    coverage_given_played,
    pit,
    summarise,
    summarise_point,
)
from bayes.predictive import fit_fold

OUT = Path(__file__).parent / "artifacts"

# How a fold decides who counts as draftable.
#   causal - qualified on seasons the fold has already seen. The honest rule:
#            it is what a drafter standing in that August could have applied.
#   window - qualified anywhere in 2021-2025, the filter as literally stated.
#            Reported to size the lookahead, not to be believed: it selects the
#            universe on outcomes the fold is being scored against.
#   off    - no filter, the whole league. The pre-change reference.
MODES = ("causal", "window", "off")


def _subset_projection(proj, athlete_ids):
    """Restrict a fitted fold's rows and draws to ``athlete_ids``, in that order.

    Used to score the all-players model on the draftable rows only. Aligning by
    athlete id rather than trusting two frames to be in the same order is the
    difference between a comparison and a silent mismatch.
    """
    pos = {a: i for i, a in enumerate(proj.rows["athlete_id"].to_numpy())}
    idx = np.array([pos[a] for a in athlete_ids if a in pos])
    return proj.rows.iloc[idx].reset_index(drop=True), proj.draws[:, idx], proj.games_draws[:, idx]


def run(
    quick: bool = False,
    use_inputs: bool = True,
    modes=("causal", "window"),
    chains_spec: tuple[int, int, int] | None = None,
) -> pd.DataFrame:
    OUT.mkdir(exist_ok=True)
    panel = attach_next_season(build_panel())

    warmup, samples, chains = chains_spec or ((300, 400, 2) if quick else (800, 1000, 4))
    fit_kw = dict(
        use_inputs=use_inputs,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        progress=False,
    )

    # The first fold trains on a single season, so it has no transition to learn
    # availability from. Start where there is at least one.
    cutoffs = list(range(panel["season"].min() + 2, LAST_SEASON))

    records, folds = [], {}
    for cutoff in cutoffs:
        # The all-players fit is the same model regardless of which filter mode
        # is being evaluated, so it is fit once and re-scored per mode.
        t0 = time.time()
        proj_all = fit_fold(panel, cutoff=cutoff, **fit_kw)
        print(f"[{cutoff + 1}] all-players fit {time.time() - t0:.0f}s")

        for mode in modes:
            if mode == "off":
                _score(records, folds, "off", cutoff, proj_all, proj_all)
                continue

            through = cutoff if mode == "causal" else None
            sub = apply_peak_filter(panel, through=through)

            t0 = time.time()
            proj_f = fit_fold(sub, cutoff=cutoff, **fit_kw)
            print(
                f"[{cutoff + 1}] {mode} fit {time.time() - t0:.0f}s "
                f"({sub['athlete_id'].nunique()} players, {len(sub)} rows)"
            )
            _score(records, folds, mode, cutoff, proj_f, proj_all)

    results = pd.DataFrame(records)
    results.to_csv(OUT / "backtest_results.csv", index=False)
    with open(OUT / "backtest_folds.pkl", "wb") as fh:
        pickle.dump(folds, fh)

    _report(results, folds, modes)
    return results


def _score(records, folds, mode, cutoff, proj_f, proj_all) -> None:
    """Score the filtered model and the reference arms on the filtered rows."""
    rows = proj_f.rows
    y = rows["next_fp_ppr"].to_numpy()

    entries = {
        "bayes (filtered fit)": summarise(proj_f.draws, y),
        "repeat last season": summarise_point(rows["fp_ppr"].to_numpy(), y),
        "last season ppg x 17": summarise_point(rows["ppg"].to_numpy() * 17, y),
        "position mean": summarise_point(
            rows.groupby("pos")["fp_ppr"].transform("mean").to_numpy(), y
        ),
    }

    # The number a drafter actually faces. Who is out of the league is known in
    # August and handled outside the model, so both sides are conditioned on it:
    # the predictive keeps only its playing draws, and the scoring keeps only the
    # players who did play. What is left is points given he is on a roster -
    # injuries included, since those are not knowable in August.
    active = rows["next_games"].to_numpy() > 0
    cond, _ = conditional_on_playing(proj_f.draws, proj_f.games_draws)
    entries["bayes | in the league"] = summarise(cond[:, active], y[active])
    entries["repeat last season | in the league"] = summarise_point(
        rows["fp_ppr"].to_numpy()[active], y[active]
    )

    # The all-players model, scored on exactly these rows. Skipped in "off"
    # mode, where it is the same object as the filtered fit.
    if mode != "off":
        rows_a, draws_a, games_a = _subset_projection(
            proj_all, rows["athlete_id"].to_numpy()
        )
        assert np.array_equal(
            rows_a["athlete_id"].to_numpy(), rows["athlete_id"].to_numpy()
        ), "row alignment between the two fits"
        entries["bayes (all-players fit)"] = summarise(
            draws_a, rows_a["next_fp_ppr"].to_numpy()
        )
        cond_a, _ = conditional_on_playing(draws_a, games_a)
        entries["bayes (all-players fit) | in the league"] = summarise(
            cond_a[:, active], y[active]
        )

    for name, m in entries.items():
        records.append({"mode": mode, "predicts": cutoff + 1, "model": name, **m})

    folds[(mode, cutoff)] = {
        "rows": rows,
        "draws": proj_f.draws,
        "games": proj_f.games_draws,
        "production_summary": _param_summary(proj_f.production_mcmc),
    }

    b = entries["bayes (filtered fit)"]
    ref = entries.get("bayes (all-players fit)", b)
    c = entries["bayes | in the league"]
    print(
        f"  {mode:7s} predicts {cutoff + 1}  n={b['n']:4d}  "
        f"RMSE {b['rmse']:6.2f} (all-players fit {ref['rmse']:6.2f})  "
        f"CRPS {b['crps']:6.2f}  cov80 {b['cov80']:.2f}  "
        f"|| in-league n={c['n']:4d} RMSE {c['rmse']:6.2f} CRPS {c['crps']:6.2f} "
        f"cov80 {c['cov80']:.2f}"
    )


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


def _report(results: pd.DataFrame, folds: dict, modes) -> None:
    pd.set_option("display.width", 240)
    summary = {"min_peak_fp": MIN_PEAK_FP, "modes": {}}

    for mode in modes:
        r = results[results["mode"] == mode]
        if r.empty:
            continue
        print(f"\n{'=' * 78}\n=== mode: {mode} ===\n{'=' * 78}")

        print("\n--- RMSE per fold ---")
        print(
            r.pivot(index="predicts", columns="model", values="rmse")
            .round(2)
            .to_string()
        )

        print("\n--- averaged over folds ---")
        agg = (
            r.groupby("model")[
                ["n", "rmse", "mae", "crps", "cov50", "cov80", "cov90", "spearman", "bias"]
            ]
            .mean()
            .sort_values("crps")
        )
        print(agg.round(3).to_string())

        mf = {k: v for k, v in folds.items() if k[0] == mode}
        cal_slope = _pooled(mf)
        summary["modes"][mode] = {
            "per_fold_rmse": r.pivot(index="predicts", columns="model", values="rmse")
            .round(3)
            .to_dict(),
            "averaged": agg.round(4).to_dict(),
            "calibration_slope": cal_slope,
        }

    (OUT / "backtest_summary.json").write_text(json.dumps(summary, indent=2))

    ref_mode = "causal" if "causal" in modes else list(modes)[0]
    fig = diagnostics(
        {k[1]: v for k, v in folds.items() if k[0] == ref_mode}, OUT / "diagnostics.png"
    )
    print(f"\nwrote {OUT / 'backtest_results.csv'}, {OUT / 'backtest_folds.pkl'}, {fig}")


def _pooled(folds: dict) -> float:
    """Calibration and coverage over every fold's rows at once.

    Pooling gives one check with real sample size behind it rather than four
    noisy ones.
    """
    all_draws = np.concatenate([f["draws"] for f in folds.values()], axis=1)
    all_y = np.concatenate([f["rows"]["next_fp_ppr"].to_numpy() for f in folds.values()])
    all_games = np.concatenate([f["games"] for f in folds.values()], axis=1)
    all_y_games = np.concatenate(
        [f["rows"]["next_games"].to_numpy() for f in folds.values()]
    )

    print("\n--- interval coverage ---")
    print("  level   all players   given he plays")
    for lv in (0.5, 0.8, 0.9):
        lo, hi = np.quantile(all_draws, [(1 - lv) / 2, 1 - (1 - lv) / 2], axis=0)
        uncond = np.mean((all_y >= lo) & (all_y <= hi))
        cond = coverage_given_played(all_draws, all_games, all_y, all_y_games, lv)
        print(f"  {lv:5.0%}   {uncond:11.3f}   {cond:14.3f}")

    u = pit(all_draws, all_y)
    counts, _ = np.histogram(u, bins=10, range=(0, 1))
    print(f"\n--- PIT over all {len(all_y):,} pooled predictions ---")
    print("uniform would be", f"{len(all_y) / 10:.0f} per decile")
    print(" ".join(f"{c:5d}" for c in counts))

    pred = all_draws.mean(axis=0)
    bucket = pd.qcut(pred, 10, labels=False)
    cal = pd.DataFrame({"pred": pred, "actual": all_y, "b": bucket})
    print("\n--- calibration by projection decile ---")
    print(
        cal.groupby("b")
        .agg(n=("pred", "size"), mean_proj=("pred", "mean"), mean_actual=("actual", "mean"))
        .round(1)
        .to_string()
    )
    slope = np.polyfit(pred, all_y, 1)
    print(f"\nactual ~ a + b*projection:  b = {slope[0]:.3f} (1.0 = correctly shrunk)")
    return float(slope[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short chains, for iteration")
    ap.add_argument(
        "--no-inputs", action="store_true", help="drop the usage / late-form inputs"
    )
    ap.add_argument(
        "--modes",
        default="causal,window",
        help=f"comma-separated subset of {MODES}",
    )
    ap.add_argument(
        "--chains",
        default=None,
        help="warmup,samples,chains - overrides --quick. Tiny values smoke-test "
             "the code path without waiting for real chains.",
    )
    args = ap.parse_args()
    chosen = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    assert set(chosen) <= set(MODES), f"unknown mode in {chosen}"
    spec = tuple(int(x) for x in args.chains.split(",")) if args.chains else None
    run(
        quick=args.quick,
        use_inputs=not args.no_inputs,
        modes=chosen,
        chains_spec=spec,
    )
