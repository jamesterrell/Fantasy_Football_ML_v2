"""Single-fold backtest of the Bayesian projection model.

The fold trains on every season through ``LAST_SEASON - 1`` and is scored on
what actually happened in ``LAST_SEASON``. That is the same information set the
projection run works from, one year earlier: standing in the August before
``LAST_SEASON``, every season in the training window had already been played.

Nothing is ever trained on a season later than the one it predicts, and the
split is by season rather than at random - a player's 2022 and 2023 rows share
most of their signal, so a random split would let the model see a version of the
answer.

**One fold, not three.** Earlier cutoffs were dropped: each one costs a full
NUTS fit, and a fold that predicts 2023 from two seasons of history is testing a
model trained on a third of the data the real projection uses. The season being
drafted for is the season worth being right about.

**No peak filter.** The model is fit and scored on every player in the panel.
Restricting the fit to players with a 50-point season behind them was measured
and it was worse - 1.4 RMSE worse, and it pushed bias from +1.1 to +10.2 points
on the very players the filter kept. Cutting the bottom off the panel moves the
position baselines and the shrinkage target up with it, and it removes the low
end that identifies the dropout cliff near replacement level, which is what the
skill-squared term in the availability model exists to fit.

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

from bayes.data import LAST_SEASON, SEASON_GAMES, attach_next_season, build_panel
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

# The fold. One cutoff, the latest one that leaves a season to be scored
# against.
CUTOFF = LAST_SEASON - 1


def run(
    quick: bool = False,
    use_inputs: bool = True,
    chains_spec: tuple[int, int, int] | None = None,
) -> pd.DataFrame:
    OUT.mkdir(exist_ok=True)
    panel = attach_next_season(build_panel())

    warmup, samples, chains = chains_spec or ((300, 400, 2) if quick else (800, 1000, 4))

    t0 = time.time()
    proj = fit_fold(
        panel,
        cutoff=CUTOFF,
        use_inputs=use_inputs,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        progress=False,
    )
    print(
        f"[{CUTOFF + 1}] fit {time.time() - t0:.0f}s "
        f"({panel['athlete_id'].nunique()} players, {len(panel)} rows, "
        f"{len(proj.rows)} projected)"
    )

    records = _score(proj)
    results = pd.DataFrame(records)
    results.to_csv(OUT / "backtest_results.csv", index=False)

    # The fold stored here is the model that ships. run_projection.py fits the
    # same configuration one season later, so the diagnostics below describe the
    # thing actually producing the board - which was not true while this file
    # fit two arms and saved the one it had just argued against.
    folds = {
        CUTOFF: {
            "rows": proj.rows,
            "draws": proj.draws,
            "games": proj.games_draws,
            "production_summary": _param_summary(proj.production_mcmc),
        }
    }
    with open(OUT / "backtest_folds.pkl", "wb") as fh:
        pickle.dump(folds, fh)

    _report(results, folds)
    return results


def _score(proj) -> list[dict]:
    """Scorecard for the model and the reference arms, on every projected row."""
    rows = proj.rows
    y = rows["next_fp_ppr"].to_numpy()

    # A backtest with no labels is not a bad backtest, it is not a backtest.
    # Worth an explicit check because the failure is silent downstream: every
    # metric comes back NaN, `next_games > 0` is False for every row because
    # NaN comparisons are, and the first thing to actually raise is a quantile
    # over an empty slice, thirty lines and one full fit later.
    missing = np.isnan(y).sum()
    if missing:
        raise ValueError(
            f"{missing} of {len(y)} scored rows have no next-season label. "
            "The caller must attach labels while it can still see the season "
            "after the cutoff - see the fill-don't-rebuild note in fit_fold."
        )

    entries = {
        "bayes": summarise(proj.draws, y),
        "repeat last season": summarise_point(rows["fp_ppr"].to_numpy(), y),
        f"last season ppg x {SEASON_GAMES}": summarise_point(
            rows["ppg"].to_numpy() * SEASON_GAMES, y
        ),
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
    cond, _ = conditional_on_playing(proj.draws, proj.games_draws)
    entries["bayes | in the league"] = summarise(cond[:, active], y[active])
    entries["repeat last season | in the league"] = summarise_point(
        rows["fp_ppr"].to_numpy()[active], y[active]
    )

    return [
        {"predicts": CUTOFF + 1, "model": name, **m} for name, m in entries.items()
    ]


def _param_summary(mcmc) -> pd.DataFrame:
    """Posterior mean and sd for every parameter, at any dimensionality.

    Written generically on purpose. The earlier version handled only scalar and
    vector parameters, so when `rho` and `sigma` gained an established-player
    axis they became 3-D and dropped out of the summary silently - leaving a
    forty-minute backtest that could not answer whether the new split had
    identified, which was the entire question it was run to settle.
    """
    post = mcmc.get_samples()
    rows = {}
    for name, values in post.items():
        v = np.asarray(values)
        if v.ndim == 1:
            rows[name] = (v.mean(), v.std())
            continue
        # Flatten every trailing axis into one index label, e.g. rho[2,1].
        flat = v.reshape(v.shape[0], -1)
        for j, idx in enumerate(np.ndindex(*v.shape[1:])):
            label = ",".join(str(i) for i in idx)
            rows[f"{name}[{label}]"] = (flat[:, j].mean(), flat[:, j].std())
    return pd.DataFrame(rows, index=["mean", "sd"]).T


def _report(results: pd.DataFrame, folds: dict) -> None:
    pd.set_option("display.width", 240)

    print(f"\n{'=' * 78}\n=== trained through {CUTOFF}, scored on {CUTOFF + 1} ===\n{'=' * 78}")

    cols = ["n", "rmse", "mae", "crps", "cov50", "cov80", "cov90", "spearman", "bias"]
    table = results.set_index("model")[cols].sort_values("crps")
    print("\n--- scorecard ---")
    print(table.round(3).to_string())

    cal_slope = _pooled(folds)

    summary = {
        "cutoff": CUTOFF,
        "predicts": CUTOFF + 1,
        "peak_filter": None,
        "scorecard": table.round(4).to_dict(),
        "calibration_slope": cal_slope,
    }
    (OUT / "backtest_summary.json").write_text(json.dumps(summary, indent=2))

    fig = diagnostics(folds, OUT / "diagnostics.png")
    print(f"\nwrote {OUT / 'backtest_results.csv'}, {OUT / 'backtest_folds.pkl'}, {fig}")


def _pooled(folds: dict) -> float:
    """Calibration and coverage over the fold's rows.

    Kept as a pooled helper over ``folds`` rather than folded into ``_score`` so
    the shape of the report does not change if a second cutoff is ever added
    back.
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
        "--chains",
        default=None,
        help="warmup,samples,chains - overrides --quick. Tiny values smoke-test "
             "the code path without waiting for real chains.",
    )
    args = ap.parse_args()
    spec = tuple(int(x) for x in args.chains.split(",")) if args.chains else None
    run(
        quick=args.quick,
        use_inputs=not args.no_inputs,
        chains_spec=spec,
    )
