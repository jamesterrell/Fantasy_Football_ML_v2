"""Expanding-window backtest of the Bayesian projection model.

Each fold trains on every season through its cutoff and is scored on what
actually happened the season after. That is the same information set the
projection run works from, some number of years earlier: standing in the August
before the scored season, every season in the training window had already been
played.

Nothing is ever trained on a season later than the one it predicts, and the
split is by season rather than at random - a player's 2022 and 2023 rows share
most of their signal, so a random split would let the model see a version of the
answer.

**Four folds, not one.** The single-fold design was argued for on the grounds
that a fold predicting 2023 trains on a third of the data the real projection
uses. That was true of a panel starting in 2020 and is not true of one starting
in 2016: the earliest fold here trains on five seasons (2016-2020) and predicts
2021. Four folds score ~2,100 rows instead of ~540, and - more to the point -
they show whether a change wins everywhere or wins on one season. A single fold
cannot distinguish those and this one had been mined.

**2025 is sealed.** The development folds stop at a 2024 outcome and the panel
handed to them is truncated at 2024, so the held-out season is not merely
unscored, it is absent. Spending it requires ``--final``, which says so loudly.

**No peak filter.** The model is fit and scored on every player in the panel.
Restricting the fit to players with a 50-point season behind them was measured
and it was worse - 1.4 RMSE worse, and it pushed bias from +1.1 to +10.2 points
on the very players the filter kept. Cutting the bottom off the panel moves the
position baselines and the shrinkage target up with it, and it removes the low
end that identifies the dropout cliff near replacement level, which is what the
skill-squared term in the availability model exists to fit.

Run:  python run_backtest.py [--quick] [--cutoffs 2020,2021,2022,2023]
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
    attach_next_season,
    build_panel,
    season_length,
)
from bayes.figures import diagnostics
from bayes.metrics import (
    TOP_K,
    TOP_KS,
    conditional_on_playing,
    coverage_given_played,
    crps_from_samples,
    pit,
    self_consistent_top,
    summarise,
    summarise_point,
    top_of_board,
)
from bayes.predictive import fit_fold

OUT = Path(__file__).parent / "artifacts"

# The development folds: cutoff -> scored season. 2020 is the earliest cutoff
# worth fitting, because a fold needs transitions to estimate rho and sigma from
# and 2016 alone has none; from 2020 it has four.
DEV_CUTOFFS = (2020, 2021, 2022, 2023)

# The held-out fold. Not in DEV_CUTOFFS on purpose - see the module docstring.
HELD_OUT_CUTOFF = LAST_SEASON - 1


def run(
    quick: bool = False,
    use_inputs: bool = True,
    chains_spec: tuple[int, int, int] | None = None,
    cutoffs: tuple[int, ...] = DEV_CUTOFFS,
    final: bool = False,
    inference_production: str = "nuts",
    inference_availability: str = "nuts",
    out_dir: Path | None = None,
) -> pd.DataFrame:
    # ``out_dir`` exists so an approximate-inference run can be scored beside the
    # NUTS run rather than on top of it. Overwriting the incumbent's artifacts is
    # how a comparison stops being possible.
    out = Path(out_dir) if out_dir is not None else OUT
    out.mkdir(parents=True, exist_ok=True)

    cutoffs = tuple(sorted(cutoffs))
    if final:
        cutoffs = cutoffs + (HELD_OUT_CUTOFF,)
        print(
            f"\n*** SPENDING THE HELD-OUT SEASON: this run scores "
            f"{HELD_OUT_CUTOFF + 1}. Nothing selected after this point is a "
            f"clean estimate. ***\n"
        )
    horizon = max(cutoffs) + 1
    if not final and horizon >= LAST_SEASON:
        raise ValueError(
            f"cutoff {max(cutoffs)} would score {horizon}, the held-out season. "
            "Pass --final if that is genuinely what you mean."
        )

    # Truncated *before* labels are attached, so the sealed season is not in the
    # frame at all rather than merely unscored. A season that is not in the frame
    # cannot leak through a groupby, a median, or a variance law.
    panel = build_panel()
    panel = attach_next_season(panel[panel["season"] <= horizon].copy())

    warmup, samples, chains = chains_spec or ((300, 400, 2) if quick else (800, 1000, 4))

    folds: dict[int, dict] = {}
    records: list[dict] = []
    timing: dict[int, dict] = {}
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
            inference_production=inference_production,
            inference_availability=inference_availability,
        )
        elapsed = time.time() - t0
        train = panel[panel["season"] <= cutoff]
        stages = " ".join(f"{k[:-2] if k.endswith('_s') else k}={v:.1f}s"
                          for k, v in proj.timings.items())
        print(
            f"[{cutoff + 1}] fit {elapsed:.0f}s [{stages}] "
            f"({train['athlete_id'].nunique()} players, {len(train)} rows, "
            f"{len(proj.rows)} projected)",
            flush=True,
        )
        diag = {
            half: getattr(m, "diagnostics", None)
            for half, m in (("production", proj.production_mcmc),
                            ("availability", proj.availability_mcmc))
        }
        for half, d in diag.items():
            if d:
                print(
                    f"    laplace/{half}: dim={d['dim']} "
                    f"logpost={d['logpost']:.1f} newton={d['newton_step']:.3f} "
                    f"eig_min={d['eig_min']:.3e} nonpos={d['n_nonpositive']} "
                    f"psis_khat={d.get('psis_khat', float('nan')):.2f} "
                    f"ess={d.get('weight_ess_count', float('nan')):.1f}"
                    f"/{d['n_draws']} "
                    f"max_w={d.get('max_weight', float('nan')):.3f} "
                    f"bad={d.get('n_nonfinite_draws', 0)}",
                    flush=True,
                )
        timing[cutoff] = {
            "total_s": elapsed, **proj.timings,
            "inference_production": inference_production,
            "inference_availability": inference_availability,
            "laplace_diagnostics": {k: v for k, v in diag.items() if v},
        }
        records += _score(proj, cutoff)
        folds[cutoff] = {
            "rows": proj.rows,
            "draws": proj.draws,
            "games": proj.games_draws,
            "production_summary": _param_summary(proj.production_mcmc),
        }

        # Flush after every fold rather than at the end. A four-fold run is
        # well over an hour of compute, and buffering it all until the last
        # line means any interruption - a crash in a later fold, a killed
        # process, a full disk - throws away every fold that had already
        # succeeded. That has happened twice on this project. Rewriting both
        # files each time costs a second or two against the fit that produced
        # them, and the partial artifacts are readable on their own.
        pd.DataFrame(records).to_csv(out / "backtest_results.csv", index=False)
        with open(out / "backtest_folds.pkl", "wb") as fh:
            pickle.dump(folds, fh)
        print(f"    checkpointed {len(folds)} fold(s) to {out}", flush=True)

    results = pd.DataFrame(records)

    _report(results, folds, final=final, out=out, timing=timing)
    return results


def _score(proj, cutoff: int) -> list[dict]:
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

    # The naive rate baseline projects onto the season being predicted, so it
    # multiplies by *that* season's length - 16 from 2021 on, 15 before. Using a
    # constant would hand a 2016-2020 fold a 6.7% inflation and call it a model.
    target_games = season_length(cutoff + 1)

    entries = {
        "bayes": summarise(proj.draws, y),
        "repeat last season": summarise_point(rows["fp_ppr"].to_numpy(), y),
        f"last season ppg x {target_games}": summarise_point(
            rows["ppg"].to_numpy() * target_games, y
        ),
        "position mean": summarise_point(
            rows.groupby("pos")["fp_ppr"].transform("mean").to_numpy(), y
        ),
    }
    # Every board size, for every arm. The suffix keeps them in one flat record
    # so the per-fold table can show k=24 and k=100 side by side; the aggregate
    # scorecard above is untouched, which is what makes a change that buys the
    # top by wrecking the rest still visible.
    point_arms = {
        "repeat last season": rows["fp_ppr"].to_numpy(),
        f"last season ppg x {target_games}": rows["ppg"].to_numpy() * target_games,
        "position mean": rows.groupby("pos")["fp_ppr"].transform("mean").to_numpy(),
    }
    tops: dict[str, dict] = {name: {} for name in ["bayes", *point_arms]}
    for k in TOP_KS:
        t = top_of_board(proj.draws.mean(axis=0), y, draws=proj.draws, k=k)
        t.update(self_consistent_top(proj.draws.mean(axis=0), proj.draws, k=k))
        tops["bayes"].update({f"{a}_k{k}": b for a, b in t.items()})
        for name, p in point_arms.items():
            tops[name].update(
                {f"{a}_k{k}": b for a, b in top_of_board(p, y, k=k).items()}
            )

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

    out = []
    for name, m in entries.items():
        top = {
            k: v for k, v in (tops.get(name) or {}).items() if "_idx" not in k
        }
        out.append({"predicts": cutoff + 1, "model": name, **m, **top})
    return out


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


def _report(
    results: pd.DataFrame,
    folds: dict,
    final: bool = False,
    out: Path | None = None,
    timing: dict | None = None,
) -> None:
    out = out if out is not None else OUT
    pd.set_option("display.width", 240)

    cols = ["n", "rmse", "mae", "crps", "cov50", "cov80", "cov90", "spearman", "bias"]
    for season, g in results.groupby("predicts"):
        print(f"\n{'=' * 78}\n=== trained through {season - 1}, scored on {season} ===\n{'=' * 78}")
        print(g.set_index("model")[cols].sort_values("crps").round(3).to_string())

    pooled = _pooled_scorecard(folds)
    print(f"\n{'=' * 78}\n=== pooled over {len(folds)} folds ===\n{'=' * 78}")
    print(pooled.set_index("model")[cols].sort_values("crps").round(3).to_string())

    top = _top_report(results, folds)
    cal_slope = _pooled(folds)

    summary = {
        "cutoffs": sorted(folds),
        "predicts": [c + 1 for c in sorted(folds)],
        "held_out_spent": bool(final),
        "peak_filter": None,
        "per_fold": results.round(4).to_dict(orient="records"),
        "pooled": pooled.round(4).to_dict(orient="records"),
        "calibration_slope": cal_slope,
        "top_of_board": top,
        "timing": timing or {},
    }
    (out / "backtest_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    try:
        fig = diagnostics(folds, out / "diagnostics.png")
    except Exception as exc:                                   # pragma: no cover
        fig = f"(diagnostics skipped: {exc})"
    print(f"\nwrote {out / 'backtest_results.csv'}, {out / 'backtest_folds.pkl'}, {fig}")


def _pooled_scorecard(folds: dict) -> pd.DataFrame:
    """Every fold's scored rows in one pile, scored once.

    Pooling the rows rather than averaging the per-fold numbers is what makes
    the pooled CRPS comparable to a per-fold one: folds differ in size, and a
    mean of means would silently reweight the small ones up.
    """
    rows = pd.concat([f["rows"] for f in folds.values()], ignore_index=True)
    draws = np.concatenate([f["draws"] for f in folds.values()], axis=1)
    games = np.concatenate([f["games"] for f in folds.values()], axis=1)
    y = rows["next_fp_ppr"].to_numpy()

    target_games = np.concatenate(
        [
            np.full(len(f["rows"]), season_length(c + 1), float)
            for c, f in folds.items()
        ]
    )
    entries = {
        "bayes": summarise(draws, y),
        "repeat last season": summarise_point(rows["fp_ppr"].to_numpy(), y),
        "last season ppg x season": summarise_point(
            rows["ppg"].to_numpy() * target_games, y
        ),
        "position mean": summarise_point(
            rows.groupby(["pos"])["fp_ppr"].transform("mean").to_numpy(), y
        ),
    }
    active = rows["next_games"].to_numpy() > 0
    cond, _ = conditional_on_playing(draws, games)
    entries["bayes | in the league"] = summarise(cond[:, active], y[active])
    entries["repeat last season | in the league"] = summarise_point(
        rows["fp_ppr"].to_numpy()[active], y[active]
    )
    return pd.DataFrame([{"model": k, **v} for k, v in entries.items()])


def _top_report(results: pd.DataFrame, folds: dict) -> dict:
    """The elite diagnostic, per fold and pooled, at every board size.

    Pooling here is over *per-fold* selections: the top k of each season, piled
    up. Taking the top k of the pooled frame instead would let one strong
    season supply most of the rows and turn a four-season diagnostic back into a
    one-season one.

    Every raw bias is printed next to the self-consistent reference for the same
    fold and the same k, because the raw number is monotone in dispersion rather
    than accuracy and means nothing alone. The column that carries the argument
    is ``signal`` = observed - self-consistent.
    """
    out: dict = {}
    for k in TOP_KS:
        print(f"\n{'=' * 78}\n=== top-{k} board (bias is projection - actual; "
              f"negative = under-projected) ===\n{'=' * 78}")
        show = ["predicts", "model"] + [
            f"{c}_k{k}" for c in (
                "proj_mean_proj", "proj_mean_actual", "proj_bias", "proj_bias_pct",
                "proj_crps", "actual_mean_proj", "actual_mean_actual", "actual_bias",
                "actual_bias_pct", "actual_crps", "actual_cov80",
            )
        ]
        tab = results[results["model"].isin(
            ["bayes", "repeat last season", "position mean"]
        )][show]
        tab.columns = [c.replace(f"_k{k}", "") for c in tab.columns]
        print(tab.round(2).to_string(index=False))

        print(f"\n--- top-{k} bayes: observed vs self-consistent, per fold ---")
        sc_tab = results[results["model"] == "bayes"][[
            "predicts", f"actual_bias_k{k}", f"sc_actual_bias_k{k}",
            f"proj_bias_k{k}", f"sc_proj_bias_k{k}",
        ]].copy()
        sc_tab["actual_signal"] = (
            sc_tab[f"actual_bias_k{k}"] - sc_tab[f"sc_actual_bias_k{k}"]
        )
        sc_tab["proj_signal"] = (
            sc_tab[f"proj_bias_k{k}"] - sc_tab[f"sc_proj_bias_k{k}"]
        )
        sc_tab.columns = [c.replace(f"_k{k}", "") for c in sc_tab.columns]
        print(sc_tab.round(2).to_string(index=False))

        # Pooled: the union of each fold's own top k.
        bags = {"by projection": ([], []), "by actual": ([], [])}
        sc = {"by projection": [], "by actual": []}
        for f in folds.values():
            y = f["rows"]["next_fp_ppr"].to_numpy()
            d = f["draws"]
            p = d.mean(axis=0)
            t = top_of_board(p, y, draws=d, k=k)
            s = self_consistent_top(p, d, k=k)
            sc["by projection"].append(s["sc_proj_bias"])
            sc["by actual"].append(s["sc_actual_bias"])
            for key, name in (("proj_idx", "by projection"), ("actual_idx", "by actual")):
                i = t[key]
                bags[name][0].append((p[i], y[i]))
                bags[name][1].append(d[:, i])
        for name, (bag, bagd) in bags.items():
            pred = np.concatenate([b[0] for b in bag])
            act = np.concatenate([b[1] for b in bag])
            dd = np.concatenate(bagd, axis=1)
            crps = float(np.mean(crps_from_samples(dd, act)))
            bias = float(pred.mean() - act.mean())
            ref = float(np.mean(sc[name]))
            out[f"top{k}_{name.replace(' ', '_')}"] = {
                "n": int(len(pred)), "bias": bias, "self_consistent": ref,
                "signal": bias - ref, "crps": crps,
                "mean_proj": float(pred.mean()), "mean_actual": float(act.mean()),
            }
            print(
                f"\npooled bayes, top {k} {name}: n={len(pred)}  "
                f"mean proj={pred.mean():.1f}  mean actual={act.mean():.1f}  "
                f"bias={bias:+.1f} "
                f"({100 * bias / act.mean():+.1f}%)  "
                f"self-consistent={ref:+.1f}  signal={bias - ref:+.1f}  "
                f"crps={crps:.1f}"
            )
    return out


def _pooled(folds: dict) -> float:
    """Calibration and coverage over every fold's rows."""
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
        "--cutoffs",
        default=None,
        help=f"comma-separated training cutoffs. Default {DEV_CUTOFFS}.",
    )
    ap.add_argument(
        "--final",
        action="store_true",
        help="also score the held-out season. Spends it - see the module docstring.",
    )
    ap.add_argument(
        "--chains",
        default=None,
        help="warmup,samples,chains - overrides --quick. Tiny values smoke-test "
             "the code path without waiting for real chains.",
    )
    ap.add_argument(
        "--inference",
        choices=("nuts", "laplace"),
        default="nuts",
        help="inference for both halves. NUTS is the incumbent; laplace is MAP "
             "+ a Gaussian at the mode - minutes instead of hours, at the cost "
             "of slightly narrow intervals. Read the coverage row.",
    )
    ap.add_argument(
        "--inference-production", choices=("nuts", "laplace"), default=None,
        help="override --inference for the production half only",
    )
    ap.add_argument(
        "--inference-availability", choices=("nuts", "laplace"), default=None,
        help="override --inference for the availability half only",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="artifact directory. Point an approximate run somewhere else so it "
             "does not overwrite the NUTS baseline it is being compared to.",
    )
    args = ap.parse_args()
    spec = tuple(int(x) for x in args.chains.split(",")) if args.chains else None
    run(
        quick=args.quick,
        use_inputs=not args.no_inputs,
        chains_spec=spec,
        cutoffs=tuple(int(x) for x in args.cutoffs.split(",")) if args.cutoffs
        else DEV_CUTOFFS,
        final=args.final,
        inference_production=args.inference_production or args.inference,
        inference_availability=args.inference_availability or args.inference,
        out_dir=args.out,
    )
