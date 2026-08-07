"""Fit on every season available and project the next one.

Produces the actual product: for each player who appeared in the most recent
season, a full predictive distribution over his PPR total for the season after
it. The point projection is one summary of that distribution; the interval, the
chance he misses the year, and the chance he finishes as a positional starter
are others, and for drafting purposes they matter at least as much.

Run:  python run_projection.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from bayes.data import LAST_SEASON, attach_next_season, build_panel
from bayes.figures import model_structure, projection_fan
from bayes.predictive import fit_fold

OUT = Path(__file__).parent / "artifacts"

# Roughly how many at each position start in a 12-team league. Used to report
# the probability a player finishes the season as a startable asset, which is a
# different question from his expected points and often the more useful one.
STARTER_DEPTH = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}


def positional_rank_probs(draws: np.ndarray, pos: np.ndarray) -> pd.DataFrame:
    """P(finishes top-N at his position) and expected positional finish.

    Computed inside each posterior draw, so it accounts for the fact that a
    player only reaches the top 12 by outscoring the other players in the same
    draw of the world - which is not the same as comparing everyone's marginal
    means.
    """
    out = {"p_starter": np.zeros(draws.shape[1]), "exp_pos_rank": np.zeros(draws.shape[1])}
    for p in np.unique(pos):
        mask = pos == p
        sub = draws[:, mask]
        # Rank within position within each draw; 1 is the top scorer.
        rank = sub.shape[1] - sub.argsort(axis=1).argsort(axis=1)
        depth = STARTER_DEPTH.get(p, 24)
        out["p_starter"][mask] = (rank <= depth).mean(axis=0)
        out["exp_pos_rank"][mask] = rank.mean(axis=0)
    return pd.DataFrame(out)


def run(quick: bool = False) -> pd.DataFrame:
    OUT.mkdir(exist_ok=True)
    panel = attach_next_season(build_panel())
    target = LAST_SEASON + 1

    warmup, samples, chains = (300, 400, 2) if quick else (1000, 1000, 4)
    proj = fit_fold(
        panel,
        cutoff=LAST_SEASON,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        progress=True,
    )

    print("\n=== production model ===")
    proj.production_mcmc.print_summary(exclude_deterministic=True)
    print("\n=== availability model ===")
    proj.availability_mcmc.print_summary(exclude_deterministic=True)

    # Rank probabilities are computed on the draws in their original row order,
    # then attached by athlete_id - summary() sorts by projection, and lining
    # the two up positionally would silently give every player someone else's
    # numbers.
    ranks = positional_rank_probs(proj.draws, proj.rows["pos"].to_numpy())
    ranks["athlete_id"] = proj.rows["athlete_id"].to_numpy()

    table = proj.summary().merge(ranks, on="athlete_id", how="left")

    cols = [
        "display_name", "pos", "age", "games", "fp_ppr", "proj_mean", "proj_median",
        "p05", "p25", "p75", "p95", "exp_games", "p_misses_season", "proj_ppg",
        "p_starter", "exp_pos_rank",
    ]
    table = table[cols]

    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.to_csv(OUT / f"projections_{target}.csv", index=False)
    np.savez_compressed(
        OUT / f"projection_draws_{target}.npz",
        draws=proj.draws,
        games=proj.games_draws,
        athlete_id=proj.rows["athlete_id"].to_numpy(),
    )
    post = {k: np.asarray(v) for k, v in proj.production_mcmc.get_samples().items()}
    with open(OUT / "production_posterior.pkl", "wb") as fh:
        pickle.dump({"posterior": post, "spline_spec": proj.spline_spec}, fh)

    figs = [
        model_structure(
            post,
            proj.spline_spec,
            OUT / "model_structure.png",
            full_season_zvar=float(
                panel.loc[panel["games"] >= 16, "z_var"].median()
            ),
        ),
        projection_fan(table, target, OUT / f"projections_{target}.png"),
    ]

    pd.set_option("display.width", 250)
    print(f"\n=== top 30 projections for {target} ===")
    print(
        table.head(30).to_string(
            index=False,
            formatters={
                **{
                    c: "{:.1f}".format
                    for c in ["age", "fp_ppr", "proj_mean", "proj_median", "p05",
                              "p25", "p75", "p95", "exp_games", "proj_ppg",
                              "exp_pos_rank"]
                },
                **{c: "{:.2f}".format for c in ["p_misses_season", "p_starter"]},
            },
        )
    )

    print(f"\nwrote {OUT / f'projections_{target}.csv'}")
    for f in figs:
        print(f"wrote {f}")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(quick=ap.parse_args().quick)
