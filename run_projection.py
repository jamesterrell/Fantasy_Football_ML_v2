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

from bayes.data import (
    LAST_SEASON,
    MAX_MISSED_SEASONS,
    MIN_PEAK_FP,
    apply_peak_filter,
    attach_next_season,
    build_panel,
    qualifying_players,
    rostered_players,
)
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


def run(
    quick: bool = False,
    peak_filter: bool = True,
    train_universe: str = "all",
) -> pd.DataFrame:
    """Fit through ``LAST_SEASON`` and project the season after it.

    ``peak_filter`` restricts the *output table* to the draftable universe -
    players with a 50-point season behind them. ``train_universe`` controls
    whether the model is also *fit* on only those players, and the backtest says
    it should not be: training on the filtered panel cost 1.4 RMSE and pushed
    the projection bias from +1.1 to +10.2 points on the very players the filter
    keeps. Cutting the bottom off the panel moves the position baselines and the
    shrinkage target up with it, so everyone who survives gets over-projected -
    and it removes the low end that identifies the dropout cliff near
    replacement level, which is what the skill-squared term in the availability
    model exists to fit.

    So: filter what you read, not what the model learns from.
    """
    OUT.mkdir(exist_ok=True)
    panel = attach_next_season(build_panel())
    target = LAST_SEASON + 1

    # The draftable universe. Unlike a backtest fold this one is entitled to the
    # whole window: projecting 2026 from 2021-2025 uses only seasons that have
    # already happened.
    draftable = qualifying_players(panel) if peak_filter else None
    if peak_filter:
        print(
            f"draftable universe (>= {MIN_PEAK_FP:.0f} pts in a season, "
            f"2021-{LAST_SEASON}): {len(draftable):,} of "
            f"{panel['athlete_id'].nunique():,} players"
        )
    if train_universe == "filtered":
        panel = apply_peak_filter(panel)
        print(f"training on the filtered panel only: {len(panel):,} rows")

    # Players on a roster or unsigned in August, so that missing the whole of
    # LAST_SEASON does not silently remove a player from the board. He gets a
    # zero-game row for that season, a correctly widened interval, and a real
    # dropout probability - rather than no projection at all, which is the one
    # answer that is certainly wrong.
    roster = rostered_players()
    # Mirror the rule `add_missed_seasons` actually applies, rather than the
    # raw set difference: a player joins the board off a missed season only if
    # he has history to project from and has missed no more than
    # MAX_MISSED_SEASONS. The difference is not small - the raw count is 841,
    # because it sweeps in every 2026 rookie and everyone listed as a free
    # agent since 2020.
    last_played = panel.groupby("athlete_id")["season"].max()
    returning = {
        a for a in roster
        if a in last_played.index
        and 0 < LAST_SEASON - last_played[a] <= MAX_MISSED_SEASONS
    }
    print(
        f"rostered or free agent: {len(roster):,}; of these {len(returning):,} "
        f"missed {LAST_SEASON} entirely and are projected from a zero-game season"
    )

    warmup, samples, chains = (300, 400, 2) if quick else (1000, 1000, 4)
    proj = fit_fold(
        panel,
        cutoff=LAST_SEASON,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        progress=True,
        roster_ids=roster,
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

    # Drop the undraftable players from the output, after the fit rather than
    # before it. Positional rank probabilities are computed above on the full
    # field on purpose: a player reaches the top 24 by outscoring everyone at
    # his position, including the ones who are not worth drafting themselves.
    if draftable is not None:
        before = len(table)
        table = table[table["athlete_id"].isin(draftable)].reset_index(drop=True)
        print(f"\noutput restricted to draftable players: {len(table):,} of {before:,}")

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
    ap.add_argument(
        "--no-peak-filter",
        action="store_true",
        help="output the whole league, not just the draftable universe",
    )
    ap.add_argument(
        "--train-universe",
        choices=("all", "filtered"),
        default="all",
        help="fit on every player (default) or only the draftable ones. The "
             "backtest says 'all' is better even for the draftable players.",
    )
    args = ap.parse_args()
    run(
        quick=args.quick,
        peak_filter=not args.no_peak_filter,
        train_universe=args.train_universe,
    )
