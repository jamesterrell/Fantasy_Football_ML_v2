"""Figures for the Bayesian projection model.

Three questions, three figures: is the forecast distribution honest
(``diagnostics``), what did the model actually learn (``model_structure``), and
what does it say about next season (``projection_fan``).

Colours come from a validated categorical palette. Position series appear only
on the line chart, where adjacent-pair separation is the relevant gate; the
scatter is faceted by position rather than coloured by it, because an
all-pairs comparison would not clear the colour-vision floors at four series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bayes.data import SEASON_GAMES  # noqa: E402
from bayes.metrics import coverage_given_played, pit  # noqa: E402
from bayes.production import POSITIONS  # noqa: E402
from bayes.spline import apply_spline_basis  # noqa: E402

# Categorical slots 1-4 of the reference palette, in their validated order.
SERIES = {"QB": "#2a78d6", "RB": "#eb6834", "TE": "#1baf7a", "WR": "#eda100"}
BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"
BLUE_DARK = "#184f95"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e5e4e0"


def _style(ax, ygrid: bool = True):
    """Recessive axes: the data should be the only assertive thing on screen."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3, width=0.8)
    if ygrid:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_SOFT)
    ax.yaxis.label.set_color(INK_SOFT)


def diagnostics(folds: dict, path: Path) -> Path:
    """Calibration, PIT and interval coverage, pooled over every backtest fold."""
    draws = np.concatenate([f["draws"] for f in folds.values()], axis=1)
    y = np.concatenate([f["rows"]["next_fp_ppr"].to_numpy() for f in folds.values()])
    pred = draws.mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    # --- 1. does a projection of N points mean N points?
    ax = axes[0]
    b = pd.qcut(pred, 10, labels=False)
    cal = pd.DataFrame({"p": pred, "a": y, "b": b}).groupby("b").mean()
    lim = [0, max(cal["p"].max(), cal["a"].max()) * 1.08]
    ax.plot(lim, lim, color=INK_SOFT, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.plot(cal["p"], cal["a"], color=BLUE, linewidth=2, zorder=2)
    ax.scatter(cal["p"], cal["a"], s=42, color=BLUE, zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("mean projection (PPR points)")
    ax.set_ylabel("mean actual")
    ax.set_title("Calibration by projection decile", fontsize=11, loc="left", pad=10)
    ax.annotate("perfect calibration", xy=(lim[1] * 0.62, lim[1] * 0.62),
                xytext=(lim[1] * 0.66, lim[1] * 0.44), fontsize=8.5, color=INK_SOFT,
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, linewidth=0.8))
    _style(ax)

    # --- 2. PIT: uniform if the whole distribution is right, not just its middle
    ax = axes[1]
    u = pit(draws, y)
    counts, edges = np.histogram(u, bins=10, range=(0, 1))
    ax.bar(edges[:-1], counts, width=0.092, align="edge", color=BLUE_LIGHT,
           edgecolor=BLUE, linewidth=1)
    ax.axhline(len(y) / 10, color=INK_SOFT, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xlabel("PIT value")
    ax.set_ylabel("count")
    ax.set_title("Probability integral transform", fontsize=11, loc="left", pad=10)
    ax.text(0.02, len(y) / 10 * 1.06, "uniform", fontsize=8.5, color=INK_SOFT)
    _style(ax)

    # --- 3. do the stated intervals contain what they claim to?
    #
    # Two curves, because they answer different questions. The unconditional
    # predictive has an atom at zero for the quarter of players who leave the
    # league, and a central interval spanning that gap overshoots its nominal
    # level even for a perfectly calibrated model. Conditioning both sides on
    # playing isolates the scoring model from the dropout model.
    ax = axes[2]
    games = np.concatenate([f["games"] for f in folds.values()], axis=1)
    y_games = np.concatenate(
        [f["rows"]["next_games"].to_numpy() for f in folds.values()]
    )
    levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    cov = [
        np.mean(
            (y >= np.quantile(draws, (1 - lv) / 2, axis=0))
            & (y <= np.quantile(draws, 1 - (1 - lv) / 2, axis=0))
        )
        for lv in levels
    ]
    cov_played = [
        coverage_given_played(draws, games, y, y_games, lv) for lv in levels
    ]
    ax.plot([0.45, 1.0], [0.45, 1.0], color=INK_SOFT, linewidth=1,
            linestyle=(0, (4, 3)))
    for series, color, label in [
        (cov_played, "#2a78d6", "given he plays"),
        (cov, "#eb6834", "all players"),
    ]:
        ax.plot(levels, series, color=color, linewidth=2, label=label)
        ax.scatter(levels, series, s=38, color=color, edgecolor="white",
                   linewidth=1.5, zorder=3)
    ax.set_xlabel("nominal interval")
    ax.set_ylabel("actual coverage")
    ax.set_title("Interval coverage", fontsize=11, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="upper left")
    _style(ax)

    fig.suptitle(
        f"Forecast honesty — {len(y):,} out-of-sample player-seasons",
        fontsize=13, color=INK, x=0.006, ha="left", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def model_structure(
    posterior: dict, spline_spec: dict, path: Path, full_season_zvar: float = 0.065
) -> Path:
    """The aging curve and the variance decomposition the model inferred.

    ``full_season_zvar`` is the panel's median measurement variance for a
    17-game season, which is what puts the "luck" share on the same footing as
    the two ability components.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- aging curves, on the interpretable points-per-game scale
    ax = axes[0]
    ages = np.linspace(22, 36, 120)
    basis = apply_spline_basis(ages, spline_spec)
    for pos_i, pos in enumerate(POSITIONS):
        # One curve per posterior draw, then summarised - so the band is the
        # posterior of the curve rather than a band drawn around a point fit.
        z = posterior["base"][:, pos_i][:, None] + np.einsum(
            "ak,dk->da", basis, posterior["beta_age"][:, pos_i, :]
        )
        ppg = np.maximum(z, 0) ** 2
        med = np.median(ppg, axis=0)
        lo, hi = np.percentile(ppg, [10, 90], axis=0)
        ax.fill_between(ages, lo, hi, color=SERIES[pos], alpha=0.13, linewidth=0)
        ax.plot(ages, med, color=SERIES[pos], linewidth=2, label=pos)
        ax.text(ages[-1] + 0.15, med[-1], pos, color=SERIES[pos], fontsize=9.5,
                va="center", fontweight="bold")
    ax.set_xlim(22, 37.2)
    ax.set_xlabel("age at Sept 1")
    ax.set_ylabel("PPR points per game, full season")
    ax.set_title("Aging curve for an average player at each position",
                 fontsize=11, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncols=4,
              loc="upper right")
    _style(ax)

    # --- where the variation in a season actually comes from
    ax = axes[1]
    rows = []
    for i, pos in enumerate(POSITIONS):
        perm = (posterior["sigma_u"][:, i] ** 2).mean()
        rho, sig = posterior["rho"][:, i], posterior["sigma"][:, i]
        trans = (sig ** 2 / np.maximum(1 - rho ** 2, 1e-6)).mean()
        # Game-to-game luck over a typical full season, on the same scale.
        noise = (posterior["kappa"][:, i]).mean() * full_season_zvar
        rows.append((pos, perm, trans, noise))
    d = pd.DataFrame(rows, columns=["pos", "permanent", "transient", "noise"])
    share = d[["permanent", "transient", "noise"]].div(
        d[["permanent", "transient", "noise"]].sum(axis=1), axis=0
    )
    left = np.zeros(len(d))
    parts = [("permanent", BLUE_DARK), ("transient", BLUE), ("noise", BLUE_LIGHT)]
    ypos = np.arange(len(d))
    for name, color in parts:
        ax.barh(ypos, share[name], left=left, color=color, height=0.6, label=name)
        for j, (v, l) in enumerate(zip(share[name], left)):
            if v > 0.09:
                ax.text(l + v / 2, j, f"{v:.0%}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if color != BLUE_LIGHT else INK)
        left = left + share[name].to_numpy()
    ax.set_yticks(ypos, d["pos"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of variance in a player's season")
    ax.set_title("Career-permanent skill vs form vs luck", fontsize=11,
                 loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncols=3,
              loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax.invert_yaxis()
    _style(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def projection_fan(table: pd.DataFrame, target: int, path: Path, top: int = 30) -> Path:
    """Top projections shown as intervals, because the interval is the point."""
    d = table.head(top).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 0.32 * top + 1.6))

    y = np.arange(len(d))
    ax.hlines(y, d["p05"], d["p95"], color=BLUE_LIGHT, linewidth=3.2,
              capstyle="round")
    ax.hlines(y, d["p25"], d["p75"], color=BLUE, linewidth=3.2, capstyle="round")
    ax.scatter(d["proj_median"], y, s=26, color=BLUE_DARK, zorder=3,
               edgecolor="white", linewidth=1.2)

    labels = [
        f"{n}  ({p})" for n, p in zip(d["display_name"], d["pos"])
    ]
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlabel(f"projected {target} PPR points")
    ax.set_title(
        f"{target} projections — median, 50% and 90% predictive intervals",
        fontsize=12, loc="left", pad=12,
    )
    for i, r in d.iterrows():
        ax.text(r["p95"] + 4, i, f"{r['proj_median']:.0f}", fontsize=8.5,
                color=INK_SOFT, va="center")
    _style(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path
