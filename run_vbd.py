"""Value-based draft board: points above replacement, not raw points.

Raw projected points do not compare across positions, which is why the plain
board reads as twenty quarterbacks. A quarterback at 250 and a back at 250 are
not the same asset: you start one quarterback and the tenth-best one also scores
close to 250, while the twentieth-best back scores far less. What a pick is
worth is the gap between the player and the man you would otherwise have had at
that position - his replacement.

Replacement level is set by how many at each position actually get started in
*this* league, flex included. The flex is allocated greedily: ten times over,
whichever of RB/WR/TE has the best player still undrafted takes the next slot.
That is what a room full of drafters does in aggregate, and it moves the
baselines - a deep receiver year pushes the flex toward receivers and lowers the
RB baseline, which raises every back's value.

Three value columns, because they answer different questions:

  vbd       on the mean projection - the standard measure
  vbd_floor on the 25th percentile - what he is worth if things go badly, which
            is the number to weight when you need a weekly starter
  vbd_ceil  on the 75th percentile - the swing-for-the-fences view

A player whose `vbd` and `vbd_floor` ranks disagree sharply is exactly the
boom/bust case the mean hides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "artifacts"
FLEX_ELIGIBLE = ("RB", "WR", "TE")


def starter_counts(board: pd.DataFrame, teams: int, slots: dict[str, int],
                   flex: int, value_col: str) -> dict[str, int]:
    """How many at each position are startable in this league, flex included."""
    counts = {pos: teams * n for pos, n in slots.items()}
    ranked = {
        pos: board.loc[board["pos"] == pos, value_col]
        .sort_values(ascending=False).to_numpy()
        for pos in slots
    }

    for _ in range(teams * flex):
        # The best player still on the board at each flex-eligible position;
        # -inf where that position is exhausted so it stops being chosen.
        best, best_pos = -np.inf, None
        for pos in FLEX_ELIGIBLE:
            i = counts.get(pos, 0)
            if pos in ranked and i < len(ranked[pos]) and ranked[pos][i] > best:
                best, best_pos = ranked[pos][i], pos
        if best_pos is None:
            break
        counts[best_pos] += 1
    return counts


def replacement_levels(board: pd.DataFrame, counts: dict[str, int],
                       value_col: str) -> dict[str, float]:
    """The projection of the first player at each position nobody starts.

    Falls back to the worst player present when a position runs out, rather
    than to zero: a zero baseline would silently inflate every player at a thin
    position by the whole of his projection.
    """
    levels = {}
    for pos, n in counts.items():
        vals = (board.loc[board["pos"] == pos, value_col]
                .sort_values(ascending=False).to_numpy())
        if len(vals) == 0:
            levels[pos] = 0.0
        else:
            levels[pos] = float(vals[min(n, len(vals) - 1)])
    return levels


SEASON = [2026]  # set from the CLI; the draws file is named for it


def build(board: pd.DataFrame, teams: int, slots: dict[str, int], flex: int) -> pd.DataFrame:
    out = board.copy()
    for label, col in (("vbd", "proj_mean"), ("vbd_floor", "p25"), ("vbd_ceil", "p75")):
        if col not in out.columns:
            continue
        counts = starter_counts(out, teams, slots, flex, col)
        levels = replacement_levels(out, counts, col)
        out[label] = out[col] - out["pos"].map(levels)
        if label == "vbd":
            out.attrs["counts"], out.attrs["levels"] = counts, levels

    # Probabilities read straight off the posterior draws, not inferred from the
    # summary columns. `proj_mean` is not the fifty-fifty point - the predictive
    # is right-skewed for most players and has an atom at zero, so across the
    # top 300 the actual beats the mean only 44% of the time, and for
    # quarterbacks, whose downside is severe and whose ceiling is capped, it is
    # 57%. `proj_median` is the fifty-fifty point by construction.
    draws_path = OUT / f"projection_draws_{SEASON[0]}.npz"
    if draws_path.exists():
        z = np.load(draws_path, allow_pickle=True)
        pos_of = dict(zip(z["athlete_id"].astype(str), range(len(z["athlete_id"]))))
        d = z["draws"]
        have = out["athlete_id"].astype(str).map(pos_of) if "athlete_id" in out.columns else None
        if have is not None and have.notna().any():
            idx = have.fillna(-1).astype(int).to_numpy()
            lvl = out["pos"].map(levels).to_numpy(float)
            beat = np.full(len(out), np.nan)
            ok = idx >= 0
            # Does this pick clear the player you could have had for nothing?
            beat[ok] = (d[:, idx[ok]] > lvl[ok][None, :]).mean(axis=0)
            out["p_beat_replacement"] = beat
            # How often he is a top-5 asset at his position - the upside a pick
            # in the first few rounds is actually being bought for.
            top5 = np.full(len(out), np.nan)
            for pos_name in out["pos"].dropna().unique():
                m = (out["pos"] == pos_name).to_numpy() & ok
                if not m.any():
                    continue
                sub = d[:, idx[m]]
                rank = sub.shape[1] - sub.argsort(axis=1).argsort(axis=1)
                top5[m] = (rank <= 5).mean(axis=0)
            out["p_top5_pos"] = top5

    # The middle half of outcomes. Wider than a comparable player's means the
    # same expected points bought with more variance, which is the distinction
    # a single projection number hides.
    if {"p25", "p75"}.issubset(out.columns):
        out["range50"] = out["p75"] - out["p25"]

    out["pos_rank"] = out.groupby("pos")["proj_mean"].rank(ascending=False).astype(int)
    out = out.sort_values("vbd", ascending=False).reset_index(drop=True)
    out.insert(0, "vbd_rank", np.arange(1, len(out) + 1))
    # How far a player's floor-based rank sits from his mean-based one. Positive
    # means safer than his headline value suggests; negative means the value is
    # carried by upside.
    out["floor_rank"] = out["vbd_floor"].rank(ascending=False).astype(int)
    out["risk_shift"] = out["vbd_rank"] - out["floor_rank"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=int, default=10)
    ap.add_argument("--dir", default=None,
                    help="artifacts directory to read and write; "
                         "lets a candidate board be built without "
                         "touching the one in use")
    ap.add_argument("--qb", type=int, default=1)
    ap.add_argument("--rb", type=int, default=2)
    ap.add_argument("--wr", type=int, default=3)
    ap.add_argument("--te", type=int, default=1)
    ap.add_argument("--flex", type=int, default=1)
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()
    if a.dir:
        OUT = Path(a.dir)
        OUT.mkdir(parents=True, exist_ok=True)

    src = OUT / f"projections_{a.season}.csv"
    board = pd.read_csv(src)  # carries athlete_id, used to key the draws
    SEASON[0] = a.season
    slots = {"QB": a.qb, "RB": a.rb, "WR": a.wr, "TE": a.te}
    vbd = build(board, a.teams, slots, a.flex)

    counts, levels = vbd.attrs["counts"], vbd.attrs["levels"]
    print(f"{a.teams}-team league, starting "
          f"{a.qb}QB/{a.rb}RB/{a.wr}WR/{a.te}TE/{a.flex}FLEX\n")
    print("replacement level (the first player at each position nobody starts):")
    for pos in ("QB", "RB", "WR", "TE"):
        base = a.teams * slots[pos]
        extra = counts[pos] - base
        print(f"  {pos}: {pos}{counts[pos] + 1:<3d} at {levels[pos]:6.1f} pts"
              f"   ({base} starters"
              + (f" + {extra} to flex" if extra else "") + ")")

    cols = [c for c in ("vbd_rank", "display_name", "pos", "pos_rank", "team",
                        "status", "depth_rank", "age", "proj_mean", "proj_median",
                        "vbd", "vbd_floor", "vbd_ceil", "risk_shift",
                        "p_beat_replacement", "p_top5_pos", "p_starter",
                        "exp_games", "p_misses_season", "p25", "p75", "range50")
            if c in vbd.columns]
    dest = OUT / f"vbd_board_{a.season}_{a.teams}team.csv"
    vbd[cols].to_csv(dest, index=False)
    print(f"\nwrote {dest}  ({len(vbd)} players)")

    pd.set_option("display.width", 250)
    print(f"\n=== top 40 by value over replacement ===")
    print(vbd[cols].head(40).round(1).to_string(index=False))
