"""Where is the predictive too wide?

Backtest coverage has run consistently above nominal (about 0.63 at the 50%
level), which means the predictive distribution is over-dispersed somewhere.
There are three candidate sources - the games distribution, the final
within-season luck layer, and the ability posterior - and they are separable,
because the games half can be checked against realised games on its own.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from bayes.metrics import interval_coverage, pit

pd.set_option("display.width", 240)

with open("artifacts/backtest_folds.pkl", "rb") as fh:
    folds = pickle.load(fh)

rows = pd.concat([f["rows"] for f in folds.values()], ignore_index=True)
draws = np.concatenate([f["draws"] for f in folds.values()], axis=1)
games = np.concatenate([f["games"] for f in folds.values()], axis=1)

y_pts = rows["next_fp_ppr"].to_numpy()
y_games = rows["next_games"].to_numpy()

print("=== games distribution, on its own ===")
for lv in (0.5, 0.8, 0.9):
    print(f"  nominal {lv:.0%}  actual {interval_coverage(games, y_games, lv):.3f}")
print(f"  predicted mean games {games.mean():.2f}  actual {y_games.mean():.2f}")
print(f"  predicted sd         {games.std(axis=0).mean():.2f}  "
      f"actual resid sd {np.std(y_games - games.mean(axis=0)):.2f}")
u = pit(games, y_games)
print("  PIT deciles:", " ".join(f"{c:4d}" for c in np.histogram(u, 10, (0, 1))[0]))

print("\n=== P(zero games): predicted vs actual, by predicted probability ===")
p0 = (games == 0).mean(axis=0)
b = pd.qcut(p0, 8, labels=False, duplicates="drop")
print(
    pd.DataFrame({"p0": p0, "actual": (y_games == 0).astype(float), "b": b})
    .groupby("b")
    .agg(n=("p0", "size"), predicted=("p0", "mean"), actual=("actual", "mean"))
    .round(3)
    .to_string()
)

print("\n=== points, conditional on the player actually playing ===")
played = y_games > 0
# Compare only draws where the simulated player also played, so the zero atom
# does not dominate the comparison.
cond = np.where(games > 0, draws, np.nan)
for lv in (0.5, 0.8, 0.9):
    lo, hi = np.nanquantile(cond[:, played], [(1 - lv) / 2, 1 - (1 - lv) / 2], axis=0)
    inside = (y_pts[played] >= lo) & (y_pts[played] <= hi)
    print(f"  nominal {lv:.0%}  actual {inside.mean():.3f}")

print("\n=== spread of the points predictive vs realised spread ===")
pred_sd = draws.std(axis=0)
resid = y_pts - draws.mean(axis=0)
for lo, hi in [(0, 40), (40, 90), (90, 160), (160, 400)]:
    m = (draws.mean(axis=0) >= lo) & (draws.mean(axis=0) < hi)
    print(
        f"  proj {lo:3d}-{hi:3d}  n={m.sum():4d}  "
        f"predictive sd {pred_sd[m].mean():6.1f}  realised |resid| sd {resid[m].std():6.1f}"
    )
