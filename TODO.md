# TODO

## Complete data pull for 2021–2025 seasons

**Status:** not started — blocking better model results.

### The problem

The database currently backfills prior seasons using only the **2025 top-200 rankings**,
so historical coverage is a survivorship-biased sample rather than the real league:

| season | players in `player_games` |
|--------|---------------------------|
| 2021   | 81  |
| 2022   | 114 |
| 2023   | 144 |
| 2024   | 170 |
| 2025   | 543 (full sync) |

Verified: all 170 players with 2024 rows appear in the `rankings` top-200 list — the
overlap is exact.

Two consequences, both confirmed in `catboost_starter.ipynb`:

1. **Cross-season training fails.** 2022–2024 average ~12 PPR vs 2025's 6.8, because
   the older seasons only contain players good enough to still be around in 2025.
   Training on 2022–2024 and testing on 2025 over-predicted by **3.2 pts/game** on
   every position, and lost to a naive baseline (RMSE 7.463 vs 6.938).

2. **A shortcut feature dominates.** 59.5% of 2025 rows have no prior-season row, so
   all their `prev_*` features are zero — and that group averages 2.95 PPR vs 12.01
   for players with history. But **91.4% of those "cold" players are 2+ year
   veterans** (Tyrod Taylor, James Conner, Carson Wentz…) who did play in 2024; the
   DB just lacks their rows. The model is partly detecting which players the database
   recorded, not football signal.

### The fix

Pull **all players who scored > 0 points in that season**, enumerated per season —
not tied to any top-N list, and not today's roster filtered backward (that reproduces
the same bias one layer down; players who retired before 2025 would still be missing).

**Filter at the season level, not the game level.** 1,683 of 6,197 2025 games (27%)
score exactly zero. Dropping those individually filters on the target and pushes the
mean from 6.78 → 9.46 — the model would never learn what a bust looks like. Only 62
of 543 players scored nothing across all of 2025, so the season-level filter is cheap.

```sql
-- include a player-season if the player scored at all that year,
-- then keep ALL their games, zeros included
WITH scorers AS (
  SELECT athlete_id, season
  FROM player_games
  WHERE season_type = 2
  GROUP BY athlete_id, season
  HAVING SUM(fp_ppr) > 0
)
SELECT g.*
FROM player_games g
JOIN scorers s
  ON s.athlete_id = g.athlete_id AND s.season = g.season
```

### After the pull

- Re-check whether `prev_games == 0` still separates the target. If the 9-point gap
  collapses, the artifact is gone.
- Switch the split back to multi-season (train 2022–2024, test 2025). Expected
  ~25–30k rows vs the 3,735 the current within-2025 split allows.
- **Expect the metrics to look worse at first.** Losing the shortcut means the model
  has to separate players on real statistical differences. Compare against the
  `prev_fp_ppr_avg` passthrough baseline recomputed on the new data — not against the
  current 6.413 test RMSE.

### Not solved by this

Genuine rookies have no prior season by definition (~6% of players). Needs a separate
input: draft position, college production, or a positional prior.
