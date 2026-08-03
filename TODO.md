# TODO

## ~~Complete data pull for 2021–2025 seasons~~ — done 2026-08-02

Pulled via `python -m ffdb build-season --season 2020-2024` in the database project
(`ffdb/seasonpool.py`, branch `complete-season-backfill`). Seasons are now enumerated
from **box scores** rather than today's rosters: a player who recorded a stat in a
game is in that game's box score permanently, and fantasy points can only come from
passing, rushing or receiving, so scanning those three categories cannot miss anyone
who scored.

### Coverage

| season | players before | after | games before | after |
|--------|---------------|-------|--------------|-------|
| 2020 | 66  | 553 | 854   | 5,876 |
| 2021 | 81  | 575 | 1,157 | 5,561 |
| 2022 | 114 | 555 | 1,627 | 6,346 |
| 2023 | 144 | 535 | 2,045 | 6,512 |
| 2024 | 170 | 548 | 2,456 | 6,484 |
| 2025 | 543 | 543 | 6,197 | 6,197 |

1,212 players appeared across 2020–2024; 1,057 at QB/RB/WR/TE. Every event is covered
(271/272 in 2022 — the abandoned BUF at CIN game has no box score).

**After the `SUM(fp_ppr) > 0` season filter: 30,182 games / 1,011 players for
2020–2024**, against the 3,735 the within-2025 split allowed.

### Both consequences resolved

1. **Cross-season level mismatch is gone.** Mean PPR by season is now 7.35 / 7.99 /
   6.71 / 6.70 / 6.91 against 2025's 6.78. It was ~12 for 2022–2024 before, because
   those seasons only held players good enough to still be around in 2025.

2. **The shortcut feature collapsed.** The gap between players with and without a
   prior-season row fell from **9.06** (2.95 vs 12.01) to **3.28** (4.30 vs 7.58).
   The residual is real signal, not an artifact: the same check on 2024 vs 2023 —
   both fully populated, so no collection effect is possible — gives 3.35, essentially
   identical. A player with no prior season genuinely does score less.

174 players had 2024 games and no 2025 games. They were structurally invisible before.

### Known limits of the pull

- **2021 omits some scoreless appearances upstream.** Holding the player set fixed,
  the same 417 players show 21.9% zero-point games in 2020, 15.6% in 2021, 25.6% in
  2022 — the season, not its roster. This inflates 2021's mean to 7.99. Consider a
  season indicator, and don't read 2021 as a higher-scoring year.
- **Fullbacks are excluded.** The filter is QB/RB/WR/TE, matching what 2025 already
  contains, so the exclusion is uniform across seasons and adds no cross-season bias —
  but 26 FBs who appeared in 2020–2024 are not in the data. Widening it means
  re-running 2025 with the same position list so the seasons stay comparable.
- **9 players list seasons but serve no game log.** A known ESPN limitation, logged as
  warnings rather than failing silently.
- **Genuine rookies still have no prior season** (~6% of players). Unchanged by this
  work; needs draft position, college production, or a positional prior.

## Next

- Re-run the notebook on a multi-season split (train 2020–2024, test 2025).
- **Expect the metrics to look worse at first.** Losing the shortcut means separating
  players on real statistical differences. Compare against the `prev_fp_ppr_avg`
  passthrough baseline recomputed on the new data — not against the old 6.413 test RMSE.
- 2020 rows have no prior-season features (2019 was not pulled); either accept that or
  train on 2021–2024.
