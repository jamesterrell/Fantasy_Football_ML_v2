# Bayesian next-season fantasy projection

Projects each player's **PPR point total for the following season** as a full
predictive distribution, from his own scoring history plus age and position.

```bash
python run_backtest.py      # expanding-window backtest + diagnostics figure
python run_projection.py    # fit on everything, project next season
```

Outputs land in `artifacts/`.

---

## The universe: draftable players only

The model is fit and scored on players who have posted **at least one 50-point
PPR season** (`MIN_PEAK_FP` in `data.py`). That drops 666 of 1,186 players and
1,294 of 3,309 player-seasons — a little under 40% of the panel.

This is a decision about what the model is *for*. A player who has never
cleared 50 points in a season is not draftable in any format, so the difference
between projecting him at 12 points and at 30 is a distinction nobody acts on;
spending likelihood on it buys accuracy where accuracy has no decision attached.

Two details make the cut narrower than it sounds:

- **Qualification is a property of the player, not the season.** A player who
  clears the bar once keeps *every* row he has, including his zero seasons. The
  decline from productive to nothing is exactly the trajectory the model needs,
  and cutting it would teach the model that good players stay good.
- **The label is untouched.** A qualifying player who scores nothing next
  season still contributes a zero. The filter selects who to model, never what
  the answer was.

### The filter cannot be applied with hindsight in a backtest

"Had a 50-point season in 2021–2025" is a fact available in 2026 and not
before. A fold that trains on ≤2021 and predicts 2022 must not use it: it would
keep the players who were *about to* break out and drop those about to wash
out, selecting the population on the outcomes being scored. `run_backtest.py`
therefore runs two modes — `causal`, where a player qualifies only on seasons
the fold has already seen, and `window`, the literal full-window rule — and the
gap between them is the size of the hindsight. **`causal` is the honest
number.** `run_projection.py` is entitled to the full window, because
projecting 2026 from 2021–2025 uses only seasons that have happened.

---

## Why a Bayesian model, specifically

Fantasy projection has three features that a point-estimate regression handles
badly and a hierarchical Bayesian model handles by construction.

**Sample sizes differ wildly between players.** A back who averaged 18 PPG over
four games and one who did it over sixteen are not the same evidence, and the
database says exactly how much noise is in each: the within-season game-to-game
variance. The Kalman update weights every season by its own measurement
variance, so the shrinkage is derived rather than hand-tuned. No "minimum games
played" cutoff appears anywhere in this code — the 50-point filter above selects
*which players are worth modelling*, and within that set every season a player
played is used at whatever weight its own noise level earns.

**The answer a fantasy manager needs is a distribution.** "220 points" is less
useful than "a 90% chance he plays, a median of 210, and a 1-in-20 chance he
clears 330". The model is scored on that basis, not only on RMSE.

**The uncertainty is structured, not homogeneous.** A 31-year-old with five
consistent seasons and a rookie with one loud half-season can carry identical
point projections and completely different spreads. The posterior knows the
difference; a tree ensemble's point prediction cannot express it.

---

## The model

Season points decompose into two questions that are best asked separately:

```
season points  =  games played  x  points per game
```

### 1. Production — points per game, given he plays

A hierarchical Gaussian **state-space model** on `z = sqrt(points per game)`:

```
level      m[i,t]  = base[pos] + agecurve[pos](age[i,t])
ability    z*[i,t] = m[i,t] + u[i] + w[i,t]
transient  w[i,t]  = rho[pos] * w[i,t-1] + inputs + noise
observed   z[i,t] ~ Normal(z*[i,t] + lam[pos]*missed_time, kappa[pos]*z_var[i,t])
```

Four choices in there carry most of the weight.

**The sqrt scale is empirical, not conventional.** Within-season variance is
close to proportional to the mean (`Var ≈ a + b·μ` with `b` ≈ 3.4–5.6 across
positions), which is the signature of an overdispersed count process and
exactly what a square root stabilises. On the sqrt scale the residual spread of
next-season scoring is near-constant across the skill range (0.73 / 0.73 / 0.53
by tercile); on the raw points scale it triples (2.6 / 3.5 / 3.7). A model with
one noise parameter needs the scale on which one noise parameter is true.

**Ability splits into permanent and transient.** `u[i]` is career-fixed, `w[i,t]`
decays at `rho`. A single AR(1) — the obvious first design — cannot represent
"this is his level" as distinct from "he is on a good run", so at `rho` ≈ 0.68 it
dragged four elite seasons two-thirds of the way back to the positional mean
every year. In backtest that showed up as the top projection decile landing 12%
under what those players actually scored. The permanent component is
substantial and clearly identified: `sigma_u` ≈ 0.6 on the sqrt scale.

**Missed time depresses the per-game rate, and that matters twice.** Within the
same player, seasons where he plays less are also seasons where he scores less
per game — 0.048 on the sqrt scale per game missed. Modelling it stops the
filter from reading an injury year as decline, and at projection time it feeds
the sampled games count back into the per-game rate. That second effect is what
puts back the positive covariance between games and scoring that sampling them
independently destroys: without it, `E[games x rate]` is computed as
`E[games] x E[rate]`, which loses about 5% at the top of the board.

**The latent abilities are integrated out.** The Kalman filter marginalises all
~6,000 of them analytically, so NUTS samples only ~30 hyperparameters. That is
why a fit takes under a minute instead of fighting thousands of correlated
latent variables.

### 2. Availability — how many games

A hurdle, because "out of the league" and "missed six weeks" are different
events and roughly a quarter of players in any season are gone the next one:

- **P(zero games)** — logistic in age, ability, last season's games, rookie
  status, plus `skill²` and `skill x games`. The nonlinear terms are there
  because the linear version was visibly wrong: it under-predicted dropout for
  below-average players (34% predicted against 46% actual) while over-predicting
  it for stars. Falling out of the league has a cliff near replacement level and
  almost nothing above it.
- **Games given he plays** — beta-binomial over 17. The overdispersion matters
  because missed games arrive in runs; a torn ACL costs eight straight weeks,
  not eight coin flips.

Coefficients are partially pooled across positions, so quarterbacks (463 rows)
borrow strength where their own data is thin while receivers (1,301) speak for
themselves. A **season random effect** carries league-wide turnover, which ran
0.25 / 0.28 / 0.28 / 0.25 / 0.32 over the observed transitions — a projection
draws a fresh year effect rather than asserting the training average.

### Joining them

The two halves are fit separately and combined by simulation. That is a
deliberate cut, not an oversight: letting games feed back into the production
model would allow injury-shortened seasons to quietly drag down ability
estimates. Each posterior draw samples games, then ability, then a season's
worth of game-to-game luck scaled to the games drawn — which is what makes a
projection for an injury-prone player correctly *wide* rather than merely low.

---

## Honesty of the forecast

Every fold trains on seasons up to a cutoff and is scored against what actually
happened next. The split is by season, never at random — a player's 2022 and
2023 rows share most of their signal. The measurement-variance law is refit per
fold too, so nothing from the future reaches the model, not even a nuisance
parameter.

See `artifacts/diagnostics.png` and the results table printed by
`run_backtest.py`.

---

## Known limits

- **Ten seasons, but not one league.** The panel is 2016–2025 (5,385
  player-seasons, 1,633 players); earlier seasons hold 1–7 players and are
  excluded. Nine transitions now estimate the season random effect. The cost is
  that stationarity is a live assumption rather than a shrug — 2016 and 2025
  differ in passing environment, RB committee usage, and schedule length — and a
  season random effect absorbs level shifts, not changed *relationships*.
  Whether `rho`, `sigma_u` and the aging curve differ pre- and post-2021 is
  untested.
- **Season length differs by era.** After the final scheduled week is dropped
  (see `data.py`), a season is **15 games in 2016–2020 and 16 from 2021 on**.
  `SEASON_GAMES = 16` now means only "the season being projected"; historical
  rows carry their own `season_games` and `games_frac`, and the availability
  model's beta-binomial ceiling is passed in per row.
- **Older seasons under-report marginal stat lines.** The receiving/passing yard
  identity reconciles at a median of 1.00 in all ten seasons, but the share of
  team-games missing something runs ~29–36% for 2016–20 against ~14–18% for
  2023–25. Upstream and unfixable; the interpretive consequence is that an era
  or season term partly measures completeness, not football, and it bites
  hardest on the availability model and the replacement-level dropout cliff.
- **No team context.** Offensive environment, depth chart and target competition
  are not in the model, largely because a player's *next* season team is unknown
  in August. This is the biggest single source of missable signal.
- **Rookies are cold-started from age and position alone.** No draft capital or
  college production is available in the database, so a first-year player gets
  the positional prior for his age and a wide interval — honest, but not sharp.
- **Fullbacks are excluded** (the position filter is QB/RB/WR/TE), uniformly
  across seasons.
- **The 50-point filter cannot see a breakout coming.** A player who has never
  cleared the bar is outside the universe entirely, so the model has nothing to
  say about the deep-league flier who posts 180 points out of nowhere. That is
  the accepted cost of the cut, and it is the one case where the pre-filter
  model was doing work this one does not.
- **Four development folds, and 2025 is sealed.** `run_backtest.py` predicts
  2021, 2022, 2023 and 2024 from cutoffs 2020–2023, and truncates the panel at
  2024 so the held-out season is absent rather than merely unscored. Scoring it
  requires `--final`.
- **Debut inference is only as good as the snapshot.** `experience_years` is
  frozen at a player's last active season, so `EXPERIENCE_REF - experience_years`
  is right for active players (median error 0, 74% exact) and meaningless for
  retired ones (median +4, 96% impossible). The `min` floor absorbs it, at the
  price of pinning a retired veteran's debut to his first *panel* season: ~111
  players are older than 25.5 in the season they are labelled a rookie.
- **Availability treats ability as known**, a consequence of the modularisation
  above. It slightly understates uncertainty, small next to the spread the games
  distribution contributes.

## Files

| File | What it holds |
|---|---|
| `data.py` | Season-grain panel, sqrt transform, measurement-variance law |
| `spline.py` | Natural cubic spline basis for the aging curve |
| `production.py` | State-space model + vectorised Kalman filter |
| `availability.py` | Hurdle beta-binomial games model |
| `predictive.py` | Fold fitting and posterior-predictive simulation |
| `metrics.py` | CRPS, PIT, coverage, and the point-forecast scorecard |
| `figures.py` | Diagnostics, aging curves, projection intervals |
