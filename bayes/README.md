# Bayesian next-season fantasy projection

Projects each player's **PPR point total for the following season** as a full
predictive distribution, from his own scoring history plus age and position.

```bash
python run_backtest.py      # expanding-window backtest + diagnostics figure
python run_projection.py    # fit on everything, project next season
```

Outputs land in `artifacts/`.

---

## Why a Bayesian model, specifically

Fantasy projection has three features that a point-estimate regression handles
badly and a hierarchical Bayesian model handles by construction.

**Sample sizes differ wildly between players.** A back who averaged 18 PPG over
four games and one who did it over sixteen are not the same evidence, and the
database says exactly how much noise is in each: the within-season game-to-game
variance. The Kalman update weights every season by its own measurement
variance, so the shrinkage is derived rather than hand-tuned. No "minimum games
played" cutoff appears anywhere in this code.

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

- **Six seasons, one league.** 2020–2025 is all the database holds at usable
  coverage; earlier seasons have 1–49 players and are excluded. Five transitions
  is thin for estimating a season random effect, and the aging curve is
  identified mostly cross-sectionally.
- **No team context.** Offensive environment, depth chart and target competition
  are not in the model, largely because a player's *next* season team is unknown
  in August. This is the biggest single source of missable signal.
- **Rookies are cold-started from age and position alone.** No draft capital or
  college production is available in the database, so a first-year player gets
  the positional prior for his age and a wide interval — honest, but not sharp.
- **The 2021 season under-reports scoreless appearances** upstream, which
  inflates its scoring mean. Worth a season indicator if it ever matters.
- **Fullbacks are excluded** (the position filter is QB/RB/WR/TE), uniformly
  across seasons.
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
