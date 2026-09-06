# Fantasy Football Projections

Projects every player's PPR points for the upcoming season, then turns that into
a draft board that says **who to take and when**.

The main output is one file:

```
artifacts/draft_board_2026_10team.csv
```

Open it in Excel or on GitHub. It is sorted best-pick-first. Most of this README
explains what its columns mean.

---

## What the model actually does

It answers two separate questions about each player and multiplies the answers
together.

**How many games will he play?** Injuries, retirements and losing a job are most
of the downside risk in fantasy. A quarter of players in any season are out of
the league the next one. A projection that assumes everyone plays a full season
is wrong about almost everybody.

**How many points per game when he plays?** Learned from his own scoring history,
adjusted for age, and pushed around by how he was used, how his season finished,
where he sits on the depth chart, and whether he changed teams.

Rather than producing a single number, it simulates the season **4,000 times**
and reports the whole spread. That is where the intervals and probabilities in
the board come from — they are counts over simulated seasons, not formulas
applied to a point estimate.

### What it knows

- Ten seasons of game-by-game scoring, 2016–2025
- Age, position, and real NFL experience
- Usage: carries, targets, pass attempts per game
- **Roster status in August** — on a team, released, retired, unsigned
- **Depth chart position** going into the season
- Whether he **changed teams**, and whether his **quarterback changed**
- For running backs, **how good the rest of his backfield is**

### What it does not know

- **Rookies.** A player needs at least one prior NFL season to appear at all, so
  first-year players are simply absent from the board.
- **How good a new situation will be.** It knows a receiver's quarterback
  changed; it cannot know whether the replacement is any good. Team offences
  only carry over about 20% of their character year to year, which is too little
  to forecast from.
- Injuries that have not happened yet, coaching changes, holdouts, or anything
  reported after the August roster snapshot was taken.

---

## Reading the draft board

### Who he is

| column | meaning |
|---|---|
| `vbd_rank` | **The draft order this board recommends.** 1 is the best pick available. |
| `display_name`, `pos`, `team` | Player, position, and the team he is on for the upcoming season. |
| `pos_rank` | His rank **on this board** within his own position — RB1, WR7, and so on. |
| `depth_rank` | His rank **on his real NFL team's depth chart** at his position, as of August. 1 is the starter. **Blank means he is not listed on any depth chart** — which is a warning sign, not missing data: those players average under 3 games played. |

Two different rankings sit next to each other and are easy to confuse:
`pos_rank` is where *this board* puts him among players at his position;
`depth_rank` is where *his actual coach* puts him. A player who is WR2 on
this board and WR1 on his team is a very different proposition from one who
is WR2 on both.

### When he will actually go

| column | meaning |
|---|---|
| `adp` | **Average draft position** — the pick number he goes at on average in real ESPN PPR drafts. |
| `round`, `pick_in_round` | The same thing in league terms. Round 3, pick 4 means he typically goes early in the third. |
| `value` | **The bargain column.** How many draft slots he falls past where this board says he is worth taking. Positive is good: +21 means the room lets him last 21 picks longer than his value justifies. Negative means the room pays more for him than he is worth. |

### What he is worth

| column | meaning |
|---|---|
| `proj_mean` | Projected PPR points for the season, averaged across all 4,000 simulations. |
| `proj_median` | **The coin-flip number.** He beats this half the time. Not the same as the mean — see the warnings below. |
| `p05`, `p95` | **The realistic range.** He lands between these in 9 seasons out of 10 — `p05` is a disaster year, `p95` is everything going right. The width is the honest measure of how much of a gamble the pick is: McCaffrey spans 73–496, which is most of the board. |
| `vbd` | **Value over replacement.** Points above the player you could have had for free at his position, which is what makes a quarterback and a running back comparable. `vbd_rank` sorts on this. |
| `vbd_floor` | The same, computed from a bad-but-not-disastrous outcome (25th percentile). Sort by this when you need a reliable weekly starter. |
| `risk_shift` | How much safer or riskier he is than his headline rank suggests. **Negative means his value depends on upside**; positive means he is steadier than he looks. |

### How likely each outcome is

All computed by counting simulated seasons.

| column | meaning |
|---|---|
| `p_beat_replacement` | **Probability the pick is worth a roster slot at all** — that he outscores the freely available player at his position. 0.50 is a coin flip. |
| `p_top5_pos` | Probability he finishes **top five at his position**. The league-winner column, and it separates the top of the board more sharply than points do. |
| `p_starter` | Probability he finishes as a startable asset — top 12 for QB and TE, top 24 for RB, top 36 for WR. |
| `exp_games` | Expected games played, out of a possible 16. |

### Roster reality

| column | meaning |
|---|---|
| `status` | Where he stood on an NFL roster in August. `ACT` active · `RES` injured reserve · `DEV` practice squad · `CUT` released · `RET` retired · `NONE` on no roster at all. |
| `pct_owned` | Share of ESPN leagues where he is rostered — a rough popularity check. |

---

## Three things worth knowing before trusting a number

**The mean is not the midpoint.** Across the top 300 players, the actual result
beats `proj_mean` only 44% of the time, because a player's downside is bounded
at zero while his upside is not. The error is not uniform either: every
quarterback in the top 60 has a median 13–19 points *above* his mean, while
boom-or-bust young backs run the other way. **Use `proj_median` when you want a
fifty-fifty line.**

**The intervals are too wide.** In testing, the model's "50% likely" range
contained the real answer about 66% of the time. It is more uncertain than
reality warrants, so treat the extremes as softer than they look. Numbers from
the middle of the distribution are the trustworthy ones.

**Rookies are missing, and that shifts everything else.** Beyond being absent
from the board themselves, they absorb real draft picks — so players go a little
sooner than the `round` column suggests, most noticeably in the first four
rounds.

---

## How well does it work

Tested by training on everything up to a given season and predicting the next
one, which it has never seen. Against the previous version of this model, on the
two most recent testable seasons:

| | old | new |
|---|---|---|
| average error per player | 39.4 pts | **33.1 pts** |
| distributional score (lower is better) | 26.0 | **22.1** |

The improvement is statistically solid — a paired comparison over 1,078 player
seasons gives −3.9 with a 95% interval of [−4.8, −3.1] — and it comes almost
entirely from one place: **knowing whether a player is on an NFL roster.** The
old model projected Michael Thomas at 104 points and Ryan Tannehill at 115 in a
season where both were unsigned and scored zero.

---

## Running it

```bash
python run_backtest.py     # test the model against history
python run_projection.py   # fit on all data, project next season
python run_vbd.py  --teams 10                # convert to value over replacement
python run_adp.py  --teams 10 --rounds 16    # add draft position and bargains
```

Both board scripts take flags for league size and starting lineup:

```bash
python run_vbd.py --teams 12 --qb 1 --rb 2 --wr 3 --te 1 --flex 2
```

Everything lands in `artifacts/`. Projections read from a separate database
project holding the game logs, rosters and depth charts.

The modelling details — what the two halves assume, how the aging curve is
fitted, why the final week of every season is dropped — are in
[`bayes/README.md`](bayes/README.md).
