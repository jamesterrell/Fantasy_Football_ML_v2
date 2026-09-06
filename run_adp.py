"""Attach average draft position to the value board, and find the gaps.

VBD says what a player is worth. ADP says what he costs. The draft is decided by
the difference: a player worth the 20th pick who reliably goes 45th is where a
roster is actually built, and neither number says that on its own.

ADP comes from ESPN's fantasy API - their own drafts, PPR scoring
(`leaguedefaults/3`), which is the closest available match to a 10-team PPR
league and is keyed on the same athlete ids as the rest of this project, so the
join needs no crosswalk.

The number to read is `value`: ADP rank minus value rank. Positive means the
room lets him fall past where he is worth taking.

One caution that cannot be fixed from here. ADP includes rookies and this
model does not project them, so a rookie going in the third round pushes every
veteran's ADP later without pushing his value later. Rounds are therefore
slightly optimistic - expect players to go a little earlier than the `round`
column says, most noticeably in the first four rounds where rookie capital
concentrates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).parent / "artifacts"
ESPN_URL = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
            "/seasons/{season}/segments/0/leaguedefaults/3")


def fetch_adp(season: int, limit: int = 1200) -> pd.DataFrame:
    """Every player ESPN publishes an ADP for, most-owned first."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    filt = {"players": {"limit": limit,
                        "sortPercOwned": {"sortAsc": False, "sortPriority": 1}}}
    resp = session.get(
        ESPN_URL.format(season=season),
        params={"view": "kona_player_info"},
        headers={"x-fantasy-filter": json.dumps(filt)},
        timeout=90,
    )
    resp.raise_for_status()

    rows = []
    for entry in resp.json().get("players") or []:
        player = entry.get("player") or {}
        own = player.get("ownership") or {}
        adp = own.get("averageDraftPosition")
        # ESPN returns 0 for players nobody drafts, which is not an ADP of zero.
        if not adp or adp <= 0:
            continue
        rows.append({
            "athlete_id": str(entry.get("id")),
            "espn_name": player.get("fullName"),
            "adp": float(adp),
            "pct_owned": float(own.get("percentOwned") or 0.0),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=int, default=10)
    ap.add_argument("--dir", default=None,
                    help="artifacts directory to read and write; "
                         "lets a candidate board be built without "
                         "touching the one in use")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--rounds", type=int, default=16,
                    help="how deep the draft goes; bounds the value analysis")
    a = ap.parse_args()
    if a.dir:
        OUT = Path(a.dir)
        OUT.mkdir(parents=True, exist_ok=True)

    board = pd.read_csv(OUT / f"vbd_board_{a.season}_{a.teams}team.csv")
    # Neither output CSV carries athlete_id - both are written for reading -
    # so recover it from the database by name. Names come from the same ESPN
    # feed on both sides, so this is an exact match rather than fuzzy matching,
    # and it is checked below rather than assumed.
    from helpers.db_query import query_db
    ids = query_db(
        "SELECT athlete_id, display_name FROM athletes "
        "WHERE position_abbr IN ('QB','RB','WR','TE')"
    ).drop_duplicates("display_name", keep=False)   # drop ambiguous names
    board = board.merge(ids, on="display_name", how="left")
    board["athlete_id"] = board.athlete_id.astype("string")
    missing_id = board.athlete_id.isna().sum()
    if missing_id:
        print(f"note: {missing_id} board players could not be keyed to an id "
              f"(duplicate or renamed); they will show no ADP")

    adp = fetch_adp(a.season)
    adp["athlete_id"] = adp.athlete_id.astype("string")
    print(f"ESPN published an ADP for {len(adp)} players")

    out = board.merge(adp[["athlete_id", "adp", "pct_owned"]], on="athlete_id", how="left")
    matched = out.adp.notna().sum()
    print(f"matched to the board: {matched} of {len(out)} "
          f"({matched / len(out):.0%}); top 100 by value: "
          f"{out.head(100).adp.notna().sum()}/100")

    out["adp_rank"] = out.adp.rank(method="min")
    out["round"] = out.adp.apply(lambda x: math.ceil(x / a.teams) if pd.notna(x) else pd.NA)
    out["pick_in_round"] = out.adp.apply(
        lambda x: int(round(x)) - a.teams * (math.ceil(x / a.teams) - 1)
        if pd.notna(x) else pd.NA)
    # Positive = the room lets him fall past where his value says to take him.
    out["value"] = out.adp_rank - out.vbd_rank

    cols = [c for c in ("vbd_rank", "display_name", "pos", "pos_rank", "team",
                        "adp", "round", "pick_in_round", "value",
                        "proj_mean", "proj_median",
                        "vbd", "vbd_floor", "risk_shift",
                        "p_beat_replacement", "p_top5_pos", "p_starter",
                        "exp_games", "status", "pct_owned")
            if c in out.columns]
    dest = OUT / f"draft_board_{a.season}_{a.teams}team.csv"
    out.sort_values("vbd_rank")[cols].to_csv(dest, index=False)
    print(f"\nwrote {dest}")

    pd.set_option("display.width", 250)
    # Value only means something inside the range that actually gets drafted.
    # Across all 424 the largest "values" are players who go undrafted in every
    # league, which is arithmetically true and useless: their ADP rank is huge
    # because nobody takes them, not because the room is wrong about them.
    depth = a.teams * a.rounds
    have = out[out.adp.notna() & (out.adp <= depth)].copy()
    show = [c for c in cols if c not in ("pct_owned", "status")]

    print("")
    print(f"=== BEST VALUE inside {a.rounds} rounds "
          f"(positive value = falls later than he is worth) ===")
    print(have.nlargest(20, "value")[show].round(1).to_string(index=False))

    print("")
    print("=== REACHES: the room pays more than the model says ===")
    print(have.nsmallest(12, "value")[show].round(1).to_string(index=False))

    print("")
    print("=" * 120)
    print("ROUND BY ROUND - who is on the board, and the best value in each")
    print("=" * 120)
    for rd in range(1, a.rounds + 1):
        block = have[have["round"] == rd].sort_values("vbd_rank")
        if block.empty:
            continue
        best = block.nlargest(3, "value")
        names = ", ".join(
            f"{r.display_name} ({r.pos}{r.pos_rank}, val {r.value:+.0f})"
            for r in best.itertuples())
        print(f"Round {rd:>2} (picks {(rd - 1) * a.teams + 1}-{rd * a.teams}): "
              f"{len(block)} on the board, best VBD rank {int(block.vbd_rank.min())}")
        print(f"    target: {names}")
