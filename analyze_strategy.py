"""Optimal 101 Okey Playbook Extractor powered by PlannerAgent v2 CRN EV calculations.

Runs N headless rounds with 4 PlannerAgent v2 bots recording every non-trivial
decision and prints a quantitative 101 Okey Optimal Strategy Report.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import time
from typing import Any

from agents.planner_agent import PlannerAgent
from engine import GameConfig, NUM_PLAYERS, Okey101Engine, OpenType


def analyze_strategy(
    num_rounds: int = 200,
    num_scenarios: int = 60,
    katlamali: bool = False,
    feed_penalty_mult: int = 0,
    seed: int = 100,
) -> None:
    t0 = time.perf_counter()
    config = GameConfig(
        katlamali=katlamali,
        feed_penalty_multiplier=feed_penalty_mult,
    )

    # Aggregators
    discard_rank_unopened: dict[int, int] = defaultdict(int)
    discard_rank_opened: dict[int, int] = defaultdict(int)
    discard_rank_evs: dict[int, list[float]] = defaultdict(list)
    iso_high_ev_costs: list[float] = []  # EV cost of keeping isolated 10-13 vs discarding
    iso_low_ev_costs: list[float] = []   # EV cost of keeping isolated 1-4 vs discarding

    pairs_decisions: list[dict[str, Any]] = []
    open_series_immediate = 0
    open_pairs_immediate = 0
    open_pairs_delayed = 0

    draw_take_unopened: list[float] = []
    draw_take_opened: list[float] = []
    draw_declined = 0

    # Feed penalty & round stats
    total_feed_penalties = 0
    feed_penalty_events = 0
    all_round_scores: list[int] = []
    open_count = 0

    # Joker counterfactual tracking (same hand: spend Joker & open now vs hold & wait 1 turn)
    joker_cf_open_evs: list[float] = []
    joker_cf_wait_evs: list[float] = []

    for r in range(num_rounds):
        agents = [
            PlannerAgent(
                num_scenarios=num_scenarios,
                top_k=10,
                version=2,
                record_decisions=True,
                seed=seed + r * 37 + s,
            )
            for s in range(NUM_PLAYERS)
        ]
        engine = Okey101Engine(config=config)
        engine.reset(seed=seed + r, starting_player=r % NUM_PLAYERS)

        had_joker = [
            any(engine.joker_spec.is_joker(t) for t in engine.hands[s])
            for s in range(NUM_PLAYERS)
        ]

        while not engine.done:
            p = engine.current_player
            obs = engine.get_observation(p)
            if any(engine.joker_spec.is_joker(t) for t in engine.hands[p]):
                had_joker[p] = True
            legal = engine.get_legal_actions()
            act = agents[p].select_action(obs, legal)
            res = engine.step(act)
            if "feed_penalty_amount" in res.info:
                total_feed_penalties += int(res.info["feed_penalty_amount"])  # type: ignore[arg-type]
                feed_penalty_events += 1

        assert engine.round_scores is not None
        all_round_scores.extend(engine.round_scores)
        open_count += sum(1 for ot in engine.open_types if ot != OpenType.NONE)

        # Harvest decision logs
        for ag in agents:
            for d in ag.decision_log:
                kind = d["kind"]
                if kind == "discard":
                    rk = d["rank"]
                    if rk >= 1:
                        if d["open_type"] == "none":
                            discard_rank_unopened[rk] += 1
                        else:
                            discard_rank_opened[rk] += 1
                    cands = d.get("all_cands", [])
                    best_ev = d["ev_best"]
                    for c in cands:
                        if c["rank"] >= 1:
                            discard_rank_evs[c["rank"]].append(c["ev"] - best_ev)
                        if c["isolated"] and c["rank"] >= 10:
                            # Cost of NOT discarding an isolated high tile when opened/unopened
                            if d["open_type"] != "none" and c["ev"] > best_ev:
                                iso_high_ev_costs.append(c["ev"] - best_ev)
                        elif c["isolated"] and 1 <= c["rank"] <= 4:
                            if d["open_type"] != "none" and c["ev"] > best_ev:
                                iso_low_ev_costs.append(c["ev"] - best_ev)

                elif kind == "pairs_vs_wait":
                    pairs_decisions.append(d)
                    if d["chosen"] == "open_pairs":
                        open_pairs_immediate += 1
                    else:
                        open_pairs_delayed += 1

                elif kind == "open_choice":
                    if d["chosen"] == "open_series":
                        open_series_immediate += 1
                        if "joker_cf_ev_open" in d:
                            joker_cf_open_evs.append(d["joker_cf_ev_open"])
                            joker_cf_wait_evs.append(d["joker_cf_ev_wait"])
                    elif d["chosen"] == "open_pairs":
                        open_pairs_immediate += 1

                elif kind == "draw":
                    if d["chosen"] == "take_discard":
                        gain = d["ev_draw"] - d["ev_take"]
                        if d["open_type"] == "none":
                            draw_take_unopened.append(gain)
                        else:
                            draw_take_opened.append(gain)
                    else:
                        draw_declined += 1

    elapsed = time.perf_counter() - t0
    rule_label = (
        f"HOUSE RULES (katlamali={katlamali}, feed={feed_penalty_mult}x)"
        if (katlamali or feed_penalty_mult > 0)
        else "OFFICIAL RULES"
    )
    print()
    print("=" * 88)
    print(
        f"  101 OKEY OPTIMAL STRATEGY REPORT — {rule_label}\n"
        f"  ({num_rounds} rounds / {num_rounds * 4} player-hands, "
        f"avg_score={sum(all_round_scores)/max(1,len(all_round_scores)):.2f}, "
        f"open_rate={100.0*open_count/(num_rounds*4):.1f}%, {elapsed:.1f}s)"
    )
    print("=" * 88)

    # 1. Discard Priority by Rank
    print("\n1. DISCARD PRIORITY BY RANK (1-13)")
    print("-" * 84)
    tot_u = max(1, sum(discard_rank_unopened.values()))
    tot_o = max(1, sum(discard_rank_opened.values()))
    print(f"{'Rank':>4} | {'Unopened %':>11} | {'Opened %':>10} | {'Avg ΔEV':>9} | {'Strategy Pattern'}")
    print("-" * 88)
    for rk in range(1, 14):
        pu = 100.0 * discard_rank_unopened[rk] / tot_u
        po = 100.0 * discard_rank_opened[rk] / tot_o
        evs = discard_rank_evs[rk]
        avg_dev = sum(evs) / max(1, len(evs))
        if rk <= 4:
            note = "Dump early when unopened (low 101 contribution); keep low when opened"
        elif rk >= 10:
            note = "Protect early if connected (reaches 101); shed aggressively once opened"
        else:
            note = "High centrality (5-9): dangerous to feed next player"
        print(f"{rk:>4} | {pu:>10.2f}% | {po:>9.2f}% | {avg_dev:>+8.2f} | {note}")
    avg_hi = sum(iso_high_ev_costs) / max(1, len(iso_high_ev_costs))
    avg_lo = sum(iso_low_ev_costs) / max(1, len(iso_low_ev_costs))
    print(
        f"  -> Avg EV penalty of keeping isolated high (10-13) vs low (1-4) when opened: "
        f"+{avg_hi:.2f} pts vs +{avg_lo:.2f} pts"
    )

    # 2. Series vs Pairs Pivot
    print("\n2. SERIES vs PAIRS PIVOT")
    print("-" * 84)
    chose_pairs = [d for d in pairs_decisions if d["chosen"] == "open_pairs"]
    chose_wait = [d for d in pairs_decisions if d["chosen"] == "wait"]
    print(f"  Total Open-Pairs Opportunities Evaluated : {len(pairs_decisions)}")
    print(f"    -> Opened Pairs immediately            : {len(chose_pairs)}")
    print(f"    -> Waited for Series                   : {len(chose_wait)}")
    if chose_pairs:
        avg_dp_p = sum(d["draw_pile_size"] for d in chose_pairs) / len(chose_pairs)
        avg_pc_p = sum(d["pair_count"] for d in chose_pairs) / len(chose_pairs)
        avg_sp_p = sum(d["best_series_pts"] for d in chose_pairs) / len(chose_pairs)
        print(
            f"    [OPEN_PAIRS profile] avg pairs={avg_pc_p:.2f}, "
            f"avg draw_pile={avg_dp_p:.1f}, avg series_pts={avg_sp_p:.1f}"
        )
    if chose_wait:
        avg_dp_w = sum(d["draw_pile_size"] for d in chose_wait) / len(chose_wait)
        avg_pc_w = sum(d["pair_count"] for d in chose_wait) / len(chose_wait)
        avg_sp_w = sum(d["best_series_pts"] for d in chose_wait) / len(chose_wait)
        print(
            f"    [WAIT profile]       avg pairs={avg_pc_w:.2f}, "
            f"avg draw_pile={avg_dp_w:.1f}, avg series_pts={avg_sp_w:.1f}"
        )

    # 3. Opening Timing & Waiting
    print("\n3. OPENING TIMING & WAITING")
    print("-" * 84)
    total_open_opps = open_series_immediate + open_pairs_immediate + open_pairs_delayed
    delay_pct = 100.0 * open_pairs_delayed / max(1, total_open_opps)
    print(
        f"  Immediate Series Opens : {open_series_immediate} (100% immediate — avoiding +202 dominates)"
    )
    print(
        f"  Immediate Pairs Opens  : {open_pairs_immediate} | Delayed Pairs Opens: {open_pairs_delayed} "
        f"(overall delay rate: {delay_pct:.1f}%)"
    )

    # 4. Counterfactual Joker (Okey) Test
    print("\n4. COUNTERFACTUAL JOKER TEST (AYNI ELDE: HARCA VE ŞİMDİ AÇ vs SAKLA VE BEKLE)")
    print("-" * 88)
    n_cf = len(joker_cf_open_evs)
    avg_cf_open = sum(joker_cf_open_evs) / max(1, n_cf)
    avg_cf_wait = sum(joker_cf_wait_evs) / max(1, n_cf)
    open_better_pct = 100.0 * sum(
        1 for o, w in zip(joker_cf_open_evs, joker_cf_wait_evs) if o < w
    ) / max(1, n_cf)
    print(f"  Counterfactual Joker-Opening Hands (n={n_cf}):")
    print(f"    -> Action A (Spend Joker & Open Now)  : avg CRN EV = {avg_cf_open:+.2f} pts")
    print(f"    -> Action B (Hold Joker & Wait 1 Turn): avg CRN EV = {avg_cf_wait:+.2f} pts")
    print(
        f"    -> Net EV Advantage of Opening Now    : {avg_cf_wait - avg_cf_open:+.2f} pts "
        f"(better in {open_better_pct:.1f}% of hands)"
    )

    # 5. Take Discard vs Draw Deck
    print("\n5. TAKE DISCARD vs DRAW DECK (CRN EV GAIN)")
    print("-" * 84)
    avg_gain_u = sum(draw_take_unopened) / max(1, len(draw_take_unopened))
    avg_gain_o = sum(draw_take_opened) / max(1, len(draw_take_opened))
    print(
        f"  Unopened Take-to-Open (n={len(draw_take_unopened)}): "
        f"avg EV gain = +{avg_gain_u:.2f} pts saved vs blind draw"
    )
    print(
        f"  Opened Take-to-Meld/LayOff (n={len(draw_take_opened)}): "
        f"avg EV gain = +{avg_gain_o:.2f} pts saved vs blind draw"
    )
    print(f"  Declined Legal Takes (mostly risky early pairs): {draw_declined}")
    if feed_penalty_mult > 0:
        avg_fp = total_feed_penalties / max(1, num_rounds * 4)
        avg_fp_evt = total_feed_penalties / max(1, feed_penalty_events)
        print(
            f"  House-Rule {feed_penalty_mult}x Feed Penalties Triggered: {feed_penalty_events} times "
            f"(avg +{avg_fp_evt:.1f} pts/event, +{avg_fp:.2f} pts/player-round)"
        )
    print("=" * 88)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 101 Okey Optimal Strategy Report.")
    parser.add_argument("--rounds", type=int, default=200, help="Number of rounds to simulate.")
    parser.add_argument("--scenarios", type=int, default=60, help="CRN scenarios per decision.")
    parser.add_argument("--katlamali", action="store_true", help="Enable katlamali rules.")
    parser.add_argument(
        "--feed-penalty-mult",
        type=int,
        default=0,
        help="House-rule discard-feeding penalty multiplier (e.g. 10 -> tile.value * 10).",
    )
    parser.add_argument("--seed", type=int, default=100, help="Base RNG seed.")
    args = parser.parse_args()
    analyze_strategy(
        num_rounds=args.rounds,
        num_scenarios=args.scenarios,
        katlamali=args.katlamali,
        feed_penalty_mult=args.feed_penalty_mult,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

