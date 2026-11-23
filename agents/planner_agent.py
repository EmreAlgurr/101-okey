"""Determinized Monte Carlo Planner (v1 & v2) for Official Professional 101 Okey.

PlannerAgent v2 upgrades:
1. Opponent-Finish-Risk Horizon: dynamically truncates CRN draw scenarios based on
   opened opponents' remaining hand sizes, forcing timely openings and deadwood shedding.
2. Defensive Discard Model: adds a calibrated next-player feeding penalty based on
   next-player open status, rank centrality (5-9 vs edges), and unseen neighbor liveness.
3. Higher Precision: default 80 CRN scenarios (< 350 ms/act).
4. Optional decision telemetry (`record_decisions=True`) for strategy extraction.

Strictly information-set: only `obs` is used (never `engine_ref`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from typing import Any, Optional, Sequence

from agents.base import Agent
from agents.heuristic_agent import (
    HeuristicAgent,
    _count_visible_keys,
    _is_layable_anywhere,
    score_discard_candidate,
)
from agents.ismcts_agent import _find_immediate_win
from engine import (
    Action,
    ActionType,
    NUM_PLAYERS,
    Observation,
    Okey101Engine,
    OpenType,
    TurnPhase,
)
from solver import solve_best_opening, solve_best_pairs, solve_min_deadwood
from tiles import (
    Color,
    JokerSpec,
    Meld,
    NUM_NUMBERS,
    Tile,
    apply_lay_off,
    can_lay_off,
    color_number_to_key,
    create_deck,
)

UNOPENED_PENALTY = 202.0
_DECK: tuple[Tile, ...] = tuple(create_deck())


@dataclass(frozen=True, slots=True)
class _Ctx:
    spec: JokerSpec
    req_pts: int
    req_pairs: int
    table: tuple[Meld, ...]


# ---------------------------------------------------------------------------
# Fast hand helpers
# ---------------------------------------------------------------------------


def _key_counts(hand: Sequence[Tile], spec: JokerSpec) -> dict[int, int]:
    counts: dict[int, int] = {}
    for t in hand:
        k = spec.effective_key(t)
        if k >= 0:
            counts[k] = counts.get(k, 0) + 1
    return counts


def _conn(k: int, counts: dict[int, int]) -> float:
    """Weighted meld-connectivity of tile kind `k` w.r.t. hand `counts` (no 12-13-1 wrap)."""
    c, r = divmod(k, NUM_NUMBERS)
    base = c * NUM_NUMBERS
    s = 0.0
    if r >= 1 and counts.get(base + r - 1):
        s += 1.0
    if r <= 11 and counts.get(base + r + 1):
        s += 1.0
    if r >= 2 and counts.get(base + r - 2):
        s += 0.5
    if r <= 10 and counts.get(base + r + 2):
        s += 0.5
    for oc in range(4):
        if oc != c and counts.get(oc * NUM_NUMBERS + r):
            s += 1.0
    if counts.get(k, 0) >= 2:
        s += 0.3
    return s


def _connects(tile: Tile, hand: Sequence[Tile], spec: JokerSpec) -> bool:
    """True iff `tile` could participate in some Run/Set with tiles of `hand`."""
    k = spec.effective_key(tile)
    if k == -1:
        return True
    c, r = divmod(k, NUM_NUMBERS)
    jokers = 0
    for h in hand:
        hk = spec.effective_key(h)
        if hk == -1:
            jokers += 1
            if jokers >= 2:
                return True
            continue
        hc, hr = divmod(hk, NUM_NUMBERS)
        if hc == c:
            if hr != r and abs(hr - r) <= 2:
                return True
        elif hr == r:
            return True
    return False


def _layable(tile: Tile, table: Sequence[Meld], spec: JokerSpec) -> bool:
    for m in table:
        if can_lay_off(m, tile, spec):
            return True
    return False


def _win_value(okey: bool, elden: bool, pairs: bool) -> float:
    mult = 1
    if okey:
        mult *= 2
    if elden:
        mult *= 2
    if pairs:
        mult *= 2
    return -101.0 * mult


def _final_penalty(hand: Sequence[Tile], ot: OpenType, spec: JokerSpec) -> float:
    if ot == OpenType.NONE:
        return UNOPENED_PENALTY
    s = float(sum(spec.hand_penalty_value(t) for t in hand))
    return 2.0 * s if ot == OpenType.PAIRS else s


def _series_reduce(
    hand: list[Tile], table: list[Meld], spec: JokerSpec
) -> list[Tile]:
    """Greedy lay-offs (non-jokers by value), min-deadwood melding, then joker lay-offs."""
    progress = True
    while progress and hand:
        progress = False
        order = sorted(
            (t for t in hand if not spec.is_joker(t)),
            key=spec.hand_penalty_value,
            reverse=True,
        )
        for t in order:
            for i, m in enumerate(table):
                if can_lay_off(m, t, spec):
                    table[i] = apply_lay_off(m, t, spec)
                    hand.remove(t)
                    progress = True
                    break
            if progress:
                break
    if len(hand) >= 3:
        dw = solve_min_deadwood(hand, spec)
        if dw.melds:
            table.extend(dw.melds)
            hand = list(dw.deadwood_tiles)
    for t in [x for x in hand if spec.is_joker(x)]:
        if len(hand) < 2:
            break
        for i, m in enumerate(table):
            if can_lay_off(m, t, spec):
                table[i] = apply_lay_off(m, t, spec)
                hand.remove(t)
                break
    return hand


def _pick_rollout_discard(
    hand: Sequence[Tile],
    ot: OpenType,
    melded_ids: frozenset[int],
    table: Sequence[Meld],
    spec: JokerSpec,
    remaining: int,
) -> Tile:
    """Fast rollout discard policy (never discards a real Joker or işlek tile if avoidable)."""
    counts = _key_counts(hand, spec)
    num_pairs = sum(1 for c in counts.values() if c >= 2)
    scored: list[tuple[float, Tile]] = []
    for t in hand:
        if spec.is_joker(t):
            continue
        k = spec.effective_key(t)
        val = k % NUM_NUMBERS + 1
        if ot == OpenType.NONE:
            pair_bonus = (
                18.0
                if (counts.get(k, 0) >= 2 and num_pairs >= 2)
                else (4.0 if counts.get(k, 0) >= 2 else 0.0)
            )
            val_term = 0.8 * val if remaining >= 2 else -0.5 * val
            s = (
                (1000.0 if t.tile_id in melded_ids else 0.0)
                + 12.0 * _conn(k, counts)
                + pair_bonus
                + val_term
            )
        elif ot == OpenType.SERIES:
            s = -(val - (2.5 * _conn(k, counts) if remaining > 0 else 0.0))
        else:
            s = (1000.0 if counts.get(k, 0) >= 2 else 0.0) - val
        scored.append((s, t))
    scored.sort(key=lambda x: x[0])
    for _, t in scored:
        if not _layable(t, table, spec):
            return t
    if scored:
        return scored[0][1]
    return hand[0]


def _rollout(
    start_hand: Sequence[Tile],
    ot: OpenType,
    draws: Sequence[Tile],
    ctx: _Ctx,
    melded_ids: frozenset[int],
    truncated_by_opp: bool = False,
) -> float:
    """Simulate our own remaining turns on one sampled draw sequence; return our final score."""
    spec = ctx.spec
    hand = list(start_hand)
    table = list(ctx.table)
    n = len(draws)

    for i, t in enumerate(draws):
        remaining = n - i - 1
        elden = False
        if ot == OpenType.NONE:
            connects = _connects(t, hand, spec)
            hand.append(t)
            if connects:
                sol = solve_best_opening(hand, spec, min_points=ctx.req_pts)
                if sol.meets_threshold:
                    ot = OpenType.SERIES
                    elden = True
                    table.extend(sol.melds)
                    hand = list(sol.deadwood_tiles)
                else:
                    melded_ids = frozenset(
                        pt.tile.tile_id for m in sol.melds for pt in m.placed_tiles
                    )
            if ot == OpenType.NONE and remaining == 0:
                psol = solve_best_pairs(hand, spec, min_pairs=ctx.req_pairs)
                if psol.meets_threshold:
                    dw = list(psol.deadwood_tiles)
                    if len(dw) <= 1:
                        return _win_value(len(dw) == 1 and spec.is_joker(dw[0]), True, True)
                    dw.remove(
                        _pick_rollout_discard(dw, OpenType.PAIRS, frozenset(), table, spec, 0)
                    )
                    return min(2.0 * sum(spec.hand_penalty_value(x) for x in dw), UNOPENED_PENALTY)
        else:
            hand.append(t)

        if ot == OpenType.SERIES:
            hand = _series_reduce(hand, table, spec)
        elif ot == OpenType.PAIRS:
            psol = solve_best_pairs(hand, spec, min_pairs=1)
            if psol.pairs:
                hand = list(psol.deadwood_tiles)

        if ot != OpenType.NONE:
            if not hand:
                return _win_value(False, elden, ot == OpenType.PAIRS)
            if len(hand) == 1:
                return _win_value(spec.is_joker(hand[0]), elden, ot == OpenType.PAIRS)

        hand.remove(_pick_rollout_discard(hand, ot, melded_ids, table, spec, remaining))

    pen = _final_penalty(hand, ot, spec)
    # If an opponent won early, small expected multiplier risk (e.g. Okey/Pairs finish)
    return pen * 1.08 if truncated_by_opp else pen


# ---------------------------------------------------------------------------
# v2 Opponent-Finish-Risk Horizon & Defensive Feeding Model
# ---------------------------------------------------------------------------


def opponent_cycle_finish_hazard(obs: Observation) -> float:
    """Estimate probability that an opponent finishes within 1 full table cycle.

    Strictly gated to opened opponents with <= 4 tiles left so early-round
    CRN scenarios are never artificially truncated.
    """
    me = obs.player_id
    survive = 1.0
    for opp in range(NUM_PLAYERS):
        if opp == me:
            continue
        if obs.open_types[opp] == OpenType.NONE:
            continue
        sz = obs.hand_sizes[opp]
        if sz <= 1:
            h = 0.60
        elif sz == 2:
            h = 0.35
        elif sz == 3:
            h = 0.18
        elif sz == 4:
            h = 0.09
        else:
            continue
        survive *= 1.0 - h
    return 1.0 - survive


def next_player_feed_penalty(
    tile: Tile,
    obs: Observation,
    visible_counts: Counter[int],
) -> float:
    """Calibrated expected-cost penalty for potentially feeding the next player.

    - Official rules (`feed_penalty_multiplier == 0`): gentle defensive tie-break
      (halved while unopened so opening rate is never sacrificed).
    - House rules (`feed_penalty_multiplier > 0`): scales directly with
      `tile.value * feed_penalty_multiplier * P(next_player_takes_tile)`.
    """
    spec = obs.joker_spec
    if spec.is_joker(tile):
        return 101.0
    eff = spec.effective_color_number(tile)
    if eff is None:
        return 101.0
    c, n = eff
    k = color_number_to_key(c, n)

    next_p = (obs.player_id + 1) % NUM_PLAYERS
    next_ot = obs.open_types[next_p]
    next_sz = obs.hand_sizes[next_p]
    my_ot = obs.open_types[obs.player_id]

    # Rank centrality: middle ranks 5-9 connect on both sides; edges 1 & 13 on one side
    if 5 <= n <= 9:
        centrality = 1.0
    elif n in (3, 4, 10, 11):
        centrality = 0.78
    elif n in (2, 12):
        centrality = 0.48
    else:
        centrality = 0.22

    # Liveness: fraction of adjacent / same-rank / duplicate tiles still unseen
    unseen_score = 0.0
    max_possible = 0.0

    if n > 1:
        unseen_score += max(0, 2 - visible_counts[color_number_to_key(c, n - 1)])
        max_possible += 2.0
    if n < 13:
        unseen_score += max(0, 2 - visible_counts[color_number_to_key(c, n + 1)])
        max_possible += 2.0
    if n > 2:
        unseen_score += 0.35 * max(0, 2 - visible_counts[color_number_to_key(c, n - 2)])
        max_possible += 0.7
    if n < 12:
        unseen_score += 0.35 * max(0, 2 - visible_counts[color_number_to_key(c, n + 2)])
        max_possible += 0.7

    for oc in Color:
        if oc != c:
            unseen_score += 0.65 * max(0, 2 - visible_counts[color_number_to_key(oc, n)])
            max_possible += 1.3

    unseen_dup = max(0, 2 - visible_counts[k])
    unseen_score += 0.45 * unseen_dup
    max_possible += 0.9

    liveness = unseen_score / max(1.0, max_possible)

    # Estimate P(next player immediately takes this discard)
    if next_ot != OpenType.NONE:
        base_take = 0.24 if next_sz <= 5 else 0.16
        p_take = base_take * centrality * liveness
    else:
        base_take = 0.13 if obs.draw_pile_size <= 12 else 0.08
        high_boost = 1.0 + (0.08 * (n - 8) if n >= 9 else 0.0)
        p_take = base_take * centrality * liveness * high_boost

    # House-rule direct feed penalty: tile.value * mult * P(take) + tactical swing
    if obs.feed_penalty_multiplier > 0:
        direct_pen = n * obs.feed_penalty_multiplier * p_take
        tactical_pen = 6.0 * p_take
        return direct_pen + tactical_pen

    # Official rules: halved weight while unopened to protect opening rate
    tactical_weight = 6.0 if my_ot == OpenType.NONE else 12.0
    return tactical_weight * p_take


# ---------------------------------------------------------------------------
# Draw sampling (information-set consistent)
# ---------------------------------------------------------------------------


def _draw_positions(draw_pile_size: int, first: int) -> list[int]:
    return list(range(first, draw_pile_size + 1, NUM_PLAYERS))


def _unseen_pool(obs: Observation) -> tuple[list[Tile], Optional[Tile]]:
    known: set[int] = {t.tile_id for t in obs.hand}
    for pile in obs.discard_piles:
        known.update(t.tile_id for t in pile)
    for tm in obs.table_melds:
        known.update(pt.tile.tile_id for pt in tm.meld.placed_tiles)
    ind = obs.joker_spec.indicator
    bottom = ind if (obs.draw_pile_size >= 1 and ind.tile_id not in known) else None
    pool = [t for t in _DECK if t.tile_id not in known and t.tile_id != ind.tile_id]
    return pool, bottom


def _sample_seqs(
    pool: Sequence[Tile],
    bottom: Optional[Tile],
    positions: Sequence[int],
    draw_pile_size: int,
    n: int,
    rng: random.Random,
    opp_hazard: float = 0.0,
) -> list[tuple[list[Tile], bool]]:
    """Sample `n` draw sequences, optionally truncating stochastically by `opp_hazard`."""
    m = len(positions)
    if m == 0:
        return [([], False)]
    need_bottom = bottom is not None and positions[-1] == draw_pile_size
    k = min(len(pool), m - (1 if need_bottom else 0))
    seqs: list[tuple[list[Tile], bool]] = []
    for _ in range(n):
        s = rng.sample(pool, k)
        if need_bottom:
            assert bottom is not None
            s.append(bottom)
        truncated = False
        if opp_hazard > 0.0:
            eff_len = 0
            while eff_len < len(s):
                if rng.random() < opp_hazard:
                    truncated = True
                    break
                eff_len += 1
            s = s[:eff_len]
        seqs.append((s, truncated))
    return seqs


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PlannerAgent(Agent):
    """Determinized CRN Monte Carlo planner (v1 or opponent-aware v2)."""

    def __init__(
        self,
        num_scenarios: int = 80,
        top_k: int = 10,
        tiebreak: float = 0.14,
        version: int = 2,
        record_decisions: bool = False,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        rollouts: Optional[int] = None,
    ) -> None:
        default_name = f"Planner-v{version}"
        super().__init__(name=name or default_name)
        self.version = version
        self.num_scenarios = rollouts if rollouts is not None else num_scenarios
        self.top_k = top_k
        self.tiebreak = tiebreak
        self.record_decisions = record_decisions
        self.decision_log: list[dict[str, Any]] = []
        self.rng = random.Random(seed)
        self._heur = HeuristicAgent(seed=seed)

    def select_action(
        self,
        obs: Observation,
        legal_actions: Sequence[Action],
        engine_ref: Optional[Okey101Engine] = None,
    ) -> Action:
        del engine_ref  # Strictly information-set: never peek at hidden state.
        if len(legal_actions) == 1:
            return legal_actions[0]
        if obs.phase == TurnPhase.DRAW:
            return self._select_draw(obs, legal_actions)
        win = _find_immediate_win(obs.hand, legal_actions, obs.joker_spec)
        if win is not None:
            return win
        return self._select_play(obs, legal_actions)

    def _ctx(self, obs: Observation) -> _Ctx:
        return _Ctx(
            spec=obs.joker_spec,
            req_pts=obs.required_series_points,
            req_pairs=obs.required_pairs_count,
            table=tuple(tm.meld for tm in obs.table_melds),
        )

    def _seqs(self, obs: Observation, first: int) -> list[tuple[list[Tile], bool]]:
        positions = _draw_positions(obs.draw_pile_size, first)
        pool, bottom = _unseen_pool(obs)
        n = self.num_scenarios if positions else 1
        hazard = opponent_cycle_finish_hazard(obs) if self.version >= 2 else 0.0
        return _sample_seqs(
            pool, bottom, positions, obs.draw_pile_size, n, self.rng, opp_hazard=hazard
        )

    def _pairs_open_value(
        self,
        hand: Sequence[Tile],
        pair_melds: Sequence[Meld],
        ctx: _Ctx,
        seqs: Sequence[tuple[list[Tile], bool]],
    ) -> float:
        used = {pt.tile.tile_id for m in pair_melds for pt in m.placed_tiles}
        rest = [t for t in hand if t.tile_id not in used]
        if len(rest) <= 1:
            return _win_value(len(rest) == 1 and ctx.spec.is_joker(rest[0]), True, True)
        rem0 = len(seqs[0][0]) if seqs else 0
        rest.remove(
            _pick_rollout_discard(rest, OpenType.PAIRS, frozenset(), ctx.table, ctx.spec, rem0)
        )
        return sum(
            _rollout(rest, OpenType.PAIRS, s, ctx, frozenset(), trunc)
            for s, trunc in seqs
        ) / len(seqs)

    def _log_decision(self, kind: str, obs: Observation, payload: dict[str, Any]) -> None:
        if not self.record_decisions:
            return
        me = obs.player_id
        self.decision_log.append(
            {
                "kind": kind,
                "open_type": obs.open_types[me].value,
                "draw_pile_size": obs.draw_pile_size,
                "hand_size": len(obs.hand),
                "has_joker": any(obs.joker_spec.is_joker(t) for t in obs.hand),
                "min_opp_open_hand": min(
                    (
                        obs.hand_sizes[p]
                        for p in range(NUM_PLAYERS)
                        if p != me and obs.open_types[p] != OpenType.NONE
                    ),
                    default=22,
                ),
                **payload,
            }
        )

    def _select_draw(self, obs: Observation, legal: Sequence[Action]) -> Action:
        take = next((a for a in legal if a.action_type == ActionType.TAKE_DISCARD), None)
        draw = next((a for a in legal if a.action_type == ActionType.DRAW_DECK), None)
        if take is None or draw is None:
            return take or draw or legal[0]

        me = obs.player_id
        spec = obs.joker_spec
        top = obs.discard_piles[(me - 1) % NUM_PLAYERS][-1]
        cand = list(obs.hand) + [top]
        ctx = self._ctx(obs)

        if obs.open_types[me] != OpenType.NONE:
            chosen = self._heur._select_draw(obs, legal)
            if self.record_decisions:
                seqs_take = self._seqs(obs, first=NUM_PLAYERS)
                seqs_draw = self._seqs(obs, first=1)
                ot = obs.open_types[me]
                rest_take = _series_reduce(list(cand), list(ctx.table), spec) if ot == OpenType.SERIES else list(cand)
                if len(rest_take) <= 1:
                    v_take = _win_value(len(rest_take) == 1 and spec.is_joker(rest_take[0]), False, ot == OpenType.PAIRS)
                else:
                    rest_take.remove(_pick_rollout_discard(rest_take, ot, frozenset(), ctx.table, spec, len(seqs_take[0][0])))
                    v_take = sum(_rollout(rest_take, ot, s, ctx, frozenset(), tr) for s, tr in seqs_take) / len(seqs_take)
                v_draw = sum(_rollout(obs.hand, ot, s, ctx, frozenset(), tr) for s, tr in seqs_draw) / len(seqs_draw)
                self._log_decision(
                    "draw",
                    obs,
                    {
                        "chosen": chosen.action_type.value,
                        "ev_take": v_take,
                        "ev_draw": v_draw,
                        "ev_advantage": abs(v_draw - v_take),
                    },
                )
            return chosen

        s_sol = solve_best_opening(
            cand, spec, min_points=obs.required_series_points, required_tile_id=top.tile_id
        )
        if s_sol.meets_threshold:
            if self.record_decisions:
                seqs_take = self._seqs(obs, first=NUM_PLAYERS)
                seqs_draw = self._seqs(obs, first=1)
                dw = list(s_sol.deadwood_tiles)
                if len(dw) <= 1:
                    v_take = _win_value(len(dw) == 1 and spec.is_joker(dw[0]), True, False)
                else:
                    dw.remove(_pick_rollout_discard(dw, OpenType.SERIES, frozenset(), ctx.table, spec, len(seqs_take[0][0])))
                    v_take = sum(_rollout(dw, OpenType.SERIES, s, ctx, frozenset(), tr) for s, tr in seqs_take) / len(seqs_take)
                sol0 = solve_best_opening(obs.hand, spec, min_points=obs.required_series_points)
                m0 = frozenset(pt.tile.tile_id for m in sol0.melds for pt in m.placed_tiles)
                v_draw = sum(_rollout(obs.hand, OpenType.NONE, s, ctx, m0, tr) for s, tr in seqs_draw) / len(seqs_draw)
                self._log_decision(
                    "draw",
                    obs,
                    {
                        "chosen": "take_discard",
                        "ev_take": v_take,
                        "ev_draw": v_draw,
                        "ev_advantage": max(0.0, v_draw - v_take),
                    },
                )
            return take

        psol = solve_best_pairs(
            cand, spec, min_pairs=obs.required_pairs_count, required_tile_id=top.tile_id
        )
        if not psol.meets_threshold:
            return draw

        v_take = self._pairs_open_value(cand, psol.pairs, ctx, self._seqs(obs, first=NUM_PLAYERS))
        sol = solve_best_opening(obs.hand, spec, min_points=obs.required_series_points)
        melded = frozenset(pt.tile.tile_id for m in sol.melds for pt in m.placed_tiles)
        seqs_draw = self._seqs(obs, first=1)
        v_draw = sum(
            _rollout(obs.hand, OpenType.NONE, s, ctx, melded, tr) for s, tr in seqs_draw
        ) / len(seqs_draw)
        chosen = take if v_take < v_draw else draw
        self._log_decision(
            "draw",
            obs,
            {
                "chosen": chosen.action_type.value,
                "ev_take": v_take,
                "ev_draw": v_draw,
                "ev_advantage": abs(v_draw - v_take),
            },
        )
        return chosen

    def _select_play(self, obs: Observation, legal: Sequence[Action]) -> Action:
        spec = obs.joker_spec
        hand = obs.hand
        by_type: dict[ActionType, list[Action]] = {}
        for a in legal:
            by_type.setdefault(a.action_type, []).append(a)

        open_series = by_type.get(ActionType.OPEN_SERIES)
        open_pairs = by_type.get(ActionType.OPEN_PAIRS)
        discards = by_type.get(ActionType.DISCARD, [])

        if open_series:
            if open_pairs and len(hand) - sum(m.size for m in open_pairs[0].melds) <= 1:
                self._log_decision("open_choice", obs, {"chosen": "open_pairs", "ev_advantage": 101.0})
                return open_pairs[0]
            joker_used_in_open = any(
                spec.is_joker(pt.tile)
                for m in open_series[0].melds
                for pt in m.placed_tiles
            )
            payload: dict[str, Any] = {
                "chosen": "open_series",
                "series_points": sum(m.points for m in open_series[0].melds),
                "pairs_available": open_pairs is not None,
                "ev_advantage": 80.0,
            }
            if self.record_decisions and joker_used_in_open and discards:
                ctx = self._ctx(obs)
                seqs = self._seqs(obs, first=NUM_PLAYERS)
                used_ids = {pt.tile.tile_id for m in open_series[0].melds for pt in m.placed_tiles}
                dw = [t for t in hand if t.tile_id not in used_ids]
                if len(dw) <= 1:
                    ev_open = _win_value(len(dw) == 1 and spec.is_joker(dw[0]), True, False)
                else:
                    dw_rem = list(dw)
                    dw_rem.remove(
                        _pick_rollout_discard(
                            dw_rem, OpenType.SERIES, frozenset(), list(ctx.table) + list(open_series[0].melds), spec, len(seqs[0][0])
                        )
                    )
                    open_ctx = _Ctx(
                        spec=spec,
                        req_pts=ctx.req_pts,
                        req_pairs=ctx.req_pairs,
                        table=ctx.table + tuple(open_series[0].melds),
                    )
                    ev_open = sum(
                        _rollout(dw_rem, OpenType.SERIES, s, open_ctx, frozenset(), tr)
                        for s, tr in seqs
                    ) / len(seqs)
                _, ev_wait = self._evaluate_discards(obs, discards, ctx, seqs, log=False)
                payload["joker_cf_ev_open"] = ev_open
                payload["joker_cf_ev_wait"] = ev_wait
            self._log_decision("open_choice", obs, payload)
            return open_series[0]

        if open_pairs:
            ctx = self._ctx(obs)
            seqs = self._seqs(obs, first=NUM_PLAYERS)
            sol_s = solve_best_opening(hand, spec, min_points=ctx.req_pts)
            pair_count = len(open_pairs[0].melds)
            if not discards or obs.draw_pile_size <= 12 or pair_count >= 6:
                self._log_decision(
                    "pairs_vs_wait",
                    obs,
                    {
                        "chosen": "open_pairs",
                        "pair_count": pair_count,
                        "best_series_pts": sol_s.total_points,
                        "ev_advantage": 40.0,
                    },
                )
                return open_pairs[0]

            best_discard, v_stay = self._evaluate_discards(obs, discards, ctx, seqs, log=False)
            v_pairs = self._pairs_open_value(hand, open_pairs[0].melds, ctx, seqs)
            margin = 25.0 if self.version == 1 else (25.0 + 30.0 * opponent_cycle_finish_hazard(obs))
            choose_pairs = v_pairs <= v_stay + margin
            self._log_decision(
                "pairs_vs_wait",
                obs,
                {
                    "chosen": "open_pairs" if choose_pairs else "wait",
                    "pair_count": pair_count,
                    "best_series_pts": sol_s.total_points,
                    "ev_pairs": v_pairs,
                    "ev_wait": v_stay,
                    "ev_advantage": abs(v_stay - v_pairs),
                },
            )
            return open_pairs[0] if choose_pairs else best_discard

        if ActionType.MELD in by_type:
            return by_type[ActionType.MELD][0]

        layoffs = by_type.get(ActionType.LAY_OFF)
        if layoffs:
            hand_map = {t.tile_id: t for t in hand}
            if obs.must_use_tile_id is not None:
                forced = [a for a in layoffs if a.tile_id == obs.must_use_tile_id]
                if forced:
                    return forced[0]
            non_j = [a for a in layoffs if not spec.is_joker(hand_map[a.tile_id])]  # type: ignore[index]
            if non_j:
                return max(
                    non_j,
                    key=lambda a: spec.hand_penalty_value(hand_map[a.tile_id]),  # type: ignore[index]
                )
            # Holding a Joker that can lay off: keep it for Okeyle Bitme if <= 2 tiles left!
            if len(hand) >= 3:
                return layoffs[0]

        if discards:
            best, _ = self._evaluate_discards(
                obs, discards, self._ctx(obs), self._seqs(obs, first=NUM_PLAYERS), log=True
            )
            return best
        return legal[0]

    def _evaluate_discards(
        self,
        obs: Observation,
        discards: Sequence[Action],
        ctx: _Ctx,
        seqs: Sequence[tuple[list[Tile], bool]],
        log: bool = False,
    ) -> tuple[Action, float]:
        spec = obs.joker_spec
        hand = obs.hand
        ot = obs.open_types[obs.player_id]
        hand_map = {t.tile_id: t for t in hand}

        safe = [
            a
            for a in discards
            if not (
                spec.is_joker(hand_map[a.tile_id])  # type: ignore[index]
                or _is_layable_anywhere(hand_map[a.tile_id], obs.table_melds, spec)  # type: ignore[index]
            )
        ]
        pool = safe or list(discards)
        if len(pool) == 1:
            return pool[0], 0.0

        visible = _count_visible_keys(obs)
        heur = {
            a.tile_id: score_discard_candidate(hand_map[a.tile_id], hand, obs, visible)  # type: ignore[index]
            for a in pool
        }
        ranked = sorted(pool, key=lambda a: heur[a.tile_id], reverse=True)
        cands = ranked[: self.top_k]
        cand_ids = {a.tile_id for a in cands}

        melded0: frozenset[int] = frozenset()
        counts = _key_counts(hand, spec)
        if ot == OpenType.NONE:
            sol0 = solve_best_opening(hand, spec, min_points=ctx.req_pts)
            melded0 = frozenset(pt.tile.tile_id for m in sol0.melds for pt in m.placed_tiles)
            extra = sorted(
                (a for a in pool if a.tile_id not in melded0 and a.tile_id not in cand_ids),
                key=lambda a: 10.0 * _conn(spec.effective_key(hand_map[a.tile_id]), counts)  # type: ignore[index]
                + 0.3 * spec.hand_penalty_value(hand_map[a.tile_id]),  # type: ignore[index]
            )
        else:
            extra = sorted(
                (a for a in pool if a.tile_id not in cand_ids),
                key=lambda a: spec.hand_penalty_value(hand_map[a.tile_id]),  # type: ignore[index]
                reverse=True,
            )
        cands = cands + extra[:2]

        scored_cands: list[tuple[float, Action, Tile, float]] = []
        for a in cands:
            t = hand_map[a.tile_id]  # type: ignore[index]
            rest = [x for x in hand if x.tile_id != t.tile_id]
            melded = melded0
            if ot == OpenType.NONE and t.tile_id in melded0:
                s = solve_best_opening(rest, spec, min_points=ctx.req_pts)
                melded = frozenset(pt.tile.tile_id for m in s.melds for pt in m.placed_tiles)
            total = 0.0
            for seq, trunc in seqs:
                total += _rollout(rest, ot, seq, ctx, melded, trunc)
            raw_ev = total / len(seqs)
            feed_pen = next_player_feed_penalty(t, obs, visible) if self.version >= 2 else 0.0
            v = raw_ev + feed_pen - self.tiebreak * heur[a.tile_id]
            scored_cands.append((v, a, t, raw_ev))

        scored_cands.sort(key=lambda x: x[0])
        best_v, best_a, best_t, best_raw = scored_cands[0]
        second_v = scored_cands[1][0] if len(scored_cands) > 1 else best_v

        if log and self.record_decisions:
            eff = spec.effective_color_number(best_t)
            rank = eff[1] if eff else 0
            k = spec.effective_key(best_t)
            is_isolated = k >= 0 and _conn(k, counts) == 0.0
            self._log_decision(
                "discard",
                obs,
                {
                    "rank": rank,
                    "is_isolated": is_isolated,
                    "ev_best": best_raw,
                    "ev_advantage": second_v - best_v,
                    "all_cands": [
                        {
                            "rank": (spec.effective_color_number(t)[1] if spec.effective_color_number(t) else 0),
                            "isolated": (spec.effective_key(t) >= 0 and _conn(spec.effective_key(t), counts) == 0.0),
                            "ev": raw,
                        }
                        for _, _, t, raw in scored_cands
                    ],
                },
            )

        return best_a, best_v
