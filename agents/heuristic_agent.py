"""Strong rule-based heuristic baseline agent for Official 101 Okey."""

from __future__ import annotations

from collections import Counter
import random
from typing import Optional, Sequence

from agents.base import Agent
from engine import (
    Action,
    ActionType,
    Observation,
    Okey101Engine,
    OpenType,
    TableMeld,
    TurnPhase,
)
from solver import solve_best_opening, solve_best_pairs, solve_min_deadwood
from tiles import (
    Color,
    JokerSpec,
    NUM_NUMBERS,
    Tile,
    can_lay_off,
    color_number_to_key,
)


def _is_layable_anywhere(
    tile: Tile,
    table_melds: Sequence[TableMeld],
    joker_spec: JokerSpec,
) -> bool:
    for tm in table_melds:
        if can_lay_off(tm.meld, tile, joker_spec):
            return True
    return False


def _count_visible_keys(obs: Observation) -> Counter[int]:
    """Count all visible effective tile keys (0..51) from hand, discards, table, and indicator."""
    counts: Counter[int] = Counter()
    spec = obs.joker_spec
    seen_ids: set[int] = set()

    def _add(t: Tile) -> None:
        if t.tile_id in seen_ids:
            return
        seen_ids.add(t.tile_id)
        k = spec.effective_key(t)
        if k >= 0:
            counts[k] += 1

    for t in obs.hand:
        _add(t)
    for pile in obs.discard_piles:
        for t in pile:
            _add(t)
    for tm in obs.table_melds:
        for pt in tm.meld.placed_tiles:
            _add(pt.tile)
    _add(spec.indicator)
    return counts


def score_discard_candidate(
    tile: Tile,
    hand: Sequence[Tile],
    obs: Observation,
    visible_counts: Optional[Counter[int]] = None,
) -> float:
    """Score a discard candidate (higher = better to discard).

    Factors:
    1. Winning final discard (massive bonus, extra for Joker/Okeyle bitme).
    2. Hard +101 penalty avoidance (real Joker or işlek tile on open table melds).
    3. Hand synergy (keep melds, 2-tile runs/sets, pairs).
    4. Deadwood shedding (discard high isolated deadwood once open; protect 101 potential when unopened).
    5. Tile counting / defensive safety against the next player.
    """
    spec = obs.joker_spec
    is_joker = spec.is_joker(tile)

    # 1. Winning final discard!
    if len(hand) == 1:
        return 20000.0 if is_joker else 10000.0

    score = 0.0

    # 2. Hard +101 penalty avoidance
    if is_joker:
        return -1000.0
    if _is_layable_anywhere(tile, obs.table_melds, spec):
        score -= 350.0

    eff = spec.effective_color_number(tile)
    assert eff is not None
    c, n = eff
    k = color_number_to_key(c, n)

    # Count effective keys of the OTHER tiles in hand
    other_keys: Counter[int] = Counter()
    other_by_num: Counter[int] = Counter()
    num_natural_pairs = 0
    for other in hand:
        if other.tile_id == tile.tile_id:
            continue
        ok = spec.effective_key(other)
        if ok >= 0:
            other_keys[ok] += 1
            other_by_num[(ok % NUM_NUMBERS) + 1] += 1

    for cnt in other_keys.values():
        if cnt >= 2:
            num_natural_pairs += 1

    me = obs.player_id
    open_type = obs.open_types[me]

    # 3. Hand synergy
    synergy = 0.0

    # Identical copy in hand (Pair potential)
    if other_keys[k] >= 1:
        if open_type == OpenType.PAIRS:
            synergy += 80.0
        elif open_type == OpenType.NONE and num_natural_pairs >= 2:
            synergy += 22.0
        else:
            # Duplicate tile in series play is slightly useful for pairs/backup but often redundant
            synergy += 4.0

    # Same-color run neighbors (strictly respecting 1..13, no 12-13-1 wrap!)
    has_m1 = n > 1 and other_keys[color_number_to_key(c, n - 1)] > 0
    has_p1 = n < 13 and other_keys[color_number_to_key(c, n + 1)] > 0
    has_m2 = n > 2 and other_keys[color_number_to_key(c, n - 2)] > 0
    has_p2 = n < 12 and other_keys[color_number_to_key(c, n + 2)] > 0

    if open_type != OpenType.PAIRS:
        if has_m1 and has_p1:
            synergy += 36.0 + 1.5 * n  # Center of a 3-tile run
        elif has_m1 or has_p1:
            # Check if part of a 3+ run on one side
            if (has_m1 and has_m2) or (has_p1 and has_p2):
                synergy += 32.0 + 1.5 * n
            else:
                synergy += 14.0 + 0.6 * n  # 2-tile consecutive run draw
        if has_m2 and not has_m1:
            synergy += 6.0 + 0.25 * n  # Gutshot draw
        if has_p2 and not has_p1:
            synergy += 6.0 + 0.25 * n

        # Same-number different-color set synergy
        distinct_other_colors = sum(
            1
            for oc in Color
            if oc != c and other_keys[color_number_to_key(oc, n)] > 0
        )
        if distinct_other_colors >= 2:
            synergy += 34.0 + 1.8 * n  # Part of a 3- or 4-tile set
        elif distinct_other_colors == 1:
            synergy += 12.0 + 0.6 * n  # 2-tile set draw

    score -= synergy

    # 4. Deadwood / value trade-off
    if open_type == OpenType.SERIES:
        # Already open: aggressively shed high-value deadwood!
        score += 2.2 * n
    elif open_type == OpenType.PAIRS:
        # Pairs open: end-of-round penalty is doubled (x2)! Shed high non-pair tiles!
        score += 4.0 * n
    else:
        # Unopened: high tiles help reach 101 if connected, but isolated high tiles are risky late
        if obs.draw_pile_size <= 8:
            score += 0.8 * n
        else:
            # Early game: prefer discarding isolated low tiles (1..4) over high tiles (9..13)
            score += (7.0 - n) * 0.35

    # 5. Tile counting / defensive safety (avoid feeding live tiles to next player)
    if visible_counts is not None:
        next_p = (me + 1) % 4
        next_danger = 1.5 if obs.open_types[next_p] != OpenType.NONE else 0.7
        if obs.hand_sizes[next_p] <= 5:
            next_danger += 1.0

        # Count how many neighbor/same-number copies are already dead (visible)
        dead_neighbors = 0
        if n > 1:
            dead_neighbors += visible_counts[color_number_to_key(c, n - 1)]
        if n < 13:
            dead_neighbors += visible_counts[color_number_to_key(c, n + 1)]
        for oc in Color:
            if oc != c:
                dead_neighbors += visible_counts[color_number_to_key(oc, n)]

        # More dead neighbors -> safer to discard; fewer dead neighbors -> live/dangerous
        score += dead_neighbors * 0.9 * next_danger

        # If exact duplicate is already visible on table/discard, safer against pairs
        if visible_counts[k] >= 2:
            score += 2.0 * next_danger

    return score


def fast_rollout_action(
    obs: Observation,
    legal_actions: Sequence[Action],
    rng: random.Random,
) -> Action:
    """Lightweight, high-speed heuristic policy for IS-MCTS playouts."""
    if len(legal_actions) == 1:
        return legal_actions[0]

    if obs.phase == TurnPhase.DRAW:
        # Take discard if legal (guaranteed immediately usable by engine rules!)
        for a in legal_actions:
            if a.action_type == ActionType.TAKE_DISCARD:
                return a
        return legal_actions[0]

    me = obs.player_id
    hand = obs.hand
    spec = obs.joker_spec

    # Instant Okeyle bitme check: if 1 tile left and it's a Joker, discard it instead of laying off!
    if len(hand) == 1 and spec.is_joker(hand[0]):
        for a in legal_actions:
            if a.action_type == ActionType.DISCARD:
                return a

    # Prefer OPEN_SERIES > OPEN_PAIRS > MELD > LAY_OFF (except don't lay off last Joker)
    by_type: dict[ActionType, list[Action]] = {}
    for a in legal_actions:
        by_type.setdefault(a.action_type, []).append(a)

    if ActionType.OPEN_SERIES in by_type:
        return by_type[ActionType.OPEN_SERIES][0]
    if ActionType.OPEN_PAIRS in by_type:
        return by_type[ActionType.OPEN_PAIRS][0]
    if ActionType.MELD in by_type:
        return by_type[ActionType.MELD][0]
    if ActionType.LAY_OFF in by_type:
        # Prefer laying off non-joker tiles first (especially if only 2 tiles left with a Joker!)
        layoffs = by_type[ActionType.LAY_OFF]
        hand_map = {t.tile_id: t for t in hand}
        non_joker_layoffs = [
            a for a in layoffs
            if a.tile_id is not None and not spec.is_joker(hand_map[a.tile_id])
        ]
        if non_joker_layoffs:
            return max(
                non_joker_layoffs,
                key=lambda a: spec.hand_penalty_value(hand_map[a.tile_id]),  # type: ignore[index]
            )
        # Only lay off a Joker if we have >= 2 tiles left (otherwise discarding it wins -202!)
        if len(hand) >= 2:
            return layoffs[0]

    # Discard phase: fast score without full visible_counts overhead
    discards = by_type.get(ActionType.DISCARD, list(legal_actions))
    hand_map = {t.tile_id: t for t in hand}
    best_a = discards[0]
    best_s = float("-inf")
    for a in discards:
        assert a.tile_id is not None
        s = score_discard_candidate(hand_map[a.tile_id], hand, obs, visible_counts=None)
        # Tiny tie-break jitter for rollout diversity
        s += rng.random() * 0.05
        if s > best_s:
            best_s = s
            best_a = a
    return best_a


class HeuristicAgent(Agent):
    """Strong rule-based baseline agent for Official 101 Okey."""

    def __init__(self, name: str = "Heuristic", seed: Optional[int] = None) -> None:
        super().__init__(name=name)
        self.rng = random.Random(seed)

    def select_action(
        self,
        obs: Observation,
        legal_actions: Sequence[Action],
        engine_ref: Optional[Okey101Engine] = None,
    ) -> Action:
        if len(legal_actions) == 1:
            return legal_actions[0]

        if obs.phase == TurnPhase.DRAW:
            return self._select_draw(obs, legal_actions)
        return self._select_play(obs, legal_actions)

    def _select_draw(
        self,
        obs: Observation,
        legal_actions: Sequence[Action],
    ) -> Action:
        take_action = next(
            (a for a in legal_actions if a.action_type == ActionType.TAKE_DISCARD),
            None,
        )
        draw_action = next(
            (a for a in legal_actions if a.action_type == ActionType.DRAW_DECK),
            None,
        )
        if take_action is None:
            assert draw_action is not None
            return draw_action
        if draw_action is None:
            return take_action

        me = obs.player_id
        prev_p = (me - 1) % 4
        top_tile = obs.discard_piles[prev_p][-1]
        cand_hand = list(obs.hand) + [top_tile]
        open_type = obs.open_types[me]

        if open_type == OpenType.NONE:
            # Always take if it enables Series/Set opening (>= 101)
            s_sol = solve_best_opening(
                cand_hand,
                obs.joker_spec,
                min_points=obs.required_series_points,
                required_tile_id=top_tile.tile_id,
            )
            if s_sol.meets_threshold:
                return take_action
            # Take for Pairs opening (>= 5 pairs) if deadwood isn't catastrophic or draw pile is low
            p_sol = solve_best_pairs(
                cand_hand,
                obs.joker_spec,
                min_pairs=obs.required_pairs_count,
                required_tile_id=top_tile.tile_id,
            )
            if p_sol.meets_threshold and (
                p_sol.deadwood_penalty <= 85 or obs.draw_pile_size <= 10
            ):
                return take_action
            return draw_action

        if open_type == OpenType.SERIES:
            # Taking for immediate lay-off or new meld strictly reduces hand size / deadwood
            if _is_layable_anywhere(top_tile, obs.table_melds, obs.joker_spec):
                return take_action
            cur_dw = solve_min_deadwood(obs.hand, obs.joker_spec).deadwood_penalty
            new_dw = solve_min_deadwood(
                cand_hand, obs.joker_spec, required_tile_id=top_tile.tile_id
            ).deadwood_penalty
            if new_dw < cur_dw:
                return take_action
            return draw_action

        # OpenType.PAIRS: taking forms a new pair, reducing hand size by 2 -> strictly better
        return take_action

    def _select_play(
        self,
        obs: Observation,
        legal_actions: Sequence[Action],
    ) -> Action:
        me = obs.player_id
        hand = obs.hand
        spec = obs.joker_spec
        by_type: dict[ActionType, list[Action]] = {}
        for a in legal_actions:
            by_type.setdefault(a.action_type, []).append(a)

        # 1. If only 1 tile left and it's a Joker, discard it to win Okeyle bitme (-202, x2)!
        if len(hand) == 1 and spec.is_joker(hand[0]) and ActionType.DISCARD in by_type:
            return by_type[ActionType.DISCARD][0]

        # 2. Opening decisions
        open_series = by_type.get(ActionType.OPEN_SERIES)
        open_pairs = by_type.get(ActionType.OPEN_PAIRS)
        if open_series and open_pairs:
            # Prefer Series unless Pairs wins immediately (0 or 1 tile left)
            pairs_tiles_used = sum(m.size for m in open_pairs[0].melds)
            if len(hand) - pairs_tiles_used <= 1:
                return open_pairs[0]
            return open_series[0]
        if open_series:
            return open_series[0]
        if open_pairs:
            pairs_tiles_used = sum(m.size for m in open_pairs[0].melds)
            # Open pairs if mandatory (must_use_tile), low draw pile, or good pair count
            if (
                obs.must_use_tile_id is not None
                or obs.draw_pile_size <= 12
                or len(open_pairs[0].melds) >= 6
                or len(hand) - pairs_tiles_used <= 7
            ):
                return open_pairs[0]

        # 3. Additional melds when already opened
        if ActionType.MELD in by_type:
            return by_type[ActionType.MELD][0]

        # 4. Aggressive lay-offs (protecting a lone Joker for Okeyle bitme!)
        if ActionType.LAY_OFF in by_type:
            layoffs = by_type[ActionType.LAY_OFF]
            hand_map = {t.tile_id: t for t in hand}
            non_joker_layoffs = [
                a for a in layoffs
                if a.tile_id is not None and not spec.is_joker(hand_map[a.tile_id])
            ]
            if non_joker_layoffs:
                return max(
                    non_joker_layoffs,
                    key=lambda a: spec.hand_penalty_value(hand_map[a.tile_id]),  # type: ignore[index]
                )
            # Only lay off a Joker if >= 2 tiles remain and it doesn't throw away an immediate Okeyle win
            if len(hand) >= 2:
                return layoffs[0]

        # 5. Discard evaluation with full tile counting & safety
        discards = by_type[ActionType.DISCARD]
        hand_map = {t.tile_id: t for t in hand}
        visible_counts = _count_visible_keys(obs)

        best_action = discards[0]
        best_score = float("-inf")
        for a in discards:
            assert a.tile_id is not None
            t = hand_map[a.tile_id]
            s = score_discard_candidate(t, hand, obs, visible_counts=visible_counts)
            if s > best_score:
                best_score = s
                best_action = a
        return best_action

