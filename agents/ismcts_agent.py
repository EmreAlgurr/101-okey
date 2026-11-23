"""Tuned Single-Observer Information Set Monte Carlo Tree Search (SO-ISMCTS) Agent.

Features:
- Strict information-set determinization (`sample_world(obs)`) with 106-tile conservation.
- Top-K discard candidate pruning (default K=6) to eliminate dominated/penalty discards.
- AlphaZero-style PUCT selection with softmax heuristic prior.
- Fast direct 12+ step playouts and exact distance-to-101 + near-meld leaf evaluation.
- Instant fast-path on trivial (single-action) and immediate winning decisions.
"""

from __future__ import annotations

from collections import Counter
import math
import random
from typing import Optional, Sequence

from agents.base import Agent
from agents.heuristic_agent import (
    _count_visible_keys,
    _is_layable_anywhere,
    score_discard_candidate,
)
from engine import (
    Action,
    ActionType,
    GameConfig,
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
    NUM_NUMBERS,
    Tile,
    can_lay_off,
    color_number_to_key,
    create_deck,
)


def action_key(action: Action) -> tuple[object, ...]:
    """Canonical hashable key for an Action across determinizations."""
    atype = action.action_type
    if atype in (ActionType.DRAW_DECK, ActionType.TAKE_DISCARD):
        return (atype,)
    if atype == ActionType.DISCARD:
        return (atype, action.tile_id)
    if atype == ActionType.LAY_OFF:
        return (atype, action.tile_id, action.meld_id, action.lay_off_side)
    tids = tuple(
        sorted(pt.tile.tile_id for m in action.melds for pt in m.placed_tiles)
    )
    return (atype, tids)


def sample_world(
    obs: Observation,
    rng: random.Random,
    katlamali: bool = False,
) -> Okey101Engine:
    """Determinize a consistent 106-tile `Okey101Engine` state from `obs` alone."""
    me = obs.player_id
    full_deck = create_deck()

    known_ids: set[int] = {t.tile_id for t in obs.hand}
    for pile in obs.discard_piles:
        for t in pile:
            known_ids.add(t.tile_id)
    for tm in obs.table_melds:
        for pt in tm.meld.placed_tiles:
            known_ids.add(pt.tile.tile_id)

    unseen = [t for t in full_deck if t.tile_id not in known_ids]

    expected_unseen = (
        sum(obs.hand_sizes[p] for p in range(NUM_PLAYERS) if p != me)
        + obs.draw_pile_size
    )
    if len(unseen) != expected_unseen:
        raise ValueError(
            f"Determinization mismatch: {len(unseen)} unseen tiles vs {expected_unseen} expected."
        )

    ind_id = obs.joker_spec.indicator.tile_id
    indicator_reserved: Optional[Tile] = None
    if obs.draw_pile_size >= 1:
        for i, t in enumerate(unseen):
            if t.tile_id == ind_id:
                indicator_reserved = unseen.pop(i)
                break

    rng.shuffle(unseen)

    world = Okey101Engine(
        config=GameConfig(
            katlamali=katlamali,
            feed_penalty_multiplier=obs.feed_penalty_multiplier,
        )
    )
    world.hands = [[] for _ in range(NUM_PLAYERS)]
    world.hands[me] = list(obs.hand)

    cursor = 0
    for p in range(NUM_PLAYERS):
        if p == me:
            continue
        sz = obs.hand_sizes[p]
        world.hands[p] = unseen[cursor : cursor + sz]
        cursor += sz

    rem_draw = unseen[cursor:]
    if indicator_reserved is not None:
        world.draw_pile = [indicator_reserved] + rem_draw
    else:
        world.draw_pile = rem_draw
    assert len(world.draw_pile) == obs.draw_pile_size

    world.discard_piles = [list(pile) for pile in obs.discard_piles]
    world.table_melds = [tm.copy() for tm in obs.table_melds]
    world._next_meld_id = (
        max((tm.meld_id for tm in world.table_melds), default=-1) + 1
    )
    world.open_types = list(obs.open_types)
    world.penalties = list(obs.penalties)
    world.joker_spec = obs.joker_spec
    world.current_player = obs.current_player
    world.phase = obs.phase
    world.must_use_tile_id = (
        obs.must_use_tile_id if obs.current_player == me else None
    )
    world.opened_before_turn = world.open_types[world.current_player] != OpenType.NONE
    world.highest_series_open = obs.required_series_points - 1
    world.highest_pairs_open = obs.required_pairs_count - 1
    world.done = obs.done
    world.winner = obs.winner
    world.win_flags = obs.win_flags
    world.round_scores = obs.round_scores
    return world


def _find_immediate_win(
    hand: Sequence[Tile],
    legal_actions: Sequence[Action],
    spec: JokerSpec,
) -> Optional[Action]:
    """Return an immediate winning action (0 tiles left) if one exists, preferring Okeyle bitme."""
    # Prefer discarding a lone real Joker (-202, x2) over laying it off (-101, x1)
    if len(hand) == 1:
        discard_win: Optional[Action] = None
        layoff_win: Optional[Action] = None
        for a in legal_actions:
            if a.action_type == ActionType.DISCARD:
                discard_win = a
            elif a.action_type == ActionType.LAY_OFF:
                layoff_win = a
        if discard_win is not None:
            return discard_win
        if layoff_win is not None:
            return layoff_win

    for a in legal_actions:
        if a.action_type in (ActionType.OPEN_SERIES, ActionType.OPEN_PAIRS, ActionType.MELD):
            used = sum(m.size for m in a.melds)
            if used == len(hand):
                return a
    return None


def filter_and_prior_actions(
    obs: Observation,
    legal_actions: Sequence[Action],
    top_k_discards: int = 6,
    temperature: float = 12.0,
    use_visible_counts: bool = True,
) -> tuple[list[Action], dict[tuple[object, ...], float]]:
    """Filter discards to Top-K non-penalty candidates and compute softmax PUCT priors."""
    if len(legal_actions) == 1:
        a = legal_actions[0]
        return [a], {action_key(a): 1.0}

    spec = obs.joker_spec
    hand = obs.hand

    if obs.phase == TurnPhase.DRAW:
        # DRAW_DECK vs TAKE_DISCARD
        me = obs.player_id
        unopened = obs.open_types[me] == OpenType.NONE
        logits: list[tuple[Action, float]] = []
        for a in legal_actions:
            if a.action_type == ActionType.TAKE_DISCARD:
                # Taking discard when unopened guarantees immediate opening -> strong prior
                logits.append((a, 2.2 if unopened else 1.2))
            else:
                logits.append((a, 0.0))
        return _softmax_priors(logits, temperature=1.0)

    # PLAY phase
    non_discards: list[Action] = []
    discards: list[Action] = []
    for a in legal_actions:
        if a.action_type == ActionType.DISCARD:
            discards.append(a)
        else:
            non_discards.append(a)

    hand_map = {t.tile_id: t for t in hand}
    visible_counts = _count_visible_keys(obs) if use_visible_counts else None

    # Score all legal discards
    scored_discards: list[tuple[Action, float, bool]] = []
    for a in discards:
        assert a.tile_id is not None
        t = hand_map[a.tile_id]
        s = score_discard_candidate(t, hand, obs, visible_counts=visible_counts)
        is_penalty = len(hand) > 1 and (
            spec.is_joker(t) or _is_layable_anywhere(t, obs.table_melds, spec)
        )
        scored_discards.append((a, s, is_penalty))

    scored_discards.sort(key=lambda x: x[1], reverse=True)

    # Exclude +101 penalty discards whenever at least 1 non-penalty discard exists
    non_pen = [x for x in scored_discards if not x[2]]
    pool = non_pen if non_pen else scored_discards
    kept_discards = pool[:top_k_discards]

    best_discard_score = kept_discards[0][1] if kept_discards else 0.0
    logits = [(a, s) for a, s, _ in kept_discards]

    # Assign calibrated prior logits to non-discard tactical actions
    for a in non_discards:
        if a.action_type == ActionType.OPEN_SERIES:
            logits.append((a, best_discard_score + 42.0))
        elif a.action_type == ActionType.OPEN_PAIRS:
            bonus = 32.0 if (obs.draw_pile_size <= 14 or len(a.melds) >= 6) else 18.0
            logits.append((a, best_discard_score + bonus))
        elif a.action_type == ActionType.MELD:
            logits.append((a, best_discard_score + 26.0))
        elif a.action_type == ActionType.LAY_OFF:
            assert a.tile_id is not None
            t = hand_map[a.tile_id]
            if spec.is_joker(t):
                # Keep Joker lay-off lower priority if close to Okeyle bitme
                bonus = -5.0 if len(hand) <= 2 else 10.0
            else:
                bonus = 18.0 + 0.5 * spec.hand_penalty_value(t)
            logits.append((a, best_discard_score + bonus))

    return _softmax_priors(logits, temperature=temperature)


def _softmax_priors(
    action_logits: Sequence[tuple[Action, float]],
    temperature: float,
) -> tuple[list[Action], dict[tuple[object, ...], float]]:
    max_l = max(l for _, l in action_logits)
    inv_t = 1.0 / max(1e-6, temperature)
    exps = [math.exp(max(-30.0, (l - max_l) * inv_t)) for _, l in action_logits]
    total = sum(exps)
    actions: list[Action] = []
    priors: dict[tuple[object, ...], float] = {}
    for (a, _), e in zip(action_logits, exps):
        actions.append(a)
        priors[action_key(a)] = e / total
    return actions, priors


def _near_meld_synergy(tiles: Sequence[Tile], spec: JokerSpec) -> float:
    """Estimate near-meld potential (2-tile runs, gutshots, 2-tile sets, pairs) on deadwood."""
    counts: Counter[int] = Counter()
    jokers = 0
    for t in tiles:
        k = spec.effective_key(t)
        if k == -1:
            jokers += 1
        else:
            counts[k] += 1

    potential = 0.0
    # 2-tile same-color adjacent runs & gutshots
    for c in Color:
        for n in range(1, 13):
            k1 = color_number_to_key(c, n)
            k2 = color_number_to_key(c, n + 1)
            if counts[k1] > 0 and counts[k2] > 0:
                potential += 0.45 * (2 * n + 1)
            elif n <= 11:
                k3 = color_number_to_key(c, n + 2)
                if counts[k1] > 0 and counts[k3] > 0:
                    potential += 0.20 * (2 * n + 2)

    # 2-tile same-number distinct-color sets
    for n in range(1, 14):
        present = sum(1 for c in Color if counts[color_number_to_key(c, n)] > 0)
        if present == 2:
            potential += 0.40 * (2 * n)

    # Unused real jokers in deadwood carry huge flexibility toward future melds
    potential += jokers * 18.0
    return potential


def _fast_hand_meld_estimate(
    hand: Sequence[Tile],
    spec: JokerSpec,
) -> tuple[float, int, float]:
    """O(N) estimate of (meld_points, pair_count, raw_penalty_sum) for fast opponent eval."""
    counts: Counter[int] = Counter()
    jokers = 0
    raw_penalty = 0.0
    for t in hand:
        k = spec.effective_key(t)
        if k == -1:
            jokers += 1
            raw_penalty += 101.0
        else:
            counts[k] += 1
            raw_penalty += (k % NUM_NUMBERS) + 1

    natural_pairs = sum(1 for c in counts.values() if c >= 2)
    singletons = sum(1 for c in counts.values() if c == 1)
    est_pairs = natural_pairs + min(jokers, singletons)

    rem = dict(counts)
    est_pts = 0.0

    for color in Color:
        n = 1
        while n <= 13:
            k = color_number_to_key(color, n)
            if rem.get(k, 0) > 0:
                run_start = n
                while n <= 13 and rem.get(color_number_to_key(color, n), 0) > 0:
                    n += 1
                if n - run_start >= 3:
                    for x in range(run_start, n):
                        rem[color_number_to_key(color, x)] -= 1
                        est_pts += x
            else:
                n += 1

    for n in range(13, 0, -1):
        present = [c for c in Color if rem.get(color_number_to_key(c, n), 0) > 0]
        if len(present) >= 3:
            for c in present:
                rem[color_number_to_key(c, n)] -= 1
            est_pts += n * len(present)

    if jokers > 0:
        est_pts += jokers * 24.0

    return est_pts, est_pairs, raw_penalty


def evaluate_world(
    world: Okey101Engine,
    observer_id: Optional[int] = None,
) -> list[float]:
    """Return normalized value vector in [-1, 1] for all 4 players (higher = better).

    For `observer_id`, uses exact `solver.py` distance-to-101 + near-meld synergy
    and draw-pile pressure so MCTS accurately values opening progress and deadwood.
    """
    if world.done and world.round_scores is not None:
        return [
            max(-1.0, min(1.0, -s / 300.0)) for s in world.round_scores
        ]

    spec = world.joker_spec
    draw_left = len(world.draw_pile)
    draw_ratio = draw_left / 21.0
    # Draw-pile pressure: as deck shrinks (especially <= 12 tiles), unopened +202 risk rises sharply
    deck_pressure = 1.0 - draw_ratio
    late_pressure = max(0.0, (12.0 - draw_left) / 12.0)
    values: list[float] = []

    for p in range(NUM_PLAYERS):
        hand = world.hands[p]
        ot = world.open_types[p]

        if p == observer_id:
            if ot == OpenType.NONE:
                req_pts = max(1, world.required_series_points)
                req_pairs = max(1, world.required_pairs_count)
                op_sol = solve_best_opening(hand, spec, min_points=req_pts)
                pr_sol = solve_best_pairs(hand, spec, min_pairs=req_pairs)

                near_syn = _near_meld_synergy(op_sol.deadwood_tiles, spec)
                effective_pts = op_sol.total_points + near_syn * (0.35 + 0.65 * draw_ratio)
                series_prog = min(1.0, effective_pts / req_pts)
                pairs_prog = min(1.0, pr_sol.pair_count / req_pairs)

                if op_sol.meets_threshold or pr_sol.meets_threshold:
                    # Ready to open! Strongly prefer over staying unopened
                    open_prob = 0.94
                    est_deadwood = (
                        op_sol.deadwood_penalty
                        if op_sol.meets_threshold
                        else 2.0 * pr_sol.deadwood_penalty
                    )
                else:
                    # Harsher decay on opening probability when draw pile drops below 12 tiles
                    time_factor = (0.15 + 0.85 * draw_ratio) * (1.0 - 0.35 * late_pressure)
                    open_prob = max(
                        0.0,
                        min(0.85, (max(series_prog, pairs_prog) ** 1.45) * time_factor),
                    )
                    est_deadwood = max(14.0, op_sol.deadwood_penalty * 0.70)

                # Harsher unopened risk as deck shrinks (strongly pushes MCTS to cross 101!)
                unopened_risk = 202.0 * (1.0 + 0.25 * deck_pressure + 0.35 * late_pressure)
                est_hand_pen = (1.0 - open_prob) * unopened_risk + open_prob * est_deadwood
                # Direct gradient toward higher exact meld sum + near-meld synergy
                est_hand_pen -= min(req_pts, op_sol.total_points) * 0.35 + near_syn * 0.25

            elif ot == OpenType.SERIES:
                if len(hand) == 0:
                    est_hand_pen = -101.0
                elif len(hand) == 1:
                    est_hand_pen = -95.0 if spec.is_joker(hand[0]) else -65.0
                else:
                    dw_sol = solve_min_deadwood(hand, spec)
                    layable_relief = sum(
                        spec.hand_penalty_value(t) * 0.8
                        for t in dw_sol.deadwood_tiles
                        if _is_layable_anywhere(t, world.table_melds, spec)
                    )
                    est_hand_pen = max(0.0, dw_sol.deadwood_penalty - layable_relief)
                    if len(hand) <= 4:
                        est_hand_pen -= (5 - len(hand)) * 14.0

            else:  # OpenType.PAIRS
                if len(hand) == 0:
                    est_hand_pen = -202.0
                else:
                    pr_sol = solve_best_pairs(hand, spec, min_pairs=1)
                    est_hand_pen = 2.0 * pr_sol.deadwood_penalty
                    if len(hand) <= 3:
                        est_hand_pen -= (4 - len(hand)) * 20.0
        else:
            # Fast O(N) evaluation for opponents
            est_pts, est_pairs, raw_pen = _fast_hand_meld_estimate(hand, spec)
            if ot == OpenType.NONE:
                series_prog = min(1.0, est_pts / max(1, world.required_series_points))
                pairs_prog = min(1.0, est_pairs / max(1, world.required_pairs_count))
                open_prob = min(0.92, max(series_prog, pairs_prog) * (0.35 + 0.65 * draw_ratio))
                est_deadwood = max(10.0, raw_pen - est_pts)
                est_hand_pen = (1.0 - open_prob) * 202.0 + open_prob * est_deadwood
            elif ot == OpenType.SERIES:
                if len(hand) == 0:
                    est_hand_pen = -101.0
                elif len(hand) == 1:
                    est_hand_pen = -85.0 if spec.is_joker(hand[0]) else -55.0
                else:
                    est_hand_pen = max(0.0, raw_pen - 0.85 * est_pts)
                    if len(hand) <= 4:
                        est_hand_pen -= (5 - len(hand)) * 12.0
            else:
                if len(hand) == 0:
                    est_hand_pen = -202.0
                else:
                    est_hand_pen = 2.0 * max(0.0, raw_pen - 0.7 * est_pts)

        total_est_score = world.penalties[p] + est_hand_pen
        values.append(max(-1.0, min(1.0, -total_est_score / 300.0)))

    return values


def _fast_rollout_step(world: Okey101Engine, rng: random.Random) -> None:
    """Execute one fast, strictly legal heuristic step directly on `world`."""
    p = world.current_player
    spec = world.joker_spec

    if world.phase == TurnPhase.DRAW:
        # Check cheap TAKE_DISCARD conditions first (avoid expensive B&B on every rollout draw)
        prev_p = (p - 1) % NUM_PLAYERS
        if world.discard_piles[prev_p]:
            top_t = world.discard_piles[prev_p][-1]
            ot = world.open_types[p]
            if ot == OpenType.SERIES and world.can_tile_lay_off_anywhere(top_t):
                world.step(Action(ActionType.TAKE_DISCARD))
                return
            if ot == OpenType.NONE and world.can_take_discard(p):
                world.step(Action(ActionType.TAKE_DISCARD))
                return
        if world.draw_pile:
            world.step(Action(ActionType.DRAW_DECK))
            return
        if world.can_take_discard(p):
            world.step(Action(ActionType.TAKE_DISCARD))
        return

    # PLAY phase
    hand = world.hands[p]
    ot = world.open_types[p]

    # 1. Immediate winning discard (especially lone Joker -> Okeyle bitme)
    if len(hand) == 1 and world.must_use_tile_id is None:
        world.step(Action(ActionType.DISCARD, tile_id=hand[0].tile_id))
        return

    # 2. Try opening if unopened
    if ot == OpenType.NONE:
        # Quick lower-bound guard: only invoke solver if hand could reach 101 or 5 pairs
        raw_sum = sum(13 if spec.is_joker(t) else (spec.effective_value(t) or 0) for t in hand)
        if raw_sum >= world.required_series_points:
            sol = solve_best_opening(
                hand,
                spec,
                min_points=world.required_series_points,
                required_tile_id=world.must_use_tile_id,
            )
            if sol.meets_threshold:
                world.step(Action(ActionType.OPEN_SERIES, melds=tuple(sol.melds)))
                return
        psol = solve_best_pairs(
            hand,
            spec,
            min_pairs=world.required_pairs_count,
            required_tile_id=world.must_use_tile_id,
        )
        if psol.meets_threshold and (
            world.must_use_tile_id is not None
            or len(world.draw_pile) <= 12
            or psol.pair_count >= 6
        ):
            world.step(Action(ActionType.OPEN_PAIRS, melds=tuple(psol.pairs)))
            return

    elif ot == OpenType.SERIES:
        # Lay off non-joker tiles aggressively
        best_layoff: Optional[Action] = None
        best_val = -1
        for t in hand:
            if spec.is_joker(t) and len(hand) <= 2:
                continue
            for tm in world.table_melds:
                sides = can_lay_off(tm.meld, t, spec)
                if sides:
                    val = spec.hand_penalty_value(t)
                    if val > best_val:
                        best_val = val
                        best_layoff = Action(
                            ActionType.LAY_OFF,
                            tile_id=t.tile_id,
                            meld_id=tm.meld_id,
                            lay_off_side=sides[0],
                        )
        if best_layoff is not None:
            world.step(best_layoff)
            return

        # New melds from hand
        if len(hand) >= 3:
            dw = solve_min_deadwood(hand, spec, required_tile_id=world.must_use_tile_id)
            if dw.melds:
                world.step(Action(ActionType.MELD, melds=tuple(dw.melds)))
                return

    elif ot == OpenType.PAIRS:
        psol = solve_best_pairs(
            hand, spec, min_pairs=1, required_tile_id=world.must_use_tile_id
        )
        if psol.pairs:
            world.step(Action(ActionType.MELD, melds=tuple(psol.pairs)))
            return

    # 3. Fast safe discard selection (avoids +101 işlek / Joker penalties)
    if world.must_use_tile_id is not None:
        # Fallback if must_use_tile_id still needs to be placed
        legal = world.get_legal_actions()
        if legal:
            world.step(legal[0])
        return

    obs_proxy = world.get_observation(p)
    best_t = hand[0]
    best_s = float("-inf")
    for t in hand:
        s = score_discard_candidate(t, hand, obs_proxy, visible_counts=None) + rng.random() * 0.1
        if s > best_s:
            best_s = s
            best_t = t
    world.step(Action(ActionType.DISCARD, tile_id=best_t.tile_id))


class _PUCTNode:
    """Tree node for Single-Observer IS-MCTS with PUCT priors."""

    __slots__ = (
        "parent",
        "action_key",
        "prior",
        "children",
        "visits",
        "availability",
        "total_values",
    )

    def __init__(
        self,
        parent: Optional[_PUCTNode] = None,
        action_key: Optional[tuple[object, ...]] = None,
        prior: float = 1.0,
    ) -> None:
        self.parent = parent
        self.action_key = action_key
        self.prior = prior
        self.children: dict[tuple[object, ...], _PUCTNode] = {}
        self.visits: int = 0
        self.availability: int = 0
        self.total_values: list[float] = [0.0, 0.0, 0.0, 0.0]

    def select_puct_child(
        self,
        legal_keys: Sequence[tuple[object, ...]],
        priors: dict[tuple[object, ...], float],
        player_id: int,
        c_puct: float,
    ) -> tuple[tuple[object, ...], Optional[_PUCTNode]]:
        """Select legal action key maximizing PUCT for `player_id`.

        Returns (chosen_key, existing_child_or_None).
        """
        parent_fpu = (
            (self.total_values[player_id] / self.visits) - 0.05
            if self.visits > 0
            else 0.0
        )
        sqrt_parent = math.sqrt(max(1, self.visits))

        best_key = legal_keys[0]
        best_child: Optional[_PUCTNode] = None
        best_score = float("-inf")

        for k in legal_keys:
            child = self.children.get(k)
            p_sa = priors.get(k, 1.0 / len(legal_keys))
            if child is None:
                q_sa = parent_fpu
                u_sa = c_puct * p_sa * sqrt_parent
            else:
                q_sa = child.total_values[player_id] / max(1, child.visits)
                sqrt_avail = math.sqrt(max(1, child.availability))
                u_sa = c_puct * child.prior * sqrt_avail / (1 + child.visits)
            score = q_sa + u_sa
            if score > best_score:
                best_score = score
                best_key = k
                best_child = child

        return best_key, best_child


class ISMCTSAgent(Agent):
    """Tuned Single-Observer IS-MCTS agent with Top-K pruning and PUCT selection."""

    def __init__(
        self,
        iterations: int = 120,
        c_puct: float = 1.41,
        rollout_depth: int = 12,
        top_k_discards: int = 6,
        prior_temperature: float = 12.0,
        katlamali: bool = False,
        name: str = "ISMCTS",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=name)
        self.iterations = iterations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.top_k_discards = top_k_discards
        self.prior_temperature = prior_temperature
        self.katlamali = katlamali
        self.rng = random.Random(seed)

    def select_action(
        self,
        obs: Observation,
        legal_actions: Sequence[Action],
        engine_ref: Optional[Okey101Engine] = None,
    ) -> Action:
        """Run PUCT SO-ISMCTS strictly using `obs` (ignores `engine_ref`)."""
        del engine_ref  # Never peek at hidden state!

        # 1. Trivial single-action fast path
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 2. Immediate winning move fast path (0 tiles left)
        if obs.phase == TurnPhase.PLAY:
            win_action = _find_immediate_win(obs.hand, legal_actions, obs.joker_spec)
            if win_action is not None:
                return win_action

            # 2b. Late-deck opening fast path (avoid eating +202 when draw_pile_size <= 10)
            if obs.open_types[obs.player_id] == OpenType.NONE and obs.draw_pile_size <= 10:
                open_series = next(
                    (a for a in legal_actions if a.action_type == ActionType.OPEN_SERIES),
                    None,
                )
                if open_series is not None:
                    return open_series
                open_pairs = next(
                    (a for a in legal_actions if a.action_type == ActionType.OPEN_PAIRS),
                    None,
                )
                if open_pairs is not None:
                    return open_pairs

        # 3. Root Top-K candidate pruning & PUCT prior computation
        root_actions, root_priors = filter_and_prior_actions(
            obs,
            legal_actions,
            top_k_discards=self.top_k_discards,
            temperature=self.prior_temperature,
            use_visible_counts=True,
        )
        if len(root_actions) == 1:
            return root_actions[0]

        root = _PUCTNode()
        root_key_to_action: dict[tuple[object, ...], Action] = {
            action_key(a): a for a in root_actions
        }
        root_keys = list(root_key_to_action.keys())
        observer_id = obs.player_id

        for _ in range(self.iterations):
            world = sample_world(obs, self.rng, katlamali=self.katlamali)
            node = root

            # --- Selection & Expansion ---
            depth = 0
            max_tree_depth = 6
            while not world.done and depth < max_tree_depth:
                if node is root:
                    key_map = root_key_to_action
                    keys = root_keys
                    priors = root_priors
                else:
                    legal = world.get_legal_actions()
                    if not legal:
                        break
                    cur_obs = world.get_observation(world.current_player)
                    filtered, priors = filter_and_prior_actions(
                        cur_obs,
                        legal,
                        top_k_discards=self.top_k_discards,
                        temperature=self.prior_temperature,
                        use_visible_counts=False,
                    )
                    key_map = {action_key(a): a for a in filtered}
                    keys = list(key_map.keys())

                # Update availability for all currently legal existing children
                for k in keys:
                    if k in node.children:
                        node.children[k].availability += 1

                chosen_key, existing_child = node.select_puct_child(
                    keys, priors, world.current_player, self.c_puct
                )

                if existing_child is None:
                    # Expand newly selected high-PUCT action
                    child = _PUCTNode(
                        parent=node,
                        action_key=chosen_key,
                        prior=priors.get(chosen_key, 1.0 / len(keys)),
                    )
                    child.availability = 1
                    node.children[chosen_key] = child
                    world.step(key_map[chosen_key])
                    node = child
                    break

                world.step(key_map[chosen_key])
                node = existing_child
                depth += 1

            # --- Depth-limited fast heuristic playout ---
            rollout_steps = 0
            while not world.done and rollout_steps < self.rollout_depth:
                _fast_rollout_step(world, self.rng)
                rollout_steps += 1

            # --- Exact observer leaf evaluation & backpropagation ---
            eval_vec = evaluate_world(world, observer_id=observer_id)
            cur_node: Optional[_PUCTNode] = node
            while cur_node is not None:
                cur_node.visits += 1
                for i in range(NUM_PLAYERS):
                    cur_node.total_values[i] += eval_vec[i]
                cur_node = cur_node.parent

        # Pick root action with highest visit count (tie-break by mean value)
        best_key = max(
            (k for k in root_keys if k in root.children),
            key=lambda k: (
                root.children[k].visits,
                root.children[k].total_values[observer_id]
                / max(1, root.children[k].visits),
            ),
        )
        return root_key_to_action[best_key]
