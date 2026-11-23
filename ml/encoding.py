"""State and Action tensor encoding for Official 101 Okey Neural Policy-Value models."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from engine import (
    Action,
    ActionType,
    NUM_PLAYERS,
    Observation,
    OpenType,
    TurnPhase,
)
from solver import solve_best_opening, solve_best_pairs, solve_min_deadwood
from tiles import (
    NUM_KINDS,
    Tile,
    can_lay_off,
    color_number_to_key,
)

# ---------------------------------------------------------------------------
# Discrete Action Space Layout (61 total)
# ---------------------------------------------------------------------------
# 0..51 : Discard standard tile (color * 13 + (number - 1))
# 52    : Discard False Joker (Sahte Okey)
# 53    : Discard Real Joker (Okey)
# 54    : DRAW_DECK
# 55    : TAKE_DISCARD
# 56    : OPEN_SERIES
# 57    : OPEN_PAIRS
# 58    : LAY_OFF
# 59    : MELD
# 60    : PASS (reserved)

IDX_DISCARD_FALSE_JOKER = 52
IDX_DISCARD_REAL_JOKER = 53
IDX_DRAW_DECK = 54
IDX_TAKE_DISCARD = 55
IDX_OPEN_SERIES = 56
IDX_OPEN_PAIRS = 57
IDX_LAY_OFF = 58
IDX_MELD = 59
IDX_PASS = 60
ACTION_DIM = 61

# ---------------------------------------------------------------------------
# State Vector Layout (400 float32 features)
# ---------------------------------------------------------------------------
# 7 tile planes x 52 = 364:
#   0: own hand non-joker effective counts (/2.0)
#   1: visible discard non-joker effective counts (/2.0)
#   2: open table meld non-joker effective counts (/2.0)
#   3: active Joker one-hot (52)
#   4: indicator tile one-hot (52)
#   5: previous player's top discard one-hot (52)
#   6: own hand tiles currently layable (islek) on table melds (52)
# Context & Solver scalars = 36:
#   5 joker/special scalars: own_real_jokers/2, own_false_jokers/2,
#       discard_real_jokers/2, table_real_jokers/2, prev_discard_is_joker
#   1 draw pile scalar: draw_pile_size / 21.0
#   20 relative player scalars (4 players x 5: open_none, open_series, open_pairs,
#       hand_size/22.0, penalties/202.0)
#   2 turn phase one-hot (DRAW, PLAY)
#   3 rule scalars: req_series/101.0, req_pairs/5.0, must_use_active
#   5 solver auxiliary features: best_open_points/101.0, best_pairs_count/5.0,
#       min_deadwood/200.0, can_open_series, can_open_pairs
STATE_DIM = 7 * NUM_KINDS + 36  # 400


def normalize_score(score: float) -> float:
    """Normalize a 101 Okey round score (lower is better) to [-1.0, 1.0] (higher is better)."""
    return float(max(-1.0, min(1.0, -score / 250.0)))


def encode_obs(obs: Observation) -> np.ndarray:
    """Encode an information-set `Observation` into a fixed-size float32 vector."""
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    spec = obs.joker_spec
    me = obs.player_id

    # Offsets for the 7 tile planes
    off_hand = 0
    off_disc = NUM_KINDS
    off_table = 2 * NUM_KINDS
    off_joker = 3 * NUM_KINDS
    off_ind = 4 * NUM_KINDS
    off_prev_disc = 5 * NUM_KINDS
    off_islek = 6 * NUM_KINDS
    off_scalars = 7 * NUM_KINDS

    # Plane 0: Own hand
    own_real_jokers = 0
    own_false_jokers = 0
    for t in obs.hand:
        if spec.is_joker(t):
            own_real_jokers += 1
        else:
            if t.is_false_joker:
                own_false_jokers += 1
            k = spec.effective_key(t)
            vec[off_hand + k] += 0.5
            # Plane 6: Check if tile is currently layable (işlek) on any open meld
            if obs.table_melds:
                for tm in obs.table_melds:
                    if can_lay_off(tm.meld, t, spec):
                        vec[off_islek + k] = 1.0
                        break

    # Plane 1: Visible discards across all players
    discard_real_jokers = 0
    for pile in obs.discard_piles:
        for t in pile:
            if spec.is_joker(t):
                discard_real_jokers += 1
            else:
                vec[off_disc + spec.effective_key(t)] += 0.5

    # Plane 2: Tiles locked in open table melds
    table_real_jokers = 0
    for tm in obs.table_melds:
        for pt in tm.meld.placed_tiles:
            if spec.is_joker(pt.tile):
                table_real_jokers += 1
            else:
                vec[off_table + spec.effective_key(pt.tile)] += 0.5

    # Plane 3 & 4: Active Joker & Indicator one-hot
    vec[off_joker + spec.joker_key] = 1.0
    ind = spec.indicator
    if ind.color is not None:
        vec[off_ind + color_number_to_key(ind.color, ind.number)] = 1.0

    # Plane 5: Previous player's top discard
    prev_p = (me - 1) % NUM_PLAYERS
    prev_discard_is_joker = 0.0
    if obs.discard_piles[prev_p]:
        top_t = obs.discard_piles[prev_p][-1]
        if spec.is_joker(top_t):
            prev_discard_is_joker = 1.0
        else:
            vec[off_prev_disc + spec.effective_key(top_t)] = 1.0

    # Context scalars
    idx = off_scalars
    vec[idx] = own_real_jokers / 2.0
    vec[idx + 1] = own_false_jokers / 2.0
    vec[idx + 2] = discard_real_jokers / 2.0
    vec[idx + 3] = table_real_jokers / 2.0
    vec[idx + 4] = prev_discard_is_joker
    vec[idx + 5] = obs.draw_pile_size / 21.0
    idx += 6

    # Seat-relative player status (0=me, 1=next, 2=across, 3=prev)
    for rel in range(NUM_PLAYERS):
        p = (me + rel) % NUM_PLAYERS
        ot = obs.open_types[p]
        vec[idx + 0] = 1.0 if ot == OpenType.NONE else 0.0
        vec[idx + 1] = 1.0 if ot == OpenType.SERIES else 0.0
        vec[idx + 2] = 1.0 if ot == OpenType.PAIRS else 0.0
        vec[idx + 3] = obs.hand_sizes[p] / 22.0
        vec[idx + 4] = min(3.0, obs.penalties[p] / 202.0)
        idx += 5

    # Turn phase one-hot & rule scalars
    vec[idx] = 1.0 if obs.phase == TurnPhase.DRAW else 0.0
    vec[idx + 1] = 1.0 if obs.phase == TurnPhase.PLAY else 0.0
    vec[idx + 2] = obs.required_series_points / 101.0
    vec[idx + 3] = obs.required_pairs_count / 5.0
    vec[idx + 4] = 1.0 if obs.must_use_tile_id is not None else 0.0
    idx += 5

    # Solver auxiliary features
    hand = obs.hand
    my_ot = obs.open_types[me]
    if my_ot == OpenType.NONE:
        op_sol = solve_best_opening(hand, spec, min_points=obs.required_series_points)
        pr_sol = solve_best_pairs(hand, spec, min_pairs=obs.required_pairs_count)
        best_open_pts = float(op_sol.total_points)
        best_pairs_cnt = float(pr_sol.pair_count)
        min_dw = float(op_sol.deadwood_penalty)
        can_open_series = 1.0 if op_sol.meets_threshold else 0.0
        can_open_pairs = 1.0 if pr_sol.meets_threshold else 0.0
    elif my_ot == OpenType.SERIES:
        dw_sol = solve_min_deadwood(hand, spec)
        best_open_pts = float(dw_sol.total_meld_points)
        best_pairs_cnt = 0.0
        min_dw = float(dw_sol.deadwood_penalty)
        can_open_series = 0.0
        can_open_pairs = 0.0
    else:  # OpenType.PAIRS
        pr_sol = solve_best_pairs(hand, spec, min_pairs=1)
        best_open_pts = 0.0
        best_pairs_cnt = float(pr_sol.pair_count)
        min_dw = float(pr_sol.deadwood_penalty)
        can_open_series = 0.0
        can_open_pairs = 0.0

    vec[idx] = min(2.5, best_open_pts / 101.0)
    vec[idx + 1] = min(2.5, best_pairs_cnt / 5.0)
    vec[idx + 2] = min(2.5, min_dw / 200.0)
    vec[idx + 3] = can_open_series
    vec[idx + 4] = can_open_pairs

    return vec


def action_to_index(action: Action, obs: Observation) -> int:
    """Map a legal `Action` to its discrete index in `0..ACTION_DIM-1`."""
    atype = action.action_type
    if atype == ActionType.DRAW_DECK:
        return IDX_DRAW_DECK
    if atype == ActionType.TAKE_DISCARD:
        return IDX_TAKE_DISCARD
    if atype == ActionType.OPEN_SERIES:
        return IDX_OPEN_SERIES
    if atype == ActionType.OPEN_PAIRS:
        return IDX_OPEN_PAIRS
    if atype == ActionType.LAY_OFF:
        return IDX_LAY_OFF
    if atype == ActionType.MELD:
        return IDX_MELD
    if atype == ActionType.DISCARD:
        assert action.tile_id is not None
        tile = next(t for t in obs.hand if t.tile_id == action.tile_id)
        if obs.joker_spec.is_joker(tile):
            return IDX_DISCARD_REAL_JOKER
        if tile.is_false_joker:
            return IDX_DISCARD_FALSE_JOKER
        assert tile.color is not None
        return color_number_to_key(tile.color, tile.number)
    raise ValueError(f"Unsupported action type: {atype}")


def legal_action_mask(
    obs: Observation,
    legal_actions: Sequence[Action],
) -> np.ndarray:
    """Return a boolean mask of shape `(ACTION_DIM,)` marking legal action indices."""
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    for a in legal_actions:
        mask[action_to_index(a, obs)] = True
    return mask


def index_to_action(
    idx: int,
    obs: Observation,
    legal_actions: Sequence[Action],
) -> Action:
    """Resolve a discrete action index back to a concrete legal `Action`.

    If multiple `LAY_OFF` actions map to `IDX_LAY_OFF`, prefers laying off the
    highest-value non-joker tile (protecting a lone Joker for Okeyle bitme).
    """
    matches = [a for a in legal_actions if action_to_index(a, obs) == idx]
    if not matches:
        raise ValueError(f"Action index {idx} is not legal in current state.")
    if len(matches) == 1:
        return matches[0]
    if idx == IDX_LAY_OFF:
        hand_map = {t.tile_id: t for t in obs.hand}
        spec = obs.joker_spec
        non_jokers = [
            a for a in matches
            if a.tile_id is not None and not spec.is_joker(hand_map[a.tile_id])
        ]
        pool = non_jokers if non_jokers else matches
        return max(
            pool,
            key=lambda a: spec.hand_penalty_value(hand_map[a.tile_id]),  # type: ignore[index]
        )
    return matches[0]

