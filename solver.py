"""Exact Branch & Bound / Backtracking Hand Solver for Official 101 Okey.

Computes:
1. Highest-point valid opening melds (>= 101 check, or custom katlamali threshold).
2. Exact maximum pairs count (>= 5 check, or custom katlamali threshold).
3. Minimum deadwood melds (for already-opened hands / finishing checks).

Handles 0, 1, or 2 Jokers (and False Jokers) exactly in well under 50ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from tiles import (
    ALL_COLORS,
    Color,
    JokerSpec,
    Meld,
    MeldType,
    NUM_COLORS,
    NUM_NUMBERS,
    PlacedTile,
    Tile,
    color_number_to_key,
    key_to_color_number,
)


@dataclass(frozen=True, slots=True)
class _CandidateMeld:
    """Compact internal representation of a candidate Run or Set."""

    meld_type: MeldType
    # Tuple of (local_idx, count_used) — count_used is always 1 for runs/sets
    local_indices: tuple[int, ...]
    jokers_used: int
    points: int
    # Exact slot assignment for reconstruction: tuple of (color, number, local_idx_or_-1_for_joker)
    slots: tuple[tuple[Color, int, int], ...]
    # Bitmask of local indices used (for fast O(1) checks)
    mask: int
    # Non-joker face value sum (used for min-deadwood objective)
    non_joker_points: int


@dataclass(slots=True)
class OpeningSolution:
    """Result of solving for highest-point opening melds."""

    melds: list[Meld]
    total_points: int
    deadwood_tiles: list[Tile]
    deadwood_penalty: int
    meets_threshold: bool


@dataclass(slots=True)
class PairsSolution:
    """Result of solving for maximum pairs."""

    pairs: list[Meld]
    pair_count: int
    deadwood_tiles: list[Tile]
    deadwood_penalty: int
    meets_threshold: bool


@dataclass(slots=True)
class DeadwoodSolution:
    """Result of solving for minimum deadwood melds."""

    melds: list[Meld]
    total_meld_points: int
    deadwood_tiles: list[Tile]
    deadwood_penalty: int


def _prepare_hand(
    hand: Sequence[Tile],
    joker_spec: JokerSpec,
    required_tile_id: Optional[int] = None,
) -> tuple[
    list[int],              # unique_keys (0..51)
    list[list[Tile]],       # tiles_by_local[i]
    list[Tile],             # joker_tiles
    Optional[int],          # req_local_idx (-1 if required tile is a joker, None if none)
]:
    """Group hand tiles by effective key (0..51) and separate real Jokers."""
    buckets: dict[int, list[Tile]] = {}
    joker_tiles: list[Tile] = []
    req_key: Optional[int] = None
    req_is_joker = False

    for t in hand:
        k = joker_spec.effective_key(t)
        if k == -1:
            joker_tiles.append(t)
            if required_tile_id is not None and t.tile_id == required_tile_id:
                req_is_joker = True
        else:
            buckets.setdefault(k, []).append(t)
            if required_tile_id is not None and t.tile_id == required_tile_id:
                req_key = k

    if required_tile_id is not None and req_key is None and not req_is_joker:
        raise ValueError(f"Required tile_id {required_tile_id} is not in hand.")

    # Order unique keys: if there is a required non-joker key, put it at index 0
    # so the search enforces using it right at the root.
    sorted_keys = sorted(buckets.keys())
    if req_key is not None:
        sorted_keys.remove(req_key)
        sorted_keys.insert(0, req_key)
        req_local_idx: Optional[int] = 0
    elif req_is_joker:
        req_local_idx = -1
    else:
        req_local_idx = None

    tiles_by_local = [buckets[k] for k in sorted_keys]

    # If a specific tile is required, put it first in its bucket so reconstruction picks it
    if required_tile_id is not None:
        if req_is_joker:
            joker_tiles.sort(key=lambda t: 0 if t.tile_id == required_tile_id else 1)
        elif req_local_idx is not None:
            tiles_by_local[req_local_idx].sort(
                key=lambda t: 0 if t.tile_id == required_tile_id else 1
            )

    return sorted_keys, tiles_by_local, joker_tiles, req_local_idx


def _generate_candidates(
    unique_keys: list[int],
    num_jokers: int,
) -> list[_CandidateMeld]:
    """Generate all non-dominated candidate Runs (len 3..5) and Sets (len 3..4).

    Note: Any valid Run of length >= 6 decomposes into Runs of length 3..5 with
    the exact same multiset of tiles and exact same total points. Restricting
    to lengths 3..5 preserves exact optimality while keeping branching minimal.
    """
    key_to_local: dict[int, int] = {k: i for i, k in enumerate(unique_keys)}

    # Dominance map: (mask, jokers_used) -> best _CandidateMeld (highest points)
    best_by_signature: dict[tuple[int, int], _CandidateMeld] = {}

    def _record(cand: _CandidateMeld) -> None:
        sig = (cand.mask, cand.jokers_used)
        prev = best_by_signature.get(sig)
        if prev is None or cand.points > prev.points:
            best_by_signature[sig] = cand

    # 1. Generate Runs (lengths 3, 4, 5) for each color
    for c in ALL_COLORS:
        base = c.value * NUM_NUMBERS
        # Presence array for numbers 1..13
        local_for_num = [key_to_local.get(base + (n - 1), -1) for n in range(1, 14)]

        for length in (3, 4, 5):
            for start in range(1, 14 - length + 1):
                end = start + length  # exclusive
                missing = 0
                for n in range(start, end):
                    if local_for_num[n - 1] == -1:
                        missing += 1
                        if missing > num_jokers:
                            break
                if missing > num_jokers or missing == length:
                    continue

                slots: list[tuple[Color, int, int]] = []
                indices: list[int] = []
                mask = 0
                non_joker_pts = 0
                for n in range(start, end):
                    loc = local_for_num[n - 1]
                    if loc != -1:
                        indices.append(loc)
                        mask |= 1 << loc
                        non_joker_pts += n
                    slots.append((c, n, loc))

                total_pts = (start + (end - 1)) * length // 2
                indices.sort()
                _record(
                    _CandidateMeld(
                        meld_type=MeldType.RUN,
                        local_indices=tuple(indices),
                        jokers_used=missing,
                        points=total_pts,
                        slots=tuple(slots),
                        mask=mask,
                        non_joker_points=non_joker_pts,
                    )
                )

    # 2. Generate Sets (lengths 3, 4) for each number 1..13
    for n in range(1, 14):
        present: list[tuple[Color, int]] = []
        absent: list[Color] = []
        for c in ALL_COLORS:
            loc = key_to_local.get(color_number_to_key(c, n), -1)
            if loc != -1:
                present.append((c, loc))
            else:
                absent.append(c)

        if not present:
            continue

        # Size-3 sets
        # Case 3a: 3 real tiles, 0 jokers
        if len(present) >= 3:
            for skip in range(len(present)) if len(present) == 4 else [-1]:
                chosen = [p for i, p in enumerate(present) if i != skip]
                indices = sorted(loc for _, loc in chosen)
                mask = 0
                for loc in indices:
                    mask |= 1 << loc
                slots = tuple((c, n, loc) for c, loc in chosen)
                _record(
                    _CandidateMeld(
                        meld_type=MeldType.SET,
                        local_indices=tuple(indices),
                        jokers_used=0,
                        points=3 * n,
                        slots=slots,
                        mask=mask,
                        non_joker_points=3 * n,
                    )
                )

        # Case 3b: 2 real tiles + 1 joker
        if num_jokers >= 1 and len(present) >= 2:
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    c1, l1 = present[i]
                    c2, l2 = present[j]
                    # Pick any remaining color for the joker
                    j_color = next(c for c in ALL_COLORS if c != c1 and c != c2)
                    indices = (l1, l2) if l1 < l2 else (l2, l1)
                    mask = (1 << l1) | (1 << l2)
                    slots = tuple(
                        sorted(
                            [(c1, n, l1), (c2, n, l2), (j_color, n, -1)],
                            key=lambda s: s[0].value,
                        )
                    )
                    _record(
                        _CandidateMeld(
                            meld_type=MeldType.SET,
                            local_indices=indices,
                            jokers_used=1,
                            points=3 * n,
                            slots=slots,
                            mask=mask,
                            non_joker_points=2 * n,
                        )
                    )

        # Case 3c: 1 real tile + 2 jokers
        if num_jokers >= 2:
            for c1, l1 in present:
                rem_c = [c for c in ALL_COLORS if c != c1]
                slots = tuple(
                    sorted(
                        [(c1, n, l1), (rem_c[0], n, -1), (rem_c[1], n, -1)],
                        key=lambda s: s[0].value,
                    )
                )
                _record(
                    _CandidateMeld(
                        meld_type=MeldType.SET,
                        local_indices=(l1,),
                        jokers_used=2,
                        points=3 * n,
                        slots=slots,
                        mask=1 << l1,
                        non_joker_points=n,
                    )
                )

        # Size-4 sets
        # Case 4a: 4 real tiles, 0 jokers
        if len(present) == 4:
            indices = tuple(sorted(loc for _, loc in present))
            mask = 0
            for loc in indices:
                mask |= 1 << loc
            slots = tuple((c, n, loc) for c, loc in present)
            _record(
                _CandidateMeld(
                    meld_type=MeldType.SET,
                    local_indices=indices,
                    jokers_used=0,
                    points=4 * n,
                    slots=slots,
                    mask=mask,
                    non_joker_points=4 * n,
                )
            )

        # Case 4b: 3 real tiles + 1 joker
        if num_jokers >= 1 and len(present) >= 3:
            for skip in range(len(present)) if len(present) == 4 else [-1]:
                chosen = [p for i, p in enumerate(present) if i != skip]
                used_c = {c for c, _ in chosen}
                j_color = next(c for c in ALL_COLORS if c not in used_c)
                indices = tuple(sorted(loc for _, loc in chosen))
                mask = 0
                for loc in indices:
                    mask |= 1 << loc
                slots = tuple(
                    sorted(
                        [(c, n, loc) for c, loc in chosen] + [(j_color, n, -1)],
                        key=lambda s: s[0].value,
                    )
                )
                _record(
                    _CandidateMeld(
                        meld_type=MeldType.SET,
                        local_indices=indices,
                        jokers_used=1,
                        points=4 * n,
                        slots=slots,
                        mask=mask,
                        non_joker_points=3 * n,
                    )
                )

        # Case 4c: 2 real tiles + 2 jokers
        if num_jokers >= 2 and len(present) >= 2:
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    c1, l1 = present[i]
                    c2, l2 = present[j]
                    rem_c = [c for c in ALL_COLORS if c != c1 and c != c2]
                    indices = (l1, l2) if l1 < l2 else (l2, l1)
                    mask = (1 << l1) | (1 << l2)
                    slots = tuple(
                        sorted(
                            [
                                (c1, n, l1),
                                (c2, n, l2),
                                (rem_c[0], n, -1),
                                (rem_c[1], n, -1),
                            ],
                            key=lambda s: s[0].value,
                        )
                    )
                    _record(
                        _CandidateMeld(
                            meld_type=MeldType.SET,
                            local_indices=indices,
                            jokers_used=2,
                            points=4 * n,
                            slots=slots,
                            mask=mask,
                            non_joker_points=2 * n,
                        )
                    )

    return list(best_by_signature.values())


def _reconstruct_melds(
    chosen_cands: list[_CandidateMeld],
    tiles_by_local: list[list[Tile]],
    joker_tiles: list[Tile],
    joker_spec: JokerSpec,
) -> tuple[list[Meld], list[Tile], int]:
    """Convert chosen _CandidateMelds back into concrete Meld objects and deadwood."""
    local_cursors = [0] * len(tiles_by_local)
    joker_cursor = 0
    melds: list[Meld] = []

    for cand in chosen_cands:
        placed: list[PlacedTile] = []
        for c, n, loc in cand.slots:
            if loc == -1:
                t = joker_tiles[joker_cursor]
                joker_cursor += 1
            else:
                t = tiles_by_local[loc][local_cursors[loc]]
                local_cursors[loc] += 1
            placed.append(PlacedTile(t, c, n))
        melds.append(Meld(meld_type=cand.meld_type, placed_tiles=placed))

    deadwood: list[Tile] = []
    for loc, bucket in enumerate(tiles_by_local):
        deadwood.extend(bucket[local_cursors[loc]:])
    deadwood.extend(joker_tiles[joker_cursor:])

    deadwood_penalty = sum(joker_spec.hand_penalty_value(t) for t in deadwood)
    return melds, deadwood, deadwood_penalty


def _branch_and_bound(
    unique_keys: list[int],
    counts: list[int],
    num_jokers: int,
    candidates: list[_CandidateMeld],
    maximize_opening_points: bool,
    req_local_idx: Optional[int],
) -> tuple[int, list[_CandidateMeld]]:
    """Exact pivot-tile Branch & Bound solver over candidate melds.

    - If `maximize_opening_points=True`: maximizes sum of candidate `points`.
    - If `maximize_opening_points=False`: maximizes `non_joker_points + 101 * jokers_used`,
      which is strictly equivalent to minimizing deadwood penalty (with meld points as tie-break).
    """
    k = len(unique_keys)
    if k == 0 or not candidates:
        return 0, []

    values = [(key % NUM_NUMBERS) + 1 for key in unique_keys]

    # Group candidates by their minimum local_index (pivot).
    # Each candidate is placed ONLY in by_pivot[cand.local_indices[0]].
    # Because we iterate pivot idx = 0..k-1 in order, every disjoint multiset
    # of melds is visited in a unique canonical order!
    by_pivot: list[list[_CandidateMeld]] = [[] for _ in range(k)]
    for cand in candidates:
        by_pivot[cand.local_indices[0]].append(cand)

    # Sort each pivot's candidates by descending objective score for fast initial incumbent
    if maximize_opening_points:
        for lst in by_pivot:
            lst.sort(key=lambda c: c.points, reverse=True)
        joker_ub_unit = 13
    else:
        for lst in by_pivot:
            lst.sort(
                key=lambda c: (c.non_joker_points + 101 * c.jokers_used, c.points),
                reverse=True,
            )
        joker_ub_unit = 101

    # Suffix sum of remaining tile face values (only for tiles that participate in >= 1 candidate)
    participates = [False] * k
    for cand in candidates:
        for loc in cand.local_indices:
            participates[loc] = True

    # Upper bound helper: precompute max possible remaining value from each idx
    # (dynamically adjusted by current counts via running remaining_val)
    rem_tile_val = sum(
        counts[i] * values[i] for i in range(k) if participates[i]
    )

    best_score = -1
    best_chosen: list[_CandidateMeld] = []
    current_chosen: list[_CandidateMeld] = []

    req_non_joker = req_local_idx is not None and req_local_idx >= 0
    req_joker = req_local_idx == -1
    initial_c0 = counts[0] if req_non_joker else 0

    if req_non_joker and not by_pivot[0]:
        return 0, []

    def dfs(
        idx: int,
        rem_jokers: int,
        cur_score: int,
        cur_rem_val: int,
        min_cand_pos: int,
    ) -> None:
        nonlocal best_score, best_chosen

        # Advance to next index that still has count > 0 and has pivot candidates
        while idx < k and (counts[idx] == 0 or not by_pivot[idx]):
            if counts[idx] > 0 and participates[idx]:
                cur_rem_val -= counts[idx] * values[idx]
            idx += 1
            min_cand_pos = 0

        # Admissible upper bound check
        if cur_score + cur_rem_val + rem_jokers * joker_ub_unit <= best_score:
            return

        if idx == k:
            # Check required joker constraint
            if req_joker and rem_jokers == num_jokers:
                return
            if cur_score > best_score:
                best_score = cur_score
                best_chosen = list(current_chosen)
            return

        cands_here = by_pivot[idx]
        n_cands = len(cands_here)

        # Branch 1: Use pivot tile `idx` in one of its candidate melds
        for pos in range(min_cand_pos, n_cands):
            cand = cands_here[pos]
            if cand.jokers_used > rem_jokers:
                continue

            # Feasibility check: all local_indices must have count >= 1
            feasible = True
            for loc in cand.local_indices:
                if counts[loc] == 0:
                    feasible = False
                    break
            if not feasible:
                continue

            # Apply candidate
            for loc in cand.local_indices:
                counts[loc] -= 1
            current_chosen.append(cand)

            gain = (
                cand.points
                if maximize_opening_points
                else (cand.non_joker_points + 101 * cand.jokers_used)
            )
            next_rem_val = cur_rem_val - cand.non_joker_points

            if counts[idx] > 0:
                # Another copy of idx remains; stay at idx with min_cand_pos = pos
                dfs(idx, rem_jokers - cand.jokers_used, cur_score + gain, next_rem_val, pos)
            else:
                dfs(idx + 1, rem_jokers - cand.jokers_used, cur_score + gain, next_rem_val, 0)

            # Backtrack
            current_chosen.pop()
            for loc in cand.local_indices:
                counts[loc] += 1

        # Branch 2: Leave remaining copies of `idx` as deadwood and move on
        # Forbidden if idx == 0 is the required non-joker tile and none of its copies were used!
        if req_non_joker and idx == 0 and counts[0] == initial_c0:
            return

        drop_val = counts[idx] * values[idx] if participates[idx] else 0
        dfs(idx + 1, rem_jokers, cur_score, cur_rem_val - drop_val, 0)

    dfs(0, num_jokers, 0, rem_tile_val, 0)
    if best_score < 0:
        return 0, []
    return best_score, best_chosen


def solve_best_opening(
    hand: Sequence[Tile],
    joker_spec: JokerSpec,
    min_points: int = 101,
    required_tile_id: Optional[int] = None,
) -> OpeningSolution:
    """Find the exact highest-point disjoint valid Runs/Sets from `hand`.

    If `required_tile_id` is provided, the returned melds are constrained to
    include that tile (used to validate taking the previous player's discard).
    """
    unique_keys, tiles_by_local, joker_tiles, req_local_idx = _prepare_hand(
        hand, joker_spec, required_tile_id
    )
    counts = [len(b) for b in tiles_by_local]
    num_jokers = len(joker_tiles)

    candidates = _generate_candidates(unique_keys, num_jokers)
    best_pts, chosen = _branch_and_bound(
        unique_keys=unique_keys,
        counts=counts,
        num_jokers=num_jokers,
        candidates=candidates,
        maximize_opening_points=True,
        req_local_idx=req_local_idx,
    )

    if not chosen and required_tile_id is not None:
        # No valid meld set uses the required tile
        deadwood = list(hand)
        return OpeningSolution(
            melds=[],
            total_points=0,
            deadwood_tiles=deadwood,
            deadwood_penalty=sum(joker_spec.hand_penalty_value(t) for t in deadwood),
            meets_threshold=False,
        )

    melds, deadwood, deadwood_penalty = _reconstruct_melds(
        chosen, tiles_by_local, joker_tiles, joker_spec
    )
    total_points = sum(m.points for m in melds)
    return OpeningSolution(
        melds=melds,
        total_points=total_points,
        deadwood_tiles=deadwood,
        deadwood_penalty=deadwood_penalty,
        meets_threshold=total_points >= min_points and (required_tile_id is None or len(melds) > 0),
    )


def solve_best_pairs(
    hand: Sequence[Tile],
    joker_spec: JokerSpec,
    min_pairs: int = 5,
    required_tile_id: Optional[int] = None,
) -> PairsSolution:
    """Find the exact maximum number of valid Pairs (Çift) from `hand`.

    Among maximum-pair solutions, pairs Jokers with the highest face-value
    singletons to minimize remaining deadwood penalty.
    """
    unique_keys, tiles_by_local, joker_tiles, req_local_idx = _prepare_hand(
        hand, joker_spec, required_tile_id
    )
    jokers_rem = list(joker_tiles)
    pairs: list[Meld] = []
    singletons: list[tuple[int, Color, int, Tile]] = []  # (value, color, key, tile)
    used_required = required_tile_id is None

    # 1. Natural pairs (two identical effective tiles)
    for loc, bucket in enumerate(tiles_by_local):
        c, n = key_to_color_number(unique_keys[loc])
        if len(bucket) == 2:
            pairs.append(
                Meld(
                    meld_type=MeldType.PAIR,
                    placed_tiles=[
                        PlacedTile(bucket[0], c, n),
                        PlacedTile(bucket[1], c, n),
                    ],
                )
            )
            if req_local_idx == loc:
                used_required = True
        else:
            singletons.append((n, c, unique_keys[loc], bucket[0]))

    # 2. If required tile is a singleton non-joker, it MUST be paired with a Joker
    if req_local_idx is not None and req_local_idx >= 0 and not used_required:
        if not jokers_rem:
            deadwood = list(hand)
            return PairsSolution(
                pairs=[],
                pair_count=0,
                deadwood_tiles=deadwood,
                deadwood_penalty=sum(joker_spec.hand_penalty_value(t) for t in deadwood),
                meets_threshold=False,
            )
        req_key = unique_keys[req_local_idx]
        req_entry = next(s for s in singletons if s[2] == req_key)
        singletons.remove(req_entry)
        n, c, _, t = req_entry
        j_tile = jokers_rem.pop(0)
        pairs.append(
            Meld(
                meld_type=MeldType.PAIR,
                placed_tiles=[PlacedTile(t, c, n), PlacedTile(j_tile, c, n)],
            )
        )
        used_required = True

    # 3. Pair remaining Jokers with highest face-value singletons
    singletons.sort(key=lambda s: s[0], reverse=True)
    while jokers_rem and singletons:
        n, c, _, t = singletons.pop(0)
        j_tile = jokers_rem.pop(0)
        pairs.append(
            Meld(
                meld_type=MeldType.PAIR,
                placed_tiles=[PlacedTile(t, c, n), PlacedTile(j_tile, c, n)],
            )
        )
        if req_local_idx == -1:
            used_required = True

    # 4. If 2 Jokers remain with no singletons left, pair them together
    while len(jokers_rem) >= 2:
        j1 = jokers_rem.pop(0)
        j2 = jokers_rem.pop(0)
        pairs.append(
            Meld(
                meld_type=MeldType.PAIR,
                placed_tiles=[
                    PlacedTile(j1, Color.RED, 13),
                    PlacedTile(j2, Color.RED, 13),
                ],
            )
        )
        if req_local_idx == -1:
            used_required = True

    if not used_required:
        deadwood = list(hand)
        return PairsSolution(
            pairs=[],
            pair_count=0,
            deadwood_tiles=deadwood,
            deadwood_penalty=sum(joker_spec.hand_penalty_value(t) for t in deadwood),
            meets_threshold=False,
        )

    deadwood_tiles = [s[3] for s in singletons] + jokers_rem
    deadwood_penalty = sum(joker_spec.hand_penalty_value(t) for t in deadwood_tiles)
    return PairsSolution(
        pairs=pairs,
        pair_count=len(pairs),
        deadwood_tiles=deadwood_tiles,
        deadwood_penalty=deadwood_penalty,
        meets_threshold=len(pairs) >= min_pairs,
    )


def solve_min_deadwood(
    hand: Sequence[Tile],
    joker_spec: JokerSpec,
    required_tile_id: Optional[int] = None,
) -> DeadwoodSolution:
    """Find disjoint valid Runs/Sets that minimize remaining hand deadwood penalty.

    Used for already-opened hands (and checking whether a single new meld can use
    a taken discard tile).
    """
    unique_keys, tiles_by_local, joker_tiles, req_local_idx = _prepare_hand(
        hand, joker_spec, required_tile_id
    )
    counts = [len(b) for b in tiles_by_local]
    num_jokers = len(joker_tiles)

    candidates = _generate_candidates(unique_keys, num_jokers)
    _, chosen = _branch_and_bound(
        unique_keys=unique_keys,
        counts=counts,
        num_jokers=num_jokers,
        candidates=candidates,
        maximize_opening_points=False,
        req_local_idx=req_local_idx,
    )

    if not chosen and required_tile_id is not None:
        deadwood = list(hand)
        return DeadwoodSolution(
            melds=[],
            total_meld_points=0,
            deadwood_tiles=deadwood,
            deadwood_penalty=sum(joker_spec.hand_penalty_value(t) for t in deadwood),
        )

    melds, deadwood, deadwood_penalty = _reconstruct_melds(
        chosen, tiles_by_local, joker_tiles, joker_spec
    )
    return DeadwoodSolution(
        melds=melds,
        total_meld_points=sum(m.points for m in melds),
        deadwood_tiles=deadwood,
        deadwood_penalty=deadwood_penalty,
    )
