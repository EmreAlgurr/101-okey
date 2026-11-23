"""Comprehensive unit tests for Official Professional 101 Okey rules, solver, and engine."""

from __future__ import annotations

import random
import time
import pytest

from engine import (
    Action,
    ActionType,
    GameConfig,
    Okey101Engine,
    OpenType,
    TurnPhase,
)
from solver import solve_best_opening, solve_best_pairs, solve_min_deadwood
from tiles import (
    Color,
    JokerSpec,
    Meld,
    MeldType,
    PlacedTile,
    Tile,
    apply_lay_off,
    build_pair,
    build_run,
    build_set,
    can_lay_off,
    create_deck,
    validate_meld,
)


def _deck_lookup() -> tuple[list[Tile], dict[tuple[Color, int], list[Tile]], list[Tile]]:
    """Helper returning (full_deck, {(color, num): [copy0, copy1]}, [fj0, fj1])."""
    deck = create_deck()
    by_cn: dict[tuple[Color, int], list[Tile]] = {}
    fjs: list[Tile] = []
    for t in deck:
        if t.is_false_joker:
            fjs.append(t)
        else:
            assert t.color is not None
            by_cn.setdefault((t.color, t.number), []).append(t)
    return deck, by_cn, fjs


# ---------------------------------------------------------------------------
# 1. Deck, Indicator & Deal Tests
# ---------------------------------------------------------------------------


def test_deck_composition_and_indicator_wrap() -> None:
    deck, by_cn, fjs = _deck_lookup()
    assert len(deck) == 106
    assert len(fjs) == 2
    assert len(by_cn) == 52
    assert all(len(v) == 2 for v in by_cn.values())

    # Indicator 13 -> Joker is 1 of the same color
    ind_13 = by_cn[(Color.RED, 13)][0]
    spec = JokerSpec.from_indicator(ind_13)
    assert spec.joker_color == Color.RED
    assert spec.joker_number == 1
    assert spec.is_joker(by_cn[(Color.RED, 1)][0])
    assert spec.is_joker(by_cn[(Color.RED, 1)][1])
    assert not spec.is_joker(fjs[0])
    assert spec.effective_color_number(fjs[0]) == (Color.RED, 1)

    # False Joker cannot be indicator
    with pytest.raises(ValueError):
        JokerSpec.from_indicator(fjs[0])


def test_official_deal_counts() -> None:
    engine = Okey101Engine()
    obs = engine.reset(seed=42, starting_player=0)
    assert len(engine.hands[0]) == 22
    assert len(engine.hands[1]) == 21
    assert len(engine.hands[2]) == 21
    assert len(engine.hands[3]) == 21
    assert len(engine.draw_pile) == 21
    assert obs.phase == TurnPhase.PLAY
    assert obs.current_player == 0
    assert not engine.joker_spec.indicator.is_false_joker

    all_ids = [t.tile_id for h in engine.hands for t in h] + [
        t.tile_id for t in engine.draw_pile
    ]
    assert len(all_ids) == 106
    assert len(set(all_ids)) == 106


# ---------------------------------------------------------------------------
# 2. Strict 12-13-1 Rejection & Meld Validation
# ---------------------------------------------------------------------------


def test_12_13_1_strictly_rejected() -> None:
    _, by_cn, _ = _deck_lookup()
    spec = JokerSpec.from_indicator(by_cn[(Color.BLACK, 5)][0])  # Joker is K6

    r12 = by_cn[(Color.RED, 12)][0]
    r13 = by_cn[(Color.RED, 13)][0]
    r1 = by_cn[(Color.RED, 1)][0]

    # build_run must reject 12-13-1
    assert build_run([r12, r13, r1], spec) is None

    # Manual Meld construction of 12-13-1 must fail validate_meld
    bogus_meld = Meld(
        meld_type=MeldType.RUN,
        placed_tiles=[
            PlacedTile(r12, Color.RED, 12),
            PlacedTile(r13, Color.RED, 13),
            PlacedTile(r1, Color.RED, 1),
        ],
    )
    assert not validate_meld(bogus_meld, spec)

    # Solver must NOT form 12-13-1
    sol = solve_best_opening([r12, r13, r1], spec, min_points=1)
    assert sol.melds == []
    assert sol.total_points == 0

    # Valid boundary runs 1-2-3 and 11-12-13 MUST be accepted
    r2 = by_cn[(Color.RED, 2)][0]
    r3 = by_cn[(Color.RED, 3)][0]
    r11 = by_cn[(Color.RED, 11)][0]
    run_low = build_run([r1, r2, r3], spec)
    run_high = build_run([r11, r12, r13], spec)
    assert run_low is not None and run_low.points == 6
    assert run_high is not None and run_high.points == 36

    # Laying off 1 onto 11-12-13 MUST be rejected
    assert can_lay_off(run_high, r1, spec) == []


# ---------------------------------------------------------------------------
# 3. False Joker (Sahte Okey) Substitution
# ---------------------------------------------------------------------------


def test_false_joker_strict_substitution() -> None:
    _, by_cn, fjs = _deck_lookup()
    # Indicator R5 -> Joker is R6 -> False Joker strictly represents R6 (value 6)
    spec = JokerSpec.from_indicator(by_cn[(Color.RED, 5)][0])
    fj0, fj1 = fjs
    real_joker = by_cn[(Color.RED, 6)][0]

    # False Joker fits R5-FJ-R7
    run_valid = build_run([by_cn[(Color.RED, 5)][1], fj0, by_cn[(Color.RED, 7)][0]], spec)
    assert run_valid is not None
    assert run_valid.points == 18  # 5 + 6 + 7

    # False Joker does NOT act as wildcard for B5-FJ-B7
    assert build_run([by_cn[(Color.BLUE, 5)][0], fj0, by_cn[(Color.BLUE, 7)][0]], spec) is None

    # False Joker fits Set of 6s (Y6, B6, FJ as R6)
    set_valid = build_set([by_cn[(Color.YELLOW, 6)][0], by_cn[(Color.BLUE, 6)][0], fj0], spec)
    assert set_valid is not None
    assert set_valid.points == 18

    # Two False Jokers form a valid Pair (both are R6)
    pair_fj = build_pair([fj0, fj1], spec)
    assert pair_fj is not None

    # Hand penalty: False Joker = 6, Real Joker = 101
    assert spec.hand_penalty_value(fj0) == 6
    assert spec.hand_penalty_value(real_joker) == 101


# ---------------------------------------------------------------------------
# 4. Exact Hand Solver Tests (0, 1, 2 Jokers + Performance < 50ms)
# ---------------------------------------------------------------------------


def test_solver_exact_opening_with_jokers() -> None:
    _, by_cn, _ = _deck_lookup()
    spec = JokerSpec.from_indicator(by_cn[(Color.BLACK, 1)][0])  # Joker is K2
    j1, j2 = by_cn[(Color.BLACK, 2)]

    # Hand with 2 Jokers:
    # R11 + j1 + j2 -> best run is R11-R12-R13 = 36 pts (NOT R9-10-11 = 30!)
    # Plus Y11-Y12-Y13 (36 pts) + B10-B11-B12 (33 pts) -> total 105 >= 101!
    hand = [
        by_cn[(Color.RED, 11)][0],
        j1,
        j2,
        by_cn[(Color.YELLOW, 11)][0],
        by_cn[(Color.YELLOW, 12)][0],
        by_cn[(Color.YELLOW, 13)][0],
        by_cn[(Color.BLUE, 10)][0],
        by_cn[(Color.BLUE, 11)][0],
        by_cn[(Color.BLUE, 12)][0],
        by_cn[(Color.RED, 1)][0],
    ]
    sol = solve_best_opening(hand, spec, min_points=101)
    assert sol.meets_threshold
    assert sol.total_points == 105  # 36 + 36 + 33
    assert len(sol.deadwood_tiles) == 1
    assert sol.deadwood_tiles[0] == by_cn[(Color.RED, 1)][0]
    for m in sol.melds:
        assert validate_meld(m, spec)


def test_solver_pairs_and_min_deadwood() -> None:
    _, by_cn, fjs = _deck_lookup()
    spec = JokerSpec.from_indicator(by_cn[(Color.RED, 7)][0])  # Joker is R8
    j1, j2 = by_cn[(Color.RED, 8)]

    # 3 natural pairs + 2 False Jokers (4th pair) + 1 singleton + 1 real Joker (5th pair)
    hand = [
        by_cn[(Color.BLUE, 3)][0], by_cn[(Color.BLUE, 3)][1],
        by_cn[(Color.YELLOW, 9)][0], by_cn[(Color.YELLOW, 9)][1],
        by_cn[(Color.BLACK, 12)][0], by_cn[(Color.BLACK, 12)][1],
        fjs[0], fjs[1],
        by_cn[(Color.RED, 13)][0],
        by_cn[(Color.RED, 2)][0],
        j1,
    ]
    psol = solve_best_pairs(hand, spec, min_pairs=5)
    assert psol.meets_threshold
    assert psol.pair_count == 5
    # Joker should pair with R13 (value 13) rather than R2 (value 2) to minimize deadwood
    assert psol.deadwood_penalty == 2

    # Min deadwood should always prefer using a real Joker over leaving it as +101 deadwood
    dw_hand = [
        by_cn[(Color.BLUE, 1)][0],
        by_cn[(Color.BLUE, 2)][0],
        j1,
        by_cn[(Color.YELLOW, 10)][0],
        by_cn[(Color.YELLOW, 11)][0],
    ]
    dw = solve_min_deadwood(dw_hand, spec)
    # Using j1 with Y10-Y11 removes 10+11+101=122 penalty, leaving B1+B2=3 deadwood!
    assert dw.deadwood_penalty == 3
    assert len(dw.melds) == 1

    # Required tile that cannot participate in ANY meld must return empty melds even if other tiles meld
    isolated = by_cn[(Color.BLACK, 1)][0]
    other_run = [
        by_cn[(Color.YELLOW, 11)][0],
        by_cn[(Color.YELLOW, 12)][0],
        by_cn[(Color.YELLOW, 13)][0],
    ]
    req_sol = solve_best_opening(
        [isolated] + other_run, spec, min_points=1, required_tile_id=isolated.tile_id
    )
    assert req_sol.melds == []
    assert not req_sol.meets_threshold
    req_dw = solve_min_deadwood(
        [isolated] + other_run, spec, required_tile_id=isolated.tile_id
    )
    assert req_dw.melds == []


def test_solver_performance_under_50ms_with_two_jokers() -> None:
    rng = random.Random(12345)
    deck, by_cn, _ = _deck_lookup()
    spec = JokerSpec.from_indicator(by_cn[(Color.RED, 12)][0])  # Joker is R13
    jokers = by_cn[(Color.RED, 13)]
    non_jokers = [t for t in deck if not spec.is_joker(t)]

    max_ms = 0.0
    for _ in range(40):
        # Stress-test 22-tile hands guaranteed to contain BOTH real Jokers
        hand = list(jokers) + rng.sample(non_jokers, 20)
        t0 = time.perf_counter()
        sol = solve_best_opening(hand, spec, min_points=101)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        max_ms = max(max_ms, elapsed_ms)
        for m in sol.melds:
            assert validate_meld(m, spec)

    assert max_ms < 50.0, f"Solver exceeded 50ms budget: worst case {max_ms:.2f}ms"


# ---------------------------------------------------------------------------
# 5. Discard Penalties (+101 İşlek & +101 Joker Discard)
# ---------------------------------------------------------------------------


def test_islek_and_joker_discard_penalties() -> None:
    _, by_cn, fjs = _deck_lookup()
    engine = Okey101Engine()
    engine.reset(seed=7)
    # Override indicator to R5 -> Joker is R6, False Joker is R6
    engine.joker_spec = JokerSpec.from_indicator(by_cn[(Color.RED, 5)][0])

    # Put open run R7-R8-R9 on table
    run_r789 = build_run(
        [by_cn[(Color.RED, 7)][0], by_cn[(Color.RED, 8)][0], by_cn[(Color.RED, 9)][0]],
        engine.joker_spec,
    )
    assert run_r789 is not None
    engine._add_table_meld(owner_id=1, meld=run_r789)

    # Player 0 (even though unopened!) discards R10 (fits R7-R8-R9) -> +101 işlek penalty
    r10 = by_cn[(Color.RED, 10)][0]
    b1 = by_cn[(Color.BLUE, 1)][0]
    fj0 = fjs[0]  # Effective R6 -> ALSO fits R7-R8-R9 on the left!
    real_joker = by_cn[(Color.RED, 6)][0]

    engine.hands[0] = [r10, b1, fj0, real_joker]
    engine.current_player = 0
    engine.phase = TurnPhase.PLAY

    res1 = engine.step(Action(ActionType.DISCARD, tile_id=r10.tile_id))
    assert res1.penalty_incurred == 101
    assert engine.penalties[0] == 101
    assert res1.info.get("penalty_reason") == "islek_discard"

    # Give player 0 the turn again and discard False Joker (R6 -> fits left of R7-R8-R9!)
    engine.current_player = 0
    engine.phase = TurnPhase.PLAY
    res2 = engine.step(Action(ActionType.DISCARD, tile_id=fj0.tile_id))
    assert res2.penalty_incurred == 101
    assert engine.penalties[0] == 202
    assert res2.info.get("penalty_reason") == "islek_discard"

    # Discard real Joker without winning -> +101 (NOT +202!)
    engine.current_player = 0
    engine.phase = TurnPhase.PLAY
    res3 = engine.step(Action(ActionType.DISCARD, tile_id=real_joker.tile_id))
    assert res3.penalty_incurred == 101
    assert engine.penalties[0] == 303
    assert res3.info.get("penalty_reason") == "joker_discard"


# ---------------------------------------------------------------------------
# 6. Taking Discard Rule, Pairs Restriction & Katlamali Mode
# ---------------------------------------------------------------------------


def test_taking_discard_enforces_immediate_use() -> None:
    _, by_cn, _ = _deck_lookup()
    engine = Okey101Engine()
    engine.reset(seed=10)
    engine.joker_spec = JokerSpec.from_indicator(by_cn[(Color.BLACK, 1)][0])

    # Player 1 is unopened and has melds totaling 90 pts (needs R11 to make 101+)
    # Melds: Y10-Y11-Y12 (33) + B10-B11-B12 (33) + K8-K9-K10 (27) = 93 pts
    # Plus R9, R10 in hand. If prev player (0) discards R11 -> R9-R10-R11 (30) -> 123 pts!
    r11 = by_cn[(Color.RED, 11)][0]
    b1 = by_cn[(Color.BLUE, 1)][0]
    engine.hands[1] = [
        by_cn[(Color.YELLOW, 10)][0], by_cn[(Color.YELLOW, 11)][0], by_cn[(Color.YELLOW, 12)][0],
        by_cn[(Color.BLUE, 10)][0], by_cn[(Color.BLUE, 11)][0], by_cn[(Color.BLUE, 12)][0],
        by_cn[(Color.BLACK, 8)][0], by_cn[(Color.BLACK, 9)][0], by_cn[(Color.BLACK, 10)][0],
        by_cn[(Color.RED, 9)][0], by_cn[(Color.RED, 10)][0],
        by_cn[(Color.RED, 2)][0],
    ]
    engine.current_player = 1
    engine.phase = TurnPhase.DRAW

    # If previous player discarded B1 (useless to open), TAKE_DISCARD is illegal
    engine.discard_piles[0] = [b1]
    assert not engine.can_take_discard(1)
    with pytest.raises(ValueError):
        engine.step(Action(ActionType.TAKE_DISCARD))

    # If previous player discarded R11, TAKE_DISCARD is legal
    engine.discard_piles[0] = [r11]
    assert engine.can_take_discard(1)
    engine.step(Action(ActionType.TAKE_DISCARD))
    assert engine.must_use_tile_id == r11.tile_id

    # Cannot discard before using the taken R11!
    with pytest.raises(ValueError):
        engine.step(Action(ActionType.DISCARD, tile_id=by_cn[(Color.RED, 2)][0].tile_id))

    # Open using the solver action (which includes R11) -> clears must_use_tile_id
    open_actions = [
        a for a in engine.get_legal_actions() if a.action_type == ActionType.OPEN_SERIES
    ]
    assert len(open_actions) == 1
    engine.step(open_actions[0])
    assert engine.must_use_tile_id is None
    assert engine.open_types[1] == OpenType.SERIES


def test_series_open_take_discard_for_new_meld_and_layoff() -> None:
    """Already Series-open player can TAKE_DISCARD to form a new Run/Set OR lay off."""
    _, by_cn, _ = _deck_lookup()
    engine = Okey101Engine()
    engine.reset(seed=15)
    engine.joker_spec = JokerSpec.from_indicator(by_cn[(Color.BLACK, 1)][0])  # Joker is K2

    # Case 1: Holding R1, R2, R4 (+ B9) while Series-open, prev player (P3) discards R3
    r1 = by_cn[(Color.RED, 1)][0]
    r2 = by_cn[(Color.RED, 2)][0]
    r3 = by_cn[(Color.RED, 3)][0]
    r4 = by_cn[(Color.RED, 4)][0]
    b9 = by_cn[(Color.BLUE, 9)][0]
    y7 = by_cn[(Color.YELLOW, 7)][0]

    engine.open_types[0] = OpenType.SERIES
    engine.opened_before_turn = True
    engine.current_player = 0
    engine.phase = TurnPhase.DRAW
    engine.hands[0] = [r1, r2, r4, b9]

    # Useless discard Y7 -> cannot take
    engine.discard_piles[3] = [y7]
    assert not engine.can_take_discard(0)
    assert ActionType.TAKE_DISCARD not in [a.action_type for a in engine.get_legal_actions()]

    # R3 completes new run R1-R2-R3-R4 -> legal to take!
    engine.discard_piles[3] = [r3]
    assert engine.can_take_discard(0)
    assert ActionType.TAKE_DISCARD in [a.action_type for a in engine.get_legal_actions()]

    engine.step(Action(ActionType.TAKE_DISCARD))
    assert engine.must_use_tile_id == r3.tile_id

    # Must use R3 before discarding
    legal_after = engine.get_legal_actions()
    assert ActionType.DISCARD not in [a.action_type for a in legal_after]
    with pytest.raises(ValueError):
        engine.step(Action(ActionType.DISCARD, tile_id=b9.tile_id))

    meld_actions = [a for a in legal_after if a.action_type == ActionType.MELD]
    assert len(meld_actions) == 1
    used_ids = {pt.tile.tile_id for m in meld_actions[0].melds for pt in m.placed_tiles}
    assert {r1.tile_id, r2.tile_id, r3.tile_id, r4.tile_id}.issubset(used_ids)

    engine.step(meld_actions[0])
    assert engine.must_use_tile_id is None
    assert engine.hands[0] == [b9]

    # Now discarding B9 is legal and wins the round
    res = engine.step(Action(ActionType.DISCARD, tile_id=b9.tile_id))
    assert res.done and res.winner == 0

    # Case 2: Holding B3, Y3 while Series-open, prev player discards R3 (forms 3-set)
    engine.reset(seed=16)
    engine.joker_spec = JokerSpec.from_indicator(by_cn[(Color.BLACK, 1)][0])
    engine.open_types[0] = OpenType.SERIES
    engine.current_player = 0
    engine.phase = TurnPhase.DRAW
    engine.hands[0] = [by_cn[(Color.BLUE, 3)][0], by_cn[(Color.YELLOW, 3)][0], b9]
    engine.discard_piles[3] = [r3]
    assert engine.can_take_discard(0)

    # Case 3: Lay-off onto existing table meld (R4-R5-R6) while Series-open
    run_r456 = build_run(
        [by_cn[(Color.RED, 4)][1], by_cn[(Color.RED, 5)][0], by_cn[(Color.RED, 6)][0]],
        engine.joker_spec,
    )
    assert run_r456 is not None
    engine._add_table_meld(owner_id=1, meld=run_r456)
    engine.hands[0] = [b9, y7]
    engine.discard_piles[3] = [r3]
    assert engine.can_take_discard(0)


def test_house_rule_feed_penalty_multiplier() -> None:
    """feed_penalty_multiplier=10 charges prev_player tile.value * 10 on TAKE_DISCARD (0 otherwise)."""
    _, by_cn, fjs = _deck_lookup()
    # Indicator R5 -> Joker is R6 -> False Joker has value 6
    ind = by_cn[(Color.RED, 5)][0]

    # 1. Taking R12 from P3 charges P3 +120
    eng = Okey101Engine(feed_penalty_multiplier=10)
    eng.reset(seed=30)
    eng.joker_spec = JokerSpec.from_indicator(ind)
    eng.open_types[0] = OpenType.SERIES
    eng.current_player = 0
    eng.phase = TurnPhase.DRAW

    r10, r11, r12 = by_cn[(Color.RED, 10)][0], by_cn[(Color.RED, 11)][0], by_cn[(Color.RED, 12)][0]
    eng.hands[0] = [r10, r11, by_cn[(Color.BLUE, 1)][0]]
    eng.discard_piles[3] = [r12]

    obs = eng.get_observation(0)
    assert obs.feed_penalty_multiplier == 10
    assert eng.penalties[3] == 0

    res = eng.step(Action(ActionType.TAKE_DISCARD))
    assert eng.penalties[3] == 120
    assert eng.penalties[0] == 0
    assert res.info["feed_penalty_player"] == 3
    assert res.info["feed_penalty_amount"] == 120

    # 2. Taking False Joker (effective R6, value 6) charges P3 +60
    eng2 = Okey101Engine(config=GameConfig(katlamali=True, feed_penalty_multiplier=10))
    eng2.reset(seed=31)
    eng2.joker_spec = JokerSpec.from_indicator(ind)
    eng2.open_types[0] = OpenType.SERIES
    eng2.current_player = 0
    eng2.phase = TurnPhase.DRAW
    eng2.hands[0] = [by_cn[(Color.RED, 4)][0], by_cn[(Color.RED, 5)][1], by_cn[(Color.BLUE, 2)][0]]
    eng2.discard_piles[3] = [fjs[0]]
    eng2.step(Action(ActionType.TAKE_DISCARD))
    assert eng2.penalties[3] == 60

    # 3. Drawing from deck instead of taking discard -> 0 feed penalty
    eng3 = Okey101Engine(feed_penalty_multiplier=10)
    eng3.reset(seed=32)
    eng3.joker_spec = JokerSpec.from_indicator(ind)
    eng3.open_types[0] = OpenType.SERIES
    eng3.current_player = 0
    eng3.phase = TurnPhase.DRAW
    eng3.hands[0] = [r10, r11, by_cn[(Color.BLUE, 1)][0]]
    eng3.discard_piles[3] = [r12]
    eng3.step(Action(ActionType.DRAW_DECK))
    assert eng3.penalties[3] == 0


def test_pairs_opener_cannot_lay_off_onto_runs_sets() -> None:
    _, by_cn, _ = _deck_lookup()
    engine = Okey101Engine()
    engine.reset(seed=11)
    engine.joker_spec = JokerSpec.from_indicator(by_cn[(Color.BLACK, 1)][0])

    run_r567 = build_run(
        [by_cn[(Color.RED, 5)][0], by_cn[(Color.RED, 6)][0], by_cn[(Color.RED, 7)][0]],
        engine.joker_spec,
    )
    assert run_r567 is not None
    tm = engine._add_table_meld(owner_id=0, meld=run_r567)

    engine.open_types[1] = OpenType.PAIRS
    r8 = by_cn[(Color.RED, 8)][0]
    engine.hands[1] = [r8, by_cn[(Color.BLUE, 1)][0]]
    engine.current_player = 1
    engine.phase = TurnPhase.PLAY

    with pytest.raises(ValueError):
        engine.step(Action(ActionType.LAY_OFF, tile_id=r8.tile_id, meld_id=tm.meld_id))


def test_katlamali_strictly_increasing_open_threshold() -> None:
    _, by_cn, _ = _deck_lookup()
    engine = Okey101Engine(config=GameConfig(katlamali=True))
    engine.reset(seed=12)
    engine.joker_spec = JokerSpec.from_indicator(by_cn[(Color.BLACK, 1)][0])

    # Player 0 opens with 108 points (R11-12-13=36, Y11-12-13=36, B11-12-13=36)
    m1 = build_run([by_cn[(Color.RED, 11)][0], by_cn[(Color.RED, 12)][0], by_cn[(Color.RED, 13)][0]], engine.joker_spec)
    m2 = build_run([by_cn[(Color.YELLOW, 11)][0], by_cn[(Color.YELLOW, 12)][0], by_cn[(Color.YELLOW, 13)][0]], engine.joker_spec)
    m3 = build_run([by_cn[(Color.BLUE, 11)][0], by_cn[(Color.BLUE, 12)][0], by_cn[(Color.BLUE, 13)][0]], engine.joker_spec)
    assert m1 and m2 and m3

    engine.hands[0] = m1.tiles + m2.tiles + m3.tiles + [by_cn[(Color.BLACK, 3)][0]]
    engine.current_player = 0
    engine.phase = TurnPhase.PLAY
    engine.step(Action(ActionType.OPEN_SERIES, melds=(m1, m2, m3)))
    assert engine.required_series_points == 109  # Strictly exceeds 108!

    # Player 1 tries to open with 108 points -> rejected!
    m4 = build_run([by_cn[(Color.RED, 11)][1], by_cn[(Color.RED, 12)][1], by_cn[(Color.RED, 13)][1]], engine.joker_spec)
    m5 = build_run([by_cn[(Color.YELLOW, 11)][1], by_cn[(Color.YELLOW, 12)][1], by_cn[(Color.YELLOW, 13)][1]], engine.joker_spec)
    m6 = build_run([by_cn[(Color.BLUE, 11)][1], by_cn[(Color.BLUE, 12)][1], by_cn[(Color.BLUE, 13)][1]], engine.joker_spec)
    assert m4 and m5 and m6

    engine.hands[1] = m4.tiles + m5.tiles + m6.tiles + [by_cn[(Color.BLACK, 4)][0]]
    engine.current_player = 1
    engine.phase = TurnPhase.PLAY
    with pytest.raises(ValueError):
        engine.step(Action(ActionType.OPEN_SERIES, melds=(m4, m5, m6)))


# ---------------------------------------------------------------------------
# 7. Official Round-End Scoring & Multipliers
# ---------------------------------------------------------------------------


def test_scoring_normal_win_and_unopened_and_pairs_penalties() -> None:
    _, by_cn, _ = _deck_lookup()
    engine = Okey101Engine()
    engine.reset(seed=20)
    engine.joker_spec = JokerSpec.from_indicator(by_cn[(Color.RED, 4)][0])  # Joker is R5
    real_joker = by_cn[(Color.RED, 5)][0]

    # Player 0 already opened before this turn, has 1 normal tile left to discard -> normal win (-101)
    engine.open_types = [OpenType.SERIES, OpenType.NONE, OpenType.SERIES, OpenType.PAIRS]
    engine.opened_before_turn = True
    engine.current_player = 0
    engine.phase = TurnPhase.PLAY

    engine.hands[0] = [by_cn[(Color.BLUE, 1)][0]]
    engine.hands[1] = [by_cn[(Color.BLUE, 2)][0]]  # Unopened -> +202
    engine.hands[2] = [by_cn[(Color.YELLOW, 10)][0], real_joker]  # Series open -> 10 + 101 = +111
    engine.hands[3] = [by_cn[(Color.BLACK, 7)][0], by_cn[(Color.BLACK, 8)][0]]  # Pairs open -> (7+8)*2 = +30

    res = engine.step(Action(ActionType.DISCARD, tile_id=by_cn[(Color.BLUE, 1)][0].tile_id))
    assert res.done
    assert res.winner == 0
    assert engine.win_flags.win_multiplier == 1
    assert res.round_scores == (-101, 202, 111, 30)


def test_scoring_okey_win_multiplier() -> None:
    _, by_cn, _ = _deck_lookup()
    engine = Okey101Engine()
    engine.reset(seed=21)
    engine.joker_spec = JokerSpec.from_indicator(by_cn[(Color.RED, 4)][0])  # Joker is R5
    real_joker = by_cn[(Color.RED, 5)][0]

    engine.open_types = [OpenType.SERIES, OpenType.NONE, OpenType.SERIES, OpenType.PAIRS]
    engine.opened_before_turn = True
    engine.current_player = 0
    engine.phase = TurnPhase.PLAY

    engine.hands[0] = [real_joker]  # Winning by discarding Joker -> -202, opponents x2!
    engine.hands[1] = [by_cn[(Color.BLUE, 2)][0]]  # Unopened -> 202 * 2 = +404
    engine.hands[2] = [by_cn[(Color.YELLOW, 10)][0]]  # Series -> 10 * 2 = +20
    engine.hands[3] = [by_cn[(Color.BLACK, 5)][0]]  # Pairs -> 5 * 2 * 2 = +20

    res = engine.step(Action(ActionType.DISCARD, tile_id=real_joker.tile_id))
    assert res.done
    assert engine.win_flags.win_by_okey
    assert not engine.win_flags.win_by_elden
    assert engine.win_flags.win_multiplier == 2
    assert res.round_scores == (-202, 404, 20, 20)


def test_scoring_elden_plus_okey_x4_multiplier() -> None:
    _, by_cn, _ = _deck_lookup()
    engine = Okey101Engine()
    engine.reset(seed=22)
    engine.joker_spec = JokerSpec.from_indicator(by_cn[(Color.BLACK, 1)][0])  # Joker is K2
    real_joker = by_cn[(Color.BLACK, 2)][0]

    # Player 0 has NOT opened yet (opened_before_turn=False), opens with >= 101
    # leaving ONLY real_joker, and discards real_joker -> Elden + Okey = -404 & opponents x4!
    m1 = build_run([by_cn[(Color.RED, 11)][0], by_cn[(Color.RED, 12)][0], by_cn[(Color.RED, 13)][0]], engine.joker_spec)
    m2 = build_run([by_cn[(Color.YELLOW, 11)][0], by_cn[(Color.YELLOW, 12)][0], by_cn[(Color.YELLOW, 13)][0]], engine.joker_spec)
    m3 = build_run([by_cn[(Color.BLUE, 11)][0], by_cn[(Color.BLUE, 12)][0], by_cn[(Color.BLUE, 13)][0]], engine.joker_spec)
    assert m1 and m2 and m3

    engine.open_types = [OpenType.NONE, OpenType.NONE, OpenType.SERIES, OpenType.PAIRS]
    engine.opened_before_turn = False
    engine.current_player = 0
    engine.phase = TurnPhase.PLAY

    engine.hands[0] = m1.tiles + m2.tiles + m3.tiles + [real_joker]
    engine.hands[1] = [by_cn[(Color.BLUE, 1)][0]]     # Unopened -> 202 * 4 = 808
    engine.hands[2] = [by_cn[(Color.YELLOW, 5)][0]]   # Series -> 5 * 4 = 20
    engine.hands[3] = [by_cn[(Color.BLACK, 3)][0]]    # Pairs -> 3 * 2 * 4 = 24

    engine.step(Action(ActionType.OPEN_SERIES, melds=(m1, m2, m3)))
    res = engine.step(Action(ActionType.DISCARD, tile_id=real_joker.tile_id))
    assert res.done
    assert engine.win_flags.win_by_elden
    assert engine.win_flags.win_by_okey
    assert engine.win_flags.win_multiplier == 4
    assert res.round_scores == (-404, 808, 20, 24)


def test_scoring_pairs_win_multiplier() -> None:
    _, by_cn, _ = _deck_lookup()
    engine = Okey101Engine()
    engine.reset(seed=23)
    engine.joker_spec = JokerSpec.from_indicator(by_cn[(Color.BLACK, 1)][0])

    # Player 0 previously opened via PAIRS, now discards last tile -> Çift ile bitme (-202, x2)
    engine.open_types = [OpenType.PAIRS, OpenType.NONE, OpenType.SERIES, OpenType.PAIRS]
    engine.opened_before_turn = True
    engine.current_player = 0
    engine.phase = TurnPhase.PLAY

    engine.hands[0] = [by_cn[(Color.RED, 3)][0]]
    engine.hands[1] = [by_cn[(Color.BLUE, 1)][0]]     # Unopened -> 202 * 2 = 404
    engine.hands[2] = [by_cn[(Color.YELLOW, 6)][0]]   # Series -> 6 * 2 = 12
    engine.hands[3] = [by_cn[(Color.BLACK, 4)][0]]    # Pairs -> 4 * 2 * 2 = 16

    res = engine.step(Action(ActionType.DISCARD, tile_id=by_cn[(Color.RED, 3)][0].tile_id))
    assert res.done
    assert engine.win_flags.win_by_pairs
    assert engine.win_flags.win_multiplier == 2
    assert res.round_scores == (-202, 404, 12, 16)


def test_scoring_draw_pile_empty() -> None:
    _, by_cn, _ = _deck_lookup()
    engine = Okey101Engine()
    engine.reset(seed=24)
    engine.joker_spec = JokerSpec.from_indicator(by_cn[(Color.BLACK, 1)][0])

    engine.draw_pile = []
    engine.open_types = [OpenType.SERIES, OpenType.NONE, OpenType.PAIRS, OpenType.NONE]
    engine.penalties = [101, 0, 0, 0]
    engine.current_player = 0
    engine.phase = TurnPhase.PLAY

    engine.hands[0] = [by_cn[(Color.RED, 4)][0], by_cn[(Color.RED, 9)][0]]
    engine.hands[1] = [by_cn[(Color.BLUE, 1)][0]]
    engine.hands[2] = [by_cn[(Color.YELLOW, 5)][0]]
    engine.hands[3] = [by_cn[(Color.BLACK, 3)][0]]

    # Player 0 discards R4 leaving R9 -> draw pile empty -> round ends with no winner bonus
    res = engine.step(Action(ActionType.DISCARD, tile_id=by_cn[(Color.RED, 4)][0].tile_id))
    assert res.done
    assert res.winner is None
    assert engine.win_flags.draw_pile_empty
    # P0: 9 (hand) + 101 (prior penalty) = 110; P1: 202; P2: 5*2=10; P3: 202
    assert res.round_scores == (110, 202, 10, 202)


# ---------------------------------------------------------------------------
# 8. End-to-End Randomized Simulation Fuzz Test
# ---------------------------------------------------------------------------


def test_full_random_games_tile_conservation() -> None:
    """Play 15 full 4-player games to completion, verifying invariants every step."""
    for seed in range(15):
        rng = random.Random(seed + 1000)
        engine = Okey101Engine(config=GameConfig(katlamali=(seed % 2 == 1)))
        engine.reset(seed=seed)

        steps = 0
        while not engine.done:
            legal = engine.get_legal_actions()
            assert legal, f"No legal actions on seed={seed}, step={steps}"

            # Prefer non-discard actions with some probability to exercise openings/melds/layoffs
            non_discards = [a for a in legal if a.action_type != ActionType.DISCARD]
            if non_discards and rng.random() < 0.8:
                action = rng.choice(non_discards)
            else:
                action = rng.choice(legal)

            engine.step(action)
            steps += 1

            # Verify exact conservation of all 106 physical tiles
            all_ids = (
                [t.tile_id for h in engine.hands for t in h]
                + [t.tile_id for t in engine.draw_pile]
                + [t.tile_id for d in engine.discard_piles for t in d]
                + [pt.tile.tile_id for tm in engine.table_melds for pt in tm.meld.placed_tiles]
            )
            assert len(all_ids) == 106
            assert len(set(all_ids)) == 106
            assert steps < 500

        assert engine.round_scores is not None
        assert len(engine.round_scores) == 4
