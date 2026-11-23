"""Unit tests for HeuristicAgent and ISMCTSAgent (determinization, no-cheating, full games)."""

from __future__ import annotations

import random
import pytest

from agents.heuristic_agent import HeuristicAgent
from agents.ismcts_agent import (
    ISMCTSAgent,
    filter_and_prior_actions,
    sample_world,
)
from agents.planner_agent import PlannerAgent
from engine import (
    Action,
    ActionType,
    GameConfig,
    Okey101Engine,
    OpenType,
    TurnPhase,
)
from tiles import Color, JokerSpec, build_run, create_deck


class _ForbiddenEngineRef:
    """Sentinel object that fails immediately if an agent tries to peek at engine_ref."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(
            f"Agent attempted to access hidden engine_ref attribute '{name}'!"
        )


def test_determinization_preserves_106_tiles_and_is_stochastic() -> None:
    engine = Okey101Engine()
    engine.reset(seed=101)
    heur = HeuristicAgent(seed=1)

    # Play 8 actions so discards/table state evolve
    for _ in range(8):
        if engine.done:
            break
        p = engine.current_player
        obs = engine.get_observation(p)
        act = heur.select_action(obs, engine.get_legal_actions())
        engine.step(act)

    me = engine.current_player
    obs = engine.get_observation(me)

    seen_opp_hands: set[tuple[int, ...]] = set()
    for i in range(15):
        world = sample_world(obs, random.Random(500 + i))

        # Exact 106-tile conservation
        all_ids = (
            [t.tile_id for h in world.hands for t in h]
            + [t.tile_id for t in world.draw_pile]
            + [t.tile_id for d in world.discard_piles for t in d]
            + [pt.tile.tile_id for tm in world.table_melds for pt in tm.meld.placed_tiles]
        )
        assert len(all_ids) == 106
        assert len(set(all_ids)) == 106

        # Exact visible state match
        assert [t.tile_id for t in world.hands[me]] == [t.tile_id for t in obs.hand]
        assert tuple(len(h) for h in world.hands) == obs.hand_sizes
        assert len(world.draw_pile) == obs.draw_pile_size
        assert world.joker_spec == obs.joker_spec

        opp = (me + 1) % 4
        seen_opp_hands.add(tuple(sorted(t.tile_id for t in world.hands[opp])))

    # Stochastic determinization must generate multiple distinct opponent hands
    assert len(seen_opp_hands) > 5


def test_ismcts_never_peeks_at_engine_ref() -> None:
    engine = Okey101Engine()
    engine.reset(seed=55)
    obs = engine.get_observation(engine.current_player)
    legal = engine.get_legal_actions()

    agent = ISMCTSAgent(iterations=20, rollout_depth=2, seed=9)
    forbidden = _ForbiddenEngineRef()

    chosen = agent.select_action(obs, legal, engine_ref=forbidden)  # type: ignore[arg-type]
    assert chosen in legal


def test_heuristic_avoids_islek_and_joker_discard_and_takes_okey_win() -> None:
    deck = create_deck()
    by_cn: dict[tuple[Color, int], list] = {}
    for t in deck:
        if not t.is_false_joker and t.color is not None:
            by_cn.setdefault((t.color, t.number), []).append(t)

    engine = Okey101Engine()
    engine.reset(seed=1)
    engine.joker_spec = JokerSpec.from_indicator(by_cn[(Color.RED, 4)][0])  # Joker is R5
    real_joker = by_cn[(Color.RED, 5)][0]

    # Put R7-R8-R9 on table so R10 is işlek (+101 penalty if discarded)
    run_r789 = build_run(
        [by_cn[(Color.RED, 7)][0], by_cn[(Color.RED, 8)][0], by_cn[(Color.RED, 9)][0]],
        engine.joker_spec,
    )
    assert run_r789 is not None
    engine._add_table_meld(owner_id=1, meld=run_r789)

    safe_tile = by_cn[(Color.BLUE, 2)][0]
    islek_tile = by_cn[(Color.RED, 10)][0]
    engine.hands[0] = [real_joker, islek_tile, safe_tile]
    engine.open_types[0] = OpenType.NONE
    engine.current_player = 0
    engine.phase = TurnPhase.PLAY

    heur = HeuristicAgent(seed=0)
    obs = engine.get_observation(0)
    legal = engine.get_legal_actions()
    chosen = heur.select_action(obs, legal)
    assert chosen.action_type == ActionType.DISCARD
    assert chosen.tile_id == safe_tile.tile_id

    # Now give Player 0 ONLY the real Joker while already opened: must discard it for Okeyle bitme!
    engine.open_types[0] = OpenType.SERIES
    engine.hands[0] = [real_joker]
    obs2 = engine.get_observation(0)
    legal2 = engine.get_legal_actions()
    chosen2 = heur.select_action(obs2, legal2)
    assert chosen2.action_type == ActionType.DISCARD
    assert chosen2.tile_id == real_joker.tile_id


@pytest.mark.parametrize("katlamali", [False, True])
def test_full_4_player_game_ismcts_vs_heuristics(katlamali: bool) -> None:
    engine = Okey101Engine(config=GameConfig(katlamali=katlamali))
    engine.reset(seed=77 if not katlamali else 88)

    agents = [
        ISMCTSAgent(iterations=25, rollout_depth=3, katlamali=katlamali, seed=10),
        HeuristicAgent(seed=11),
        HeuristicAgent(seed=12),
        HeuristicAgent(seed=13),
    ]

    steps = 0
    while not engine.done:
        p = engine.current_player
        obs = engine.get_observation(p)
        legal = engine.get_legal_actions()
        assert legal
        act = agents[p].select_action(obs, legal)
        engine.step(act)
        steps += 1
        assert steps < 400

    assert engine.round_scores is not None
    assert len(engine.round_scores) == 4


def test_ismcts_top_k_pruning_and_priors() -> None:
    engine = Okey101Engine()
    engine.reset(seed=33)
    obs = engine.get_observation(0)
    legal = engine.get_legal_actions()
    discards = [a for a in legal if a.action_type == ActionType.DISCARD]
    assert len(discards) == 22

    filtered, priors = filter_and_prior_actions(obs, legal, top_k_discards=6)
    filtered_discards = [a for a in filtered if a.action_type == ActionType.DISCARD]
    assert len(filtered_discards) == 6
    assert abs(sum(priors.values()) - 1.0) < 1e-6
    assert all(p > 0.0 for p in priors.values())


def test_planner_never_peeks_and_plays_fast_legal_games() -> None:
    import time

    engine = Okey101Engine()
    obs = engine.reset(seed=91)
    legal = engine.get_legal_actions()

    planner = PlannerAgent(num_scenarios=40, seed=1)
    forbidden = _ForbiddenEngineRef()
    chosen = planner.select_action(obs, legal, engine_ref=forbidden)  # type: ignore[arg-type]
    assert chosen in legal

    for katlamali in (False, True):
        eng = Okey101Engine(config=GameConfig(katlamali=katlamali))
        eng.reset(seed=123 if not katlamali else 124)
        agents = [
            PlannerAgent(num_scenarios=40, seed=5),
            HeuristicAgent(seed=6),
            HeuristicAgent(seed=7),
            HeuristicAgent(seed=8),
        ]
        planner_ms: list[float] = []
        steps = 0
        while not eng.done:
            p = eng.current_player
            o = eng.get_observation(p)
            la = eng.get_legal_actions()
            t0 = time.perf_counter()
            act = agents[p].select_action(o, la)
            if p == 0:
                planner_ms.append((time.perf_counter() - t0) * 1000.0)
            assert act in la
            eng.step(act)
            steps += 1
            assert steps < 400
        assert eng.round_scores is not None
        avg_ms = sum(planner_ms) / max(1, len(planner_ms))
        assert avg_ms < 500.0, f"Planner too slow: {avg_ms:.2f}ms/act"


def test_planner_v2_hazard_and_feed_penalty() -> None:
    from agents.heuristic_agent import _count_visible_keys
    from agents.planner_agent import next_player_feed_penalty, opponent_cycle_finish_hazard

    eng = Okey101Engine()
    eng.reset(seed=50)
    obs0 = eng.get_observation(0)
    h_early = opponent_cycle_finish_hazard(obs0)

    # Opened opponent with 2 tiles left must raise finish hazard sharply
    eng.open_types[1] = OpenType.SERIES
    eng.hands[1] = eng.hands[1][:2]
    obs1 = eng.get_observation(0)
    h_late = opponent_cycle_finish_hazard(obs1)
    assert h_late > h_early + 0.25

    # Feeding a live middle tile (7) must carry higher feed penalty than an edge tile (1)
    vis = _count_visible_keys(obs1)
    deck = create_deck()
    t1 = next(t for t in deck if not t.is_false_joker and t.color == Color.RED and t.number == 1)
    t7 = next(t for t in deck if not t.is_false_joker and t.color == Color.RED and t.number == 7)
    if not eng.joker_spec.is_joker(t1) and not eng.joker_spec.is_joker(t7):
        assert next_player_feed_penalty(t7, obs1, vis) > next_player_feed_penalty(t1, obs1, vis)




