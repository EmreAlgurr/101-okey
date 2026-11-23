"""Unit tests for Phase 3 ML encoding, PolicyValueNet, and NeuralISMCTSAgent."""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import torch

from agents.heuristic_agent import HeuristicAgent
from agents.neural_ismcts_agent import NeuralISMCTSAgent
from engine import ActionType, Okey101Engine
from ml.encoding import (
    ACTION_DIM,
    STATE_DIM,
    action_to_index,
    encode_obs,
    index_to_action,
    legal_action_mask,
    normalize_score,
)
from ml.model import PolicyValueNet


def test_encoding_shapes_and_action_masking() -> None:
    engine = Okey101Engine()
    obs = engine.reset(seed=100)
    legal = engine.get_legal_actions()

    vec = encode_obs(obs)
    assert vec.shape == (STATE_DIM,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()

    mask = legal_action_mask(obs, legal)
    assert mask.shape == (ACTION_DIM,)
    assert mask.dtype == np.bool_
    assert mask.sum() >= 1

    # Round-trip every legal action index back to a legal action
    for a in legal:
        idx = action_to_index(a, obs)
        assert mask[idx]
        recovered = index_to_action(idx, obs, legal)
        assert recovered in legal

    # Score normalization: lower score -> higher value in [-1, 1]
    assert normalize_score(-202) > normalize_score(-101) > normalize_score(0) > normalize_score(202)
    assert -1.0 <= normalize_score(808) <= 1.0
    assert -1.0 <= normalize_score(-404) <= 1.0


def test_policy_value_net_forward_masking_and_speed(tmp_path: Path) -> None:
    engine = Okey101Engine()
    obs = engine.reset(seed=101)
    legal = engine.get_legal_actions()
    vec = encode_obs(obs)
    mask = legal_action_mask(obs, legal)

    model = PolicyValueNet(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    model.eval()

    # Batch forward pass
    x = torch.from_numpy(np.stack([vec, vec], axis=0))
    logits, val = model(x)
    assert logits.shape == (2, ACTION_DIM)
    assert val.shape == (2,)
    assert torch.all((val >= -1.0) & (val <= 1.0))

    # Single-sample inference: illegal actions must have ~0 probability
    probs, v = model.predict_single(vec, mask)
    assert probs.shape == (ACTION_DIM,)
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert float(probs[~mask].max()) < 1e-6
    assert -1.0 <= v <= 1.0

    # Checkpoint save & reload
    ckpt = tmp_path / "test_pv.pt"
    model.save_checkpoint(ckpt)
    reloaded = PolicyValueNet.load_checkpoint(ckpt, device="cpu")
    probs2, v2 = reloaded.predict_single(vec, mask)
    assert np.allclose(probs, probs2, atol=1e-5)
    assert abs(v - v2) < 1e-5

    # Speed sanity check: single-sample inference should be fast (<5ms even on cold CPU)
    t0 = time.perf_counter()
    for _ in range(20):
        reloaded.predict_single(vec, mask)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / 20.0
    assert avg_ms < 5.0, f"Single-sample inference too slow: {avg_ms:.2f}ms"


def test_neural_ismcts_full_4_player_game() -> None:
    model = PolicyValueNet(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    model.eval()

    engine = Okey101Engine()
    engine.reset(seed=202)
    agents = [
        NeuralISMCTSAgent(model_or_path=model, iterations=20, seed=1),
        HeuristicAgent(seed=2),
        HeuristicAgent(seed=3),
        HeuristicAgent(seed=4),
    ]

    steps = 0
    while not engine.done:
        p = engine.current_player
        obs = engine.get_observation(p)
        legal = engine.get_legal_actions()
        act = agents[p].select_action(obs, legal)
        assert act in legal
        engine.step(act)
        steps += 1
        assert steps < 400

    assert engine.round_scores is not None
    assert len(engine.round_scores) == 4

