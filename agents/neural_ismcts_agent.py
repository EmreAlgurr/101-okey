"""Neural Information Set Monte Carlo Tree Search (Neural IS-MCTS) Agent.

Uses a trained `PolicyValueNet` for:
- Masked Policy Head probabilities as PUCT priors (with strict +101 penalty filtering).
- Zero-rollout (or shallow) Neural Value Head leaf evaluation to eliminate
  Monte Carlo draw variance while retaining multi-turn tactical lookahead.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Optional, Sequence, Union
import numpy as np

from agents.base import Agent
from agents.heuristic_agent import _is_layable_anywhere
from agents.ismcts_agent import (
    _PUCTNode,
    _find_immediate_win,
    action_key,
    evaluate_world,
    sample_world,
)
from engine import (
    Action,
    ActionType,
    NUM_PLAYERS,
    Observation,
    Okey101Engine,
    OpenType,
    TurnPhase,
)
from ml.encoding import (
    action_to_index,
    encode_obs,
    legal_action_mask,
    normalize_score,
)
from ml.model import PolicyValueNet


def _filter_penalty_discards(
    obs: Observation,
    legal_actions: Sequence[Action],
) -> list[Action]:
    """Drop +101 işlek / real Joker discards whenever at least one safe discard exists."""
    if obs.phase != TurnPhase.PLAY or len(obs.hand) <= 1:
        return list(legal_actions)

    spec = obs.joker_spec
    hand_map = {t.tile_id: t for t in obs.hand}
    non_discards: list[Action] = []
    safe_discards: list[Action] = []
    penalty_discards: list[Action] = []

    for a in legal_actions:
        if a.action_type != ActionType.DISCARD:
            non_discards.append(a)
        else:
            assert a.tile_id is not None
            t = hand_map[a.tile_id]
            if spec.is_joker(t) or _is_layable_anywhere(t, obs.table_melds, spec):
                penalty_discards.append(a)
            else:
                safe_discards.append(a)

    discards = safe_discards if safe_discards else penalty_discards
    return non_discards + discards


def _neural_priors_and_value(
    model: PolicyValueNet,
    obs: Observation,
    legal_actions: Sequence[Action],
    top_k_discards: int = 6,
) -> tuple[list[Action], dict[tuple[object, ...], float], float]:
    """Run PolicyValueNet on `obs`, returning `(filtered_actions, puct_priors, value)`."""
    safe_actions = _filter_penalty_discards(obs, legal_actions)
    state_vec = encode_obs(obs)
    mask = legal_action_mask(obs, safe_actions)
    probs, val = model.predict_single(state_vec, mask)

    # Separate non-discards and discards; keep top-K discards by neural policy probability
    non_discards: list[tuple[Action, float]] = []
    discards: list[tuple[Action, float]] = []
    for a in safe_actions:
        p = float(probs[action_to_index(a, obs)])
        if a.action_type == ActionType.DISCARD:
            discards.append((a, p))
        else:
            non_discards.append((a, p))

    discards.sort(key=lambda x: x[1], reverse=True)
    kept = non_discards + discards[:top_k_discards]

    total_p = sum(p for _, p in kept)
    actions: list[Action] = []
    priors: dict[tuple[object, ...], float] = {}
    if total_p <= 1e-12:
        uniform = 1.0 / max(1, len(kept))
        for a, _ in kept:
            actions.append(a)
            priors[action_key(a)] = uniform
    else:
        for a, p in kept:
            actions.append(a)
            priors[action_key(a)] = p / total_p

    return actions, priors, val


class NeuralISMCTSAgent(Agent):
    """SO-ISMCTS powered by PolicyValueNet (zero-rollout neural leaf evaluation + PUCT)."""

    def __init__(
        self,
        model_or_path: Union[PolicyValueNet, str, Path] = "checkpoints/pv_net_v1.pt",
        iterations: int = 120,
        c_puct: float = 1.41,
        top_k_discards: int = 6,
        max_tree_depth: int = 6,
        katlamali: bool = False,
        name: str = "Neural-ISMCTS",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=name)
        if isinstance(model_or_path, PolicyValueNet):
            self.model = model_or_path
        else:
            # Load on CPU for <0.5ms single-sample MCTS inference
            self.model = PolicyValueNet.load_checkpoint(model_or_path, device="cpu")
        self.model.eval()

        self.iterations = iterations
        self.c_puct = c_puct
        self.top_k_discards = top_k_discards
        self.max_tree_depth = max_tree_depth
        self.katlamali = katlamali
        self.rng = random.Random(seed)

    def _evaluate_leaf(self, world: Okey101Engine, observer_id: int) -> list[float]:
        """Evaluate leaf state: exact terminal scores or Neural Value Head for observer."""
        if world.done and world.round_scores is not None:
            return [normalize_score(s) for s in world.round_scores]

        # Fast opponent estimates + exact neural Value Head for observer
        values = evaluate_world(world, observer_id=None)
        obs_leaf = world.get_observation(observer_id)
        values[observer_id] = self.model.predict_value(encode_obs(obs_leaf))
        return values

    def select_action(
        self,
        obs: Observation,
        legal_actions: Sequence[Action],
        engine_ref: Optional[Okey101Engine] = None,
    ) -> Action:
        """Run zero-rollout Neural SO-ISMCTS strictly from `obs`."""
        del engine_ref  # Never peek at hidden state!

        # 1. Trivial single-action fast path
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 2. Immediate winning move fast path (0 tiles left)
        if obs.phase == TurnPhase.PLAY:
            win_action = _find_immediate_win(obs.hand, legal_actions, obs.joker_spec)
            if win_action is not None:
                return win_action

        # 3. Root neural policy priors + penalty/Top-K filtering
        root_actions, root_priors, _ = _neural_priors_and_value(
            self.model, obs, legal_actions, top_k_discards=self.top_k_discards
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

            # Selection & Expansion (zero-rollout: expand 1 new node, evaluate with Value Head!)
            depth = 0
            while not world.done and depth < self.max_tree_depth:
                if node is root:
                    key_map = root_key_to_action
                    keys = root_keys
                    priors = root_priors
                else:
                    legal = world.get_legal_actions()
                    if not legal:
                        break
                    cur_obs = world.get_observation(world.current_player)
                    filtered, priors, _ = _neural_priors_and_value(
                        self.model,
                        cur_obs,
                        legal,
                        top_k_discards=self.top_k_discards,
                    )
                    key_map = {action_key(a): a for a in filtered}
                    keys = list(key_map.keys())

                for k in keys:
                    if k in node.children:
                        node.children[k].availability += 1

                chosen_key, existing_child = node.select_puct_child(
                    keys, priors, world.current_player, self.c_puct
                )

                if existing_child is None:
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

            # Zero-rollout Neural Leaf Evaluation & Backpropagation
            eval_vec = self._evaluate_leaf(world, observer_id)
            cur_node: Optional[_PUCTNode] = node
            while cur_node is not None:
                cur_node.visits += 1
                for i in range(NUM_PLAYERS):
                    cur_node.total_values[i] += eval_vec[i]
                cur_node = cur_node.parent

        best_key = max(
            (k for k in root_keys if k in root.children),
            key=lambda k: (
                root.children[k].visits,
                root.children[k].total_values[observer_id]
                / max(1, root.children[k].visits),
            ),
        )
        return root_key_to_action[best_key]

