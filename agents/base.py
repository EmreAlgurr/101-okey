"""Abstract base interface for Official 101 Okey agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from engine import Action, Observation, Okey101Engine


class Agent(ABC):
    """Base class for all 101 Okey agents (Heuristic, IS-MCTS, RL, etc.).

    Agents must rely strictly on `obs` and `legal_actions` (never peeking at
    hidden opponent hands or draw pile ordering via `engine_ref`).
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name: str = name or self.__class__.__name__

    @abstractmethod
    def select_action(
        self,
        obs: Observation,
        legal_actions: Sequence[Action],
        engine_ref: Optional[Okey101Engine] = None,
    ) -> Action:
        """Choose one legal action given the player's information-set observation."""
        raise NotImplementedError

