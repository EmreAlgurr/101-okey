"""Headless tournament benchmark: 1 IS-MCTS (or Neural IS-MCTS) vs 3 Heuristic Agents.

Rotates seat positions every round to eliminate seat / first-player bias.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Optional

from agents.base import Agent
from agents.heuristic_agent import HeuristicAgent
from agents.ismcts_agent import ISMCTSAgent
from engine import GameConfig, NUM_PLAYERS, Okey101Engine, OpenType


@dataclass
class AgentStats:
    """Aggregated performance metrics across tournament rounds."""

    name: str
    rounds_played: int = 0
    total_score: int = 0
    wins: int = 0
    opens: int = 0
    total_penalties_101: int = 0
    decisions: int = 0
    total_decision_ms: float = 0.0

    @property
    def avg_score(self) -> float:
        return self.total_score / max(1, self.rounds_played)

    @property
    def win_rate_pct(self) -> float:
        return 100.0 * self.wins / max(1, self.rounds_played)

    @property
    def open_rate_pct(self) -> float:
        return 100.0 * self.opens / max(1, self.rounds_played)

    @property
    def avg_penalties_per_round(self) -> float:
        return self.total_penalties_101 / max(1, self.rounds_played)

    @property
    def avg_decision_ms(self) -> float:
        return self.total_decision_ms / max(1, self.decisions)


def run_benchmark(
    num_rounds: int = 100,
    mcts_iterations: int = 120,
    rollout_depth: int = 12,
    agent_type: str = "ismcts",
    checkpoint: str = "checkpoints/pv_net_v1.pt",
    katlamali: bool = False,
    seed: int = 42,
    rollouts: int = 80,
    top_k: int = 10,
) -> tuple[AgentStats, AgentStats]:
    """Run `num_rounds` of Challenger vs Baseline with rotating seats."""
    shared_model = None
    if agent_type == "neural":
        from agents.neural_ismcts_agent import NeuralISMCTSAgent
        from ml.model import PolicyValueNet

        shared_model = PolicyValueNet.load_checkpoint(checkpoint, device="cpu")
        challenger_label = f"Neural-ISMCTS ({mcts_iterations})"
        baseline_label = "Heuristic (3x)"
    elif agent_type == "planner":
        from agents.planner_agent import PlannerAgent

        challenger_label = f"Planner v2 ({rollouts} scen)"
        baseline_label = "Heuristic (3x)"
    elif agent_type == "v2_vs_v1":
        from agents.planner_agent import PlannerAgent

        challenger_label = f"Planner v2 ({rollouts} scen)"
        baseline_label = "Planner v1 (40 scen)"
    else:
        challenger_label = f"IS-MCTS ({mcts_iterations} iters)"
        baseline_label = "Heuristic (3x)"

    challenger_stats = AgentStats(name=challenger_label)
    heuristic_stats = AgentStats(name=baseline_label)

    config = GameConfig(katlamali=katlamali)
    t_start = time.perf_counter()

    for r in range(num_rounds):
        if agent_type == "v2_vs_v1":
            challenger_seats = {0, 2} if (r % 2 == 0) else {1, 3}
        else:
            challenger_seats = {r % NUM_PLAYERS}

        agents: list[Agent] = []
        for seat in range(NUM_PLAYERS):
            if seat in challenger_seats:
                if agent_type in ("planner", "v2_vs_v1"):
                    agents.append(
                        PlannerAgent(
                            num_scenarios=rollouts,
                            top_k=top_k,
                            version=2,
                            seed=seed + r * 17 + seat,
                        )
                    )
                elif agent_type == "neural":
                    assert shared_model is not None
                    agents.append(
                        NeuralISMCTSAgent(
                            model_or_path=shared_model,
                            iterations=mcts_iterations,
                            katlamali=katlamali,
                            seed=seed + r * 17 + seat,
                        )
                    )
                else:
                    agents.append(
                        ISMCTSAgent(
                            iterations=mcts_iterations,
                            rollout_depth=rollout_depth,
                            katlamali=katlamali,
                            seed=seed + r * 17 + seat,
                        )
                    )
            else:
                if agent_type == "v2_vs_v1":
                    agents.append(
                        PlannerAgent(
                            num_scenarios=40,
                            top_k=10,
                            tiebreak=0.15,
                            version=1,
                            seed=seed + r * 31 + seat,
                        )
                    )
                else:
                    agents.append(HeuristicAgent(seed=seed + r * 31 + seat))

        engine = Okey101Engine(config=config)
        engine.reset(seed=seed + r, starting_player=0)

        while not engine.done:
            p = engine.current_player
            obs = engine.get_observation(p)
            legal = engine.get_legal_actions()

            t0 = time.perf_counter()
            action = agents[p].select_action(obs, legal)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            stats = challenger_stats if p in challenger_seats else heuristic_stats
            stats.decisions += 1
            stats.total_decision_ms += dt_ms

            engine.step(action)

        assert engine.round_scores is not None
        for seat in range(NUM_PLAYERS):
            stats = challenger_stats if seat in challenger_seats else heuristic_stats
            stats.rounds_played += 1
            stats.total_score += engine.round_scores[seat]
            if engine.winner == seat:
                stats.wins += 1
            if engine.open_types[seat] != OpenType.NONE:
                stats.opens += 1
            stats.total_penalties_101 += engine.penalties[seat] // 101

    elapsed_s = time.perf_counter() - t_start

    desc = (
        f"Neural-ISMCTS({mcts_iterations} iters, ckpt={checkpoint}) vs 3x Heuristic"
        if agent_type == "neural"
        else f"Planner v2({rollouts} scen, top_k={top_k}) vs 3x Heuristic"
        if agent_type == "planner"
        else f"2x Planner v2({rollouts}) vs 2x Planner v1(40)"
        if agent_type == "v2_vs_v1"
        else f"IS-MCTS({mcts_iterations} iters, depth={rollout_depth}) vs 3x Heuristic"
    )
    print()
    print(
        f"Official 101 Okey Benchmark | {num_rounds} rounds | "
        f"{desc} | katlamali={katlamali} | {elapsed_s:.2f}s"
    )
    print("-" * 92)
    header = (
        f"{'Agent':<24} | {'Avg Score':>10} | {'Win Rate':>9} | "
        f"{'Open Rate':>9} | {'+101/Rnd':>8} | {'Avg ms/Act':>10}"
    )
    print(header)
    print("-" * 92)
    for st in (challenger_stats, heuristic_stats):
        print(
            f"{st.name:<24} | {st.avg_score:>10.2f} | {st.win_rate_pct:>8.1f}% | "
            f"{st.open_rate_pct:>8.1f}% | {st.avg_penalties_per_round:>8.3f} | "
            f"{st.avg_decision_ms:>9.2f}ms"
        )
    print("-" * 92)
    print("Note: In Official 101 Okey, LOWER average score is better.")
    print()
    return challenger_stats, heuristic_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark IS-MCTS / Neural IS-MCTS vs Heuristic agents in Official 101 Okey."
    )
    parser.add_argument(
        "--agent",
        choices=("ismcts", "neural", "planner", "v2_vs_v1"),
        default="ismcts",
        help="Challenger agent type (default: ismcts).",
    )
    parser.add_argument(
        "--rollouts", "--num-scenarios", dest="rollouts", type=int, default=80,
        help="Planner scenarios per candidate.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Planner discard candidates.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pv_net_v1.pt",
        help="Checkpoint path when --agent neural is used.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick sanity check mode (20 rounds, 50 MCTS iterations).",
    )
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds.")
    parser.add_argument(
        "--iterations", type=int, default=None, help="MCTS iterations per decision."
    )
    parser.add_argument(
        "--rollout-depth", type=int, default=12, help="Playout depth (ismcts only)."
    )
    parser.add_argument(
        "--katlamali", action="store_true", help="Enable katlamali opening rule."
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    args = parser.parse_args()

    num_rounds = args.rounds if args.rounds is not None else (20 if args.fast else 100)
    iterations = (
        args.iterations if args.iterations is not None else (50 if args.fast else 120)
    )

    run_benchmark(
        num_rounds=num_rounds,
        mcts_iterations=iterations,
        rollout_depth=args.rollout_depth,
        agent_type=args.agent,
        checkpoint=args.checkpoint,
        katlamali=args.katlamali,
        seed=args.seed,
        rollouts=args.rollouts,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
