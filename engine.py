"""Headless 4-player Official Professional 101 Okey Game Engine.

Designed for exact rule enforcement and high-throughput IS-MCTS / RL rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Optional, Sequence

from solver import solve_best_opening, solve_best_pairs, solve_min_deadwood
from tiles import (
    JokerSpec,
    Meld,
    MeldType,
    Tile,
    apply_lay_off,
    can_lay_off,
    create_deck,
    validate_meld,
)

NUM_PLAYERS = 4


class OpenType(Enum):
    """How a player has opened their hand in the current round."""

    NONE = "none"
    SERIES = "series"
    PAIRS = "pairs"


class TurnPhase(Enum):
    """Sub-phase within the current player's turn."""

    DRAW = "draw"
    PLAY = "play"


class ActionType(Enum):
    """Legal action types in 101 Okey."""

    DRAW_DECK = "draw_deck"
    TAKE_DISCARD = "take_discard"
    OPEN_SERIES = "open_series"
    OPEN_PAIRS = "open_pairs"
    MELD = "meld"
    LAY_OFF = "lay_off"
    DISCARD = "discard"


@dataclass(frozen=True, slots=True)
class Action:
    """Action submitted to `Okey101Engine.step(action)`."""

    action_type: ActionType
    tile_id: Optional[int] = None
    melds: tuple[Meld, ...] = ()
    meld_id: Optional[int] = None
    lay_off_side: str = "auto"


@dataclass(slots=True)
class TableMeld:
    """A meld laid down on the table."""

    meld_id: int
    owner_id: int
    meld: Meld

    def copy(self) -> TableMeld:
        return TableMeld(
            meld_id=self.meld_id,
            owner_id=self.owner_id,
            meld=self.meld.copy(),
        )


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Configuration options for Official 101 Okey (and optional house rules)."""

    katlamali: bool = False
    feed_penalty_multiplier: int = 0


@dataclass(frozen=True, slots=True)
class WinFlags:
    """Breakdown of how the round ended and win multipliers applied."""

    win_by_okey: bool = False
    win_by_elden: bool = False
    win_by_pairs: bool = False
    win_multiplier: int = 1
    draw_pile_empty: bool = False


@dataclass(frozen=True, slots=True)
class Observation:
    """Information-set observation visible to a single player (IS-MCTS / RL ready)."""

    player_id: int
    hand: tuple[Tile, ...]
    joker_spec: JokerSpec
    current_player: int
    phase: TurnPhase
    draw_pile_size: int
    hand_sizes: tuple[int, int, int, int]
    discard_piles: tuple[tuple[Tile, ...], ...]
    table_melds: tuple[TableMeld, ...]
    open_types: tuple[OpenType, ...]
    penalties: tuple[int, int, int, int]
    required_series_points: int
    required_pairs_count: int
    must_use_tile_id: Optional[int]
    done: bool
    winner: Optional[int]
    win_flags: WinFlags
    round_scores: Optional[tuple[int, int, int, int]]
    feed_penalty_multiplier: int = 0


@dataclass(slots=True)
class StepResult:
    """Return value of `Okey101Engine.step(action)`."""

    observation: Observation
    done: bool
    winner: Optional[int]
    round_scores: Optional[tuple[int, int, int, int]]
    penalty_incurred: int = 0
    info: dict[str, object] = field(default_factory=dict)


class Okey101Engine:
    """Headless 4-player Official Professional 101 Okey game engine."""

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        feed_penalty_multiplier: Optional[int] = None,
    ) -> None:
        base_cfg = config or GameConfig()
        if feed_penalty_multiplier is not None:
            base_cfg = GameConfig(
                katlamali=base_cfg.katlamali,
                feed_penalty_multiplier=feed_penalty_multiplier,
            )
        self.config: GameConfig = base_cfg
        self.rng: random.Random = random.Random()

        self.hands: list[list[Tile]] = [[] for _ in range(NUM_PLAYERS)]
        self.draw_pile: list[Tile] = []
        self.discard_piles: list[list[Tile]] = [[] for _ in range(NUM_PLAYERS)]
        self.table_melds: list[TableMeld] = []
        self.open_types: list[OpenType] = [OpenType.NONE] * NUM_PLAYERS
        self.penalties: list[int] = [0] * NUM_PLAYERS

        self.joker_spec: JokerSpec = JokerSpec.from_indicator(create_deck()[0])
        self.current_player: int = 0
        self.phase: TurnPhase = TurnPhase.PLAY
        self.must_use_tile_id: Optional[int] = None
        self.opened_before_turn: bool = False
        self.highest_series_open: int = 100
        self.highest_pairs_open: int = 4
        self._next_meld_id: int = 0

        self.done: bool = False
        self.winner: Optional[int] = None
        self.win_flags: WinFlags = WinFlags()
        self.round_scores: Optional[tuple[int, int, int, int]] = None
        self.turn_count: int = 0

    @property
    def required_series_points(self) -> int:
        if self.config.katlamali:
            return self.highest_series_open + 1
        return 101

    @property
    def required_pairs_count(self) -> int:
        if self.config.katlamali:
            return self.highest_pairs_open + 1
        return 5

    def reset(self, seed: Optional[int] = None, starting_player: int = 0) -> Observation:
        """Deal a fresh 106-tile Official 101 Okey round.

        - Starting player receives 22 tiles and begins in TurnPhase.PLAY (no initial draw).
        - Other 3 players receive 21 tiles each.
        - Remaining 21 tiles form the center draw pile.
        - One non-False-Joker tile from the draw pile is flipped as the indicator.
        """
        if not (0 <= starting_player < NUM_PLAYERS):
            raise ValueError(f"starting_player must be 0..3, got {starting_player}.")

        self.rng = random.Random(seed)
        deck = create_deck()
        self.rng.shuffle(deck)

        # Deal 22 to starting_player, 21 to the other 3 players (85 total)
        self.hands = [[] for _ in range(NUM_PLAYERS)]
        cursor = 0
        for p in range(NUM_PLAYERS):
            count = 22 if p == starting_player else 21
            self.hands[p] = deck[cursor : cursor + count]
            cursor += count

        # Remaining 21 tiles form the center draw pile
        self.draw_pile = deck[cursor:]
        assert len(self.draw_pile) == 21

        # Pick indicator: first non-False-Joker tile in draw_pile (placed at bottom of pile)
        ind_idx = next(
            i for i, t in enumerate(self.draw_pile) if not t.is_false_joker
        )
        indicator = self.draw_pile[ind_idx]
        # Move indicator to index 0 (bottom of draw pile; pop() draws from top/end)
        self.draw_pile[0], self.draw_pile[ind_idx] = (
            self.draw_pile[ind_idx],
            self.draw_pile[0],
        )
        self.joker_spec = JokerSpec.from_indicator(indicator)

        self.discard_piles = [[] for _ in range(NUM_PLAYERS)]
        self.table_melds = []
        self.open_types = [OpenType.NONE] * NUM_PLAYERS
        self.penalties = [0] * NUM_PLAYERS
        self.current_player = starting_player
        self.phase = TurnPhase.PLAY  # Starting player already has 22 tiles
        self.must_use_tile_id = None
        self.opened_before_turn = False
        self.highest_series_open = 100
        self.highest_pairs_open = 4
        self._next_meld_id = 0

        self.done = False
        self.winner = None
        self.win_flags = WinFlags()
        self.round_scores = None
        self.turn_count = 0

        return self.get_observation(self.current_player)

    def clone(self) -> Okey101Engine:
        """Return a deep copy of the engine state (useful for IS-MCTS rollouts)."""
        copy_eng = Okey101Engine(config=self.config)
        copy_eng.hands = [list(h) for h in self.hands]
        copy_eng.draw_pile = list(self.draw_pile)
        copy_eng.discard_piles = [list(d) for d in self.discard_piles]
        copy_eng.table_melds = [tm.copy() for tm in self.table_melds]
        copy_eng.open_types = list(self.open_types)
        copy_eng.penalties = list(self.penalties)
        copy_eng.joker_spec = self.joker_spec
        copy_eng.current_player = self.current_player
        copy_eng.phase = self.phase
        copy_eng.must_use_tile_id = self.must_use_tile_id
        copy_eng.opened_before_turn = self.opened_before_turn
        copy_eng.highest_series_open = self.highest_series_open
        copy_eng.highest_pairs_open = self.highest_pairs_open
        copy_eng._next_meld_id = self._next_meld_id
        copy_eng.done = self.done
        copy_eng.winner = self.winner
        copy_eng.win_flags = self.win_flags
        copy_eng.round_scores = self.round_scores
        copy_eng.turn_count = self.turn_count
        return copy_eng

    def can_tile_lay_off_anywhere(self, tile: Tile) -> bool:
        """Return True iff `tile` can legally be laid off onto any open Run/Set on the table."""
        for tm in self.table_melds:
            if can_lay_off(tm.meld, tile, self.joker_spec):
                return True
        return False

    def _previous_player(self, player_id: Optional[int] = None) -> int:
        p = self.current_player if player_id is None else player_id
        return (p - 1) % NUM_PLAYERS

    def can_take_discard(self, player_id: Optional[int] = None) -> bool:
        """Check whether `player_id` can legally take the previous player's top discard.

        Official 101 Okey rule: Taking the previous player's discard is ONLY legal
        if the player immediately uses that exact tile in the same turn to open
        their hand (or, if already open, to form a new meld or lay off).
        """
        p = self.current_player if player_id is None else player_id
        prev_p = self._previous_player(p)
        if not self.discard_piles[prev_p]:
            return False

        top_tile = self.discard_piles[prev_p][-1]
        candidate_hand = self.hands[p] + [top_tile]
        open_type = self.open_types[p]

        if open_type == OpenType.NONE:
            series_sol = solve_best_opening(
                candidate_hand,
                self.joker_spec,
                min_points=self.required_series_points,
                required_tile_id=top_tile.tile_id,
            )
            if series_sol.meets_threshold:
                return True
            pairs_sol = solve_best_pairs(
                candidate_hand,
                self.joker_spec,
                min_pairs=self.required_pairs_count,
                required_tile_id=top_tile.tile_id,
            )
            return pairs_sol.meets_threshold

        if open_type == OpenType.SERIES:
            if self.can_tile_lay_off_anywhere(top_tile):
                return True
            dw_sol = solve_min_deadwood(
                candidate_hand,
                self.joker_spec,
                required_tile_id=top_tile.tile_id,
            )
            return len(dw_sol.melds) > 0

        if open_type == OpenType.PAIRS:
            pairs_sol = solve_best_pairs(
                candidate_hand,
                self.joker_spec,
                min_pairs=1,
                required_tile_id=top_tile.tile_id,
            )
            return pairs_sol.pair_count >= 1

        return False

    def get_legal_actions(self) -> list[Action]:
        """Return the set of legal actions for the current player and phase.

        Note: In `PLAY` phase, optimal opening/meld proposals are generated via the
        exact hand solver to keep the action space tractable for IS-MCTS / RL agents.
        `step(action)` also accepts and strictly validates any custom legal melds.
        """
        if self.done:
            return []

        p = self.current_player
        actions: list[Action] = []

        if self.phase == TurnPhase.DRAW:
            if self.draw_pile:
                actions.append(Action(ActionType.DRAW_DECK))
            if self.can_take_discard(p):
                actions.append(Action(ActionType.TAKE_DISCARD))
            return actions

        # TurnPhase.PLAY
        hand = self.hands[p]
        open_type = self.open_types[p]

        if open_type == OpenType.NONE:
            # Check optimal Series/Set opening
            series_sol = solve_best_opening(
                hand,
                self.joker_spec,
                min_points=self.required_series_points,
                required_tile_id=self.must_use_tile_id,
            )
            if series_sol.meets_threshold:
                actions.append(
                    Action(ActionType.OPEN_SERIES, melds=tuple(series_sol.melds))
                )

            # Check optimal Pairs opening
            pairs_sol = solve_best_pairs(
                hand,
                self.joker_spec,
                min_pairs=self.required_pairs_count,
                required_tile_id=self.must_use_tile_id,
            )
            if pairs_sol.meets_threshold:
                actions.append(
                    Action(ActionType.OPEN_PAIRS, melds=tuple(pairs_sol.pairs))
                )

        elif open_type == OpenType.SERIES:
            # 1. Lay-off actions onto existing table Runs/Sets
            for t in hand:
                for tm in self.table_melds:
                    for side in can_lay_off(tm.meld, t, self.joker_spec):
                        actions.append(
                            Action(
                                ActionType.LAY_OFF,
                                tile_id=t.tile_id,
                                meld_id=tm.meld_id,
                                lay_off_side=side,
                            )
                        )
            # 2. New melds (Runs/Sets) from hand
            dw_sol = solve_min_deadwood(
                hand,
                self.joker_spec,
                required_tile_id=self.must_use_tile_id,
            )
            if dw_sol.melds:
                actions.append(
                    Action(ActionType.MELD, melds=tuple(dw_sol.melds))
                )

        elif open_type == OpenType.PAIRS:
            # Pairs openers can ONLY add more pairs (cannot lay off onto runs/sets)
            pairs_sol = solve_best_pairs(
                hand,
                self.joker_spec,
                min_pairs=1,
                required_tile_id=self.must_use_tile_id,
            )
            if pairs_sol.pairs:
                actions.append(
                    Action(ActionType.MELD, melds=tuple(pairs_sol.pairs))
                )

        # Discards are legal ONLY if the player does not have an unused taken discard
        if self.must_use_tile_id is None:
            for t in hand:
                actions.append(Action(ActionType.DISCARD, tile_id=t.tile_id))

        return actions

    def get_observation(self, player_id: int) -> Observation:
        """Return the information-set observation for `player_id`."""
        return Observation(
            player_id=player_id,
            hand=tuple(self.hands[player_id]),
            joker_spec=self.joker_spec,
            current_player=self.current_player,
            phase=self.phase,
            draw_pile_size=len(self.draw_pile),
            hand_sizes=tuple(len(h) for h in self.hands),  # type: ignore[arg-type]
            discard_piles=tuple(tuple(d) for d in self.discard_piles),
            table_melds=tuple(tm.copy() for tm in self.table_melds),
            open_types=tuple(self.open_types),
            penalties=tuple(self.penalties),  # type: ignore[arg-type]
            required_series_points=self.required_series_points,
            required_pairs_count=self.required_pairs_count,
            must_use_tile_id=(
                self.must_use_tile_id if player_id == self.current_player else None
            ),
            done=self.done,
            winner=self.winner,
            win_flags=self.win_flags,
            round_scores=self.round_scores,
            feed_penalty_multiplier=self.config.feed_penalty_multiplier,
        )

    def _remove_tiles_from_hand(self, player_id: int, tile_ids: Sequence[int]) -> list[Tile]:
        """Validate all `tile_ids` are distinct and in `hands[player_id]`, then remove them."""
        if len(set(tile_ids)) != len(tile_ids):
            raise ValueError("Duplicate tile_id in meld/action.")
        hand_map = {t.tile_id: t for t in self.hands[player_id]}
        missing = [tid for tid in tile_ids if tid not in hand_map]
        if missing:
            raise ValueError(f"Tiles {missing} are not in player {player_id}'s hand.")
        removed = [hand_map[tid] for tid in tile_ids]
        remove_set = set(tile_ids)
        self.hands[player_id] = [
            t for t in self.hands[player_id] if t.tile_id not in remove_set
        ]
        if self.must_use_tile_id is not None and self.must_use_tile_id in remove_set:
            self.must_use_tile_id = None
        return removed

    def _add_table_meld(self, owner_id: int, meld: Meld) -> TableMeld:
        tm = TableMeld(meld_id=self._next_meld_id, owner_id=owner_id, meld=meld.copy())
        self._next_meld_id += 1
        self.table_melds.append(tm)
        return tm

    def _finalize_round(
        self,
        winner: Optional[int],
        win_by_okey: bool = False,
        draw_pile_empty: bool = False,
    ) -> None:
        """Compute official 101 Okey round scores and mark the round done."""
        self.done = True
        self.winner = winner

        if winner is None:
            # Draw pile exhausted: no winner bonus, everyone gets standard hand penalty + penalties
            self.win_flags = WinFlags(
                win_by_okey=False,
                win_by_elden=False,
                win_by_pairs=False,
                win_multiplier=1,
                draw_pile_empty=draw_pile_empty,
            )
            scores: list[int] = []
            for p in range(NUM_PLAYERS):
                hand_pen = self._base_hand_penalty(p)
                scores.append(hand_pen + self.penalties[p])
            self.round_scores = tuple(scores)  # type: ignore[assignment]
            return

        win_by_elden = not self.opened_before_turn
        win_by_pairs = self.open_types[winner] == OpenType.PAIRS

        mult = 1
        if win_by_okey:
            mult *= 2
        if win_by_elden:
            mult *= 2
        if win_by_pairs:
            mult *= 2

        self.win_flags = WinFlags(
            win_by_okey=win_by_okey,
            win_by_elden=win_by_elden,
            win_by_pairs=win_by_pairs,
            win_multiplier=mult,
            draw_pile_empty=False,
        )

        scores = [0] * NUM_PLAYERS
        # Winner receives negative bonus + any in-round penalties
        scores[winner] = (-101 * mult) + self.penalties[winner]

        for p in range(NUM_PLAYERS):
            if p == winner:
                continue
            hand_pen = self._base_hand_penalty(p) * mult
            scores[p] = hand_pen + self.penalties[p]

        self.round_scores = tuple(scores)  # type: ignore[assignment]

    def _base_hand_penalty(self, player_id: int) -> int:
        """Base hand penalty before winner's multiplier is applied.

        - Unopened: +202
        - Opened (Series): sum of face values (real Joker = +101)
        - Opened (Pairs): 2 * sum of face values (real Joker = +101)
        """
        ot = self.open_types[player_id]
        if ot == OpenType.NONE:
            return 202
        tile_sum = sum(
            self.joker_spec.hand_penalty_value(t) for t in self.hands[player_id]
        )
        if ot == OpenType.PAIRS:
            return tile_sum * 2
        return tile_sum

    def _advance_turn(self) -> None:
        self.turn_count += 1
        self.current_player = (self.current_player + 1) % NUM_PLAYERS
        self.phase = TurnPhase.DRAW
        self.must_use_tile_id = None
        self.opened_before_turn = self.open_types[self.current_player] != OpenType.NONE

    def step(self, action: Action) -> StepResult:
        """Execute and strictly validate one action for `self.current_player`."""
        if self.done:
            raise ValueError("Cannot step: round is already finished.")

        p = self.current_player
        atype = action.action_type
        penalty_incurred = 0
        info: dict[str, object] = {}

        # ---------------- DRAW PHASE ----------------
        if self.phase == TurnPhase.DRAW:
            if atype == ActionType.DRAW_DECK:
                if not self.draw_pile:
                    raise ValueError("Draw pile is empty.")
                drawn = self.draw_pile.pop()
                self.hands[p].append(drawn)
                self.phase = TurnPhase.PLAY
                info["drawn_tile"] = drawn
                return StepResult(
                    observation=self.get_observation(self.current_player),
                    done=False,
                    winner=None,
                    round_scores=None,
                    info=info,
                )

            if atype == ActionType.TAKE_DISCARD:
                if not self.can_take_discard(p):
                    raise ValueError(
                        "Illegal TAKE_DISCARD: tile cannot be immediately used to open, meld, or lay off."
                    )
                prev_p = self._previous_player(p)
                taken = self.discard_piles[prev_p].pop()
                self.hands[p].append(taken)
                self.must_use_tile_id = taken.tile_id
                self.phase = TurnPhase.PLAY
                info["taken_tile"] = taken

                # Optional house rule: Discard-Feeding Penalty (prev_player pays tile.value * mult)
                if self.config.feed_penalty_multiplier > 0:
                    eff_val = self.joker_spec.effective_value(taken)
                    tile_val = eff_val if eff_val is not None else self.joker_spec.joker_number
                    feed_pen = tile_val * self.config.feed_penalty_multiplier
                    self.penalties[prev_p] += feed_pen
                    info["feed_penalty_player"] = prev_p
                    info["feed_penalty_amount"] = feed_pen

                return StepResult(
                    observation=self.get_observation(self.current_player),
                    done=False,
                    winner=None,
                    round_scores=None,
                    info=info,
                )

            raise ValueError(f"Action {atype} is illegal during DRAW phase.")

        # ---------------- PLAY PHASE ----------------
        if atype == ActionType.OPEN_SERIES:
            if self.open_types[p] != OpenType.NONE:
                raise ValueError(f"Player {p} has already opened.")
            if not action.melds:
                raise ValueError("OPEN_SERIES requires at least one meld.")

            total_pts = 0
            all_tids: list[int] = []
            for m in action.melds:
                if m.meld_type not in (MeldType.RUN, MeldType.SET):
                    raise ValueError("OPEN_SERIES only accepts RUN and SET melds.")
                if not validate_meld(m, self.joker_spec):
                    raise ValueError(f"Invalid meld in OPEN_SERIES: {m}")
                total_pts += m.points
                all_tids.extend(pt.tile.tile_id for pt in m.placed_tiles)

            if total_pts < self.required_series_points:
                raise ValueError(
                    f"OPEN_SERIES points {total_pts} < required {self.required_series_points}."
                )

            self._remove_tiles_from_hand(p, all_tids)
            for m in action.melds:
                self._add_table_meld(p, m)

            self.open_types[p] = OpenType.SERIES
            if self.config.katlamali:
                self.highest_series_open = max(self.highest_series_open, total_pts)
            info["opened_points"] = total_pts

            if not self.hands[p]:
                self._finalize_round(winner=p, win_by_okey=False)

            return StepResult(
                observation=self.get_observation(self.current_player),
                done=self.done,
                winner=self.winner,
                round_scores=self.round_scores,
                info=info,
            )

        if atype == ActionType.OPEN_PAIRS:
            if self.open_types[p] != OpenType.NONE:
                raise ValueError(f"Player {p} has already opened.")
            if len(action.melds) < self.required_pairs_count:
                raise ValueError(
                    f"OPEN_PAIRS count {len(action.melds)} < required {self.required_pairs_count}."
                )

            all_tids = []
            for m in action.melds:
                if m.meld_type != MeldType.PAIR:
                    raise ValueError("OPEN_PAIRS only accepts PAIR melds.")
                if not validate_meld(m, self.joker_spec):
                    raise ValueError(f"Invalid pair in OPEN_PAIRS: {m}")
                all_tids.extend(pt.tile.tile_id for pt in m.placed_tiles)

            self._remove_tiles_from_hand(p, all_tids)
            for m in action.melds:
                self._add_table_meld(p, m)

            self.open_types[p] = OpenType.PAIRS
            if self.config.katlamali:
                self.highest_pairs_open = max(self.highest_pairs_open, len(action.melds))
            info["opened_pairs"] = len(action.melds)

            if not self.hands[p]:
                self._finalize_round(winner=p, win_by_okey=False)

            return StepResult(
                observation=self.get_observation(self.current_player),
                done=self.done,
                winner=self.winner,
                round_scores=self.round_scores,
                info=info,
            )

        if atype == ActionType.MELD:
            ot = self.open_types[p]
            if ot == OpenType.NONE:
                raise ValueError("Player must open before melding additional melds.")
            if not action.melds:
                raise ValueError("MELD requires at least one meld.")

            all_tids = []
            for m in action.melds:
                if ot == OpenType.SERIES and m.meld_type not in (MeldType.RUN, MeldType.SET):
                    raise ValueError("Series-opened players can only meld RUNs and SETs.")
                if ot == OpenType.PAIRS and m.meld_type != MeldType.PAIR:
                    raise ValueError("Pairs-opened players can only meld PAIRs.")
                if not validate_meld(m, self.joker_spec):
                    raise ValueError(f"Invalid meld: {m}")
                all_tids.extend(pt.tile.tile_id for pt in m.placed_tiles)

            self._remove_tiles_from_hand(p, all_tids)
            for m in action.melds:
                self._add_table_meld(p, m)

            if not self.hands[p]:
                self._finalize_round(winner=p, win_by_okey=False)

            return StepResult(
                observation=self.get_observation(self.current_player),
                done=self.done,
                winner=self.winner,
                round_scores=self.round_scores,
                info=info,
            )

        if atype == ActionType.LAY_OFF:
            if self.open_types[p] != OpenType.SERIES:
                raise ValueError(
                    "Only players opened via SERIES/SET can lay off onto table melds."
                )
            if action.tile_id is None or action.meld_id is None:
                raise ValueError("LAY_OFF requires tile_id and meld_id.")

            target_tm = next(
                (tm for tm in self.table_melds if tm.meld_id == action.meld_id), None
            )
            if target_tm is None:
                raise ValueError(f"Table meld {action.meld_id} not found.")

            tile = next((t for t in self.hands[p] if t.tile_id == action.tile_id), None)
            if tile is None:
                raise ValueError(f"Tile {action.tile_id} is not in player {p}'s hand.")

            updated_meld = apply_lay_off(
                target_tm.meld, tile, self.joker_spec, side=action.lay_off_side
            )
            self._remove_tiles_from_hand(p, [action.tile_id])
            target_tm.meld = updated_meld

            if not self.hands[p]:
                self._finalize_round(winner=p, win_by_okey=False)

            return StepResult(
                observation=self.get_observation(self.current_player),
                done=self.done,
                winner=self.winner,
                round_scores=self.round_scores,
                info=info,
            )

        if atype == ActionType.DISCARD:
            if self.must_use_tile_id is not None:
                raise ValueError(
                    f"Cannot discard: taken discard tile {self.must_use_tile_id} must be used this turn."
                )
            if action.tile_id is None:
                raise ValueError("DISCARD requires tile_id.")

            removed = self._remove_tiles_from_hand(p, [action.tile_id])
            discarded = removed[0]
            self.discard_piles[p].append(discarded)
            info["discarded_tile"] = discarded

            is_winning_discard = len(self.hands[p]) == 0
            is_joker_discard = self.joker_spec.is_joker(discarded)

            if is_winning_discard:
                self._finalize_round(winner=p, win_by_okey=is_joker_discard)
                return StepResult(
                    observation=self.get_observation(self.current_player),
                    done=True,
                    winner=self.winner,
                    round_scores=self.round_scores,
                    penalty_incurred=0,
                    info=info,
                )

            # Non-winning discard: check official +101 penalties
            if is_joker_discard:
                self.penalties[p] += 101
                penalty_incurred = 101
                info["penalty_reason"] = "joker_discard"
            elif self.can_tile_lay_off_anywhere(discarded):
                self.penalties[p] += 101
                penalty_incurred = 101
                info["penalty_reason"] = "islek_discard"

            # Check if center draw pile is exhausted
            if not self.draw_pile:
                self._finalize_round(winner=None, draw_pile_empty=True)
                return StepResult(
                    observation=self.get_observation(self.current_player),
                    done=True,
                    winner=None,
                    round_scores=self.round_scores,
                    penalty_incurred=penalty_incurred,
                    info=info,
                )

            self._advance_turn()
            return StepResult(
                observation=self.get_observation(self.current_player),
                done=False,
                winner=None,
                round_scores=None,
                penalty_incurred=penalty_incurred,
                info=info,
            )

        raise ValueError(f"Unsupported action type during PLAY phase: {atype}")

