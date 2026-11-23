"""Tile, Joker, and Meld representations for Official Professional 101 Okey."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Iterable, Optional


class Color(IntEnum):
    """Four tile colors in 101 Okey."""

    RED = 0
    YELLOW = 1
    BLUE = 2
    BLACK = 3

    @property
    def short(self) -> str:
        return ("R", "Y", "B", "K")[self.value]


ALL_COLORS: tuple[Color, ...] = (Color.RED, Color.YELLOW, Color.BLUE, Color.BLACK)
NUM_COLORS = 4
NUM_NUMBERS = 13
NUM_KINDS = NUM_COLORS * NUM_NUMBERS  # 52 distinct (color, number) pairs
DECK_SIZE = 106


def key_to_color_number(key: int) -> tuple[Color, int]:
    """Convert a compact tile kind key (0..51) to (Color, number 1..13)."""
    return Color(key // NUM_NUMBERS), (key % NUM_NUMBERS) + 1


def color_number_to_key(color: Color, number: int) -> int:
    """Convert (Color, number 1..13) to a compact key (0..51)."""
    return color.value * NUM_NUMBERS + (number - 1)


@dataclass(frozen=True, slots=True)
class Tile:
    """Immutable tile in the 106-tile 101 Okey deck.

    - Standard tiles: color in Color, number in 1..13, is_false_joker=False.
      There are two copies of each (color, number), distinguished by tile_id.
    - False Jokers (Sahte Okey): color=None, number=0, is_false_joker=True.
      There are 2 False Jokers (tile_id 104 and 105).
    """

    tile_id: int
    color: Optional[Color]
    number: int
    is_false_joker: bool = False

    def __post_init__(self) -> None:
        if self.is_false_joker:
            if self.color is not None or self.number != 0:
                raise ValueError("False Joker must have color=None and number=0.")
        else:
            if self.color is None or not (1 <= self.number <= 13):
                raise ValueError(
                    f"Standard tile must have valid Color and number 1..13, got {self.color}, {self.number}."
                )

    def __repr__(self) -> str:
        if self.is_false_joker:
            return f"FJ#{self.tile_id}"
        assert self.color is not None
        return f"{self.color.short}{self.number}#{self.tile_id}"


@dataclass(frozen=True, slots=True)
class JokerSpec:
    """Defines the indicator tile and the active Joker (Okey) for a round.

    The Joker is the same color as the indicator with number (indicator.number % 13) + 1.
    False Jokers strictly represent (joker_color, joker_number) and are NOT wildcards.
    """

    indicator: Tile
    joker_color: Color
    joker_number: int

    @classmethod
    def from_indicator(cls, indicator: Tile) -> JokerSpec:
        if indicator.is_false_joker or indicator.color is None:
            raise ValueError("Indicator tile cannot be a False Joker.")
        joker_number = (indicator.number % 13) + 1
        return cls(
            indicator=indicator,
            joker_color=indicator.color,
            joker_number=joker_number,
        )

    @property
    def joker_key(self) -> int:
        return color_number_to_key(self.joker_color, self.joker_number)

    def is_joker(self, tile: Tile) -> bool:
        """Return True iff `tile` is one of the 2 real Jokers (Okey)."""
        return (
            not tile.is_false_joker
            and tile.color == self.joker_color
            and tile.number == self.joker_number
        )

    def effective_color_number(self, tile: Tile) -> Optional[tuple[Color, int]]:
        """Return the effective (Color, number) of `tile`, or None if it is a real Joker."""
        if self.is_joker(tile):
            return None
        if tile.is_false_joker:
            return (self.joker_color, self.joker_number)
        assert tile.color is not None
        return (tile.color, tile.number)

    def effective_key(self, tile: Tile) -> int:
        """Return compact key 0..51 for non-jokers (including False Joker), or -1 for real Joker."""
        if self.is_joker(tile):
            return -1
        if tile.is_false_joker:
            return self.joker_key
        assert tile.color is not None
        return color_number_to_key(tile.color, tile.number)

    def effective_value(self, tile: Tile) -> Optional[int]:
        """Return face value (1..13) for non-jokers (including False Joker), or None for real Joker."""
        if self.is_joker(tile):
            return None
        if tile.is_false_joker:
            return self.joker_number
        return tile.number

    def hand_penalty_value(self, tile: Tile) -> int:
        """Return end-of-round hand penalty value (101 for a real Joker, face value otherwise)."""
        if self.is_joker(tile):
            return 101
        if tile.is_false_joker:
            return self.joker_number
        return tile.number


def create_deck() -> list[Tile]:
    """Create the official 106-tile 101 Okey deck."""
    tiles: list[Tile] = []
    tile_id = 0
    for _copy in range(2):
        for color in ALL_COLORS:
            for number in range(1, NUM_NUMBERS + 1):
                tiles.append(Tile(tile_id=tile_id, color=color, number=number))
                tile_id += 1
    # Two False Jokers (Sahte Okey)
    tiles.append(Tile(tile_id=tile_id, color=None, number=0, is_false_joker=True))
    tiles.append(Tile(tile_id=tile_id + 1, color=None, number=0, is_false_joker=True))
    return tiles


class MeldType(Enum):
    """Valid meld types in 101 Okey."""

    RUN = "run"    # Seri: 3+ consecutive same-color numbers (1..13, no 12-13-1 wrap)
    SET = "set"    # Per: 3 or 4 same-number tiles of distinct colors
    PAIR = "pair"  # Çift: 2 identical (color, number) tiles


@dataclass(frozen=True, slots=True)
class PlacedTile:
    """A concrete tile placed in a meld with its resolved (color, number) role."""

    tile: Tile
    assigned_color: Color
    assigned_number: int


@dataclass(slots=True)
class Meld:
    """A validated meld on the table or proposed by a player/solver."""

    meld_type: MeldType
    placed_tiles: list[PlacedTile] = field(default_factory=list)

    @property
    def tiles(self) -> list[Tile]:
        return [pt.tile for pt in self.placed_tiles]

    @property
    def points(self) -> int:
        """Sum of assigned face values (Jokers take their substituted value)."""
        return sum(pt.assigned_number for pt in self.placed_tiles)

    @property
    def size(self) -> int:
        return len(self.placed_tiles)

    def copy(self) -> Meld:
        return Meld(meld_type=self.meld_type, placed_tiles=list(self.placed_tiles))


def validate_meld(meld: Meld, joker_spec: JokerSpec) -> bool:
    """Strictly validate a Meld against Official Professional 101 Okey rules."""
    pts = meld.placed_tiles
    if not pts:
        return False

    # Ensure distinct physical tiles
    seen_ids: set[int] = set()
    for pt in pts:
        if pt.tile.tile_id in seen_ids:
            return False
        seen_ids.add(pt.tile.tile_id)
        if not (1 <= pt.assigned_number <= 13):
            return False
        eff = joker_spec.effective_color_number(pt.tile)
        if eff is not None and eff != (pt.assigned_color, pt.assigned_number):
            return False

    if meld.meld_type == MeldType.RUN:
        if not (3 <= len(pts) <= 13):
            return False
        run_color = pts[0].assigned_color
        start_num = pts[0].assigned_number
        if start_num + len(pts) - 1 > 13:
            return False
        for i, pt in enumerate(pts):
            if pt.assigned_color != run_color:
                return False
            if pt.assigned_number != start_num + i:
                return False
        return True

    if meld.meld_type == MeldType.SET:
        if len(pts) not in (3, 4):
            return False
        set_num = pts[0].assigned_number
        colors: set[Color] = set()
        for pt in pts:
            if pt.assigned_number != set_num:
                return False
            if pt.assigned_color in colors:
                return False
            colors.add(pt.assigned_color)
        return True

    if meld.meld_type == MeldType.PAIR:
        if len(pts) != 2:
            return False
        return (
            pts[0].assigned_color == pts[1].assigned_color
            and pts[0].assigned_number == pts[1].assigned_number
        )

    return False


def build_run(
    tiles: Iterable[Tile],
    joker_spec: JokerSpec,
    start_number: Optional[int] = None,
) -> Optional[Meld]:
    """Attempt to construct a valid RUN from `tiles` (maximizing points if ambiguous).

    Strictly rejects wrap-around sequences such as 12-13-1.
    Returns None if no valid run can be formed.
    """
    tile_list = list(tiles)
    length = len(tile_list)
    if not (3 <= length <= 13):
        return None
    if len({t.tile_id for t in tile_list}) != length:
        return None

    jokers: list[Tile] = []
    non_jokers: list[tuple[Tile, Color, int]] = []
    for t in tile_list:
        eff = joker_spec.effective_color_number(t)
        if eff is None:
            jokers.append(t)
        else:
            non_jokers.append((t, eff[0], eff[1]))

    if not non_jokers:
        # Cannot form a 3+ run with only jokers (max 2 jokers in deck)
        return None

    run_color = non_jokers[0][1]
    if any(c != run_color for _, c, _ in non_jokers):
        return None

    by_num: dict[int, Tile] = {}
    for t, _, n in non_jokers:
        if n in by_num:
            return None  # Duplicate number in a run
        by_num[n] = t

    min_n = min(by_num)
    max_n = max(by_num)
    if max_n - min_n + 1 > length:
        return None  # Gap too wide (also naturally rejects 12-13-1 since 13-1+1=13 > 3)

    # Valid start numbers s must satisfy: 1 <= s <= min_n and s + length - 1 >= max_n and s + length - 1 <= 13
    s_min = max(1, max_n - length + 1)
    s_max = min(min_n, 13 - length + 1)
    if s_min > s_max:
        return None

    if start_number is not None:
        if not (s_min <= start_number <= s_max):
            return None
        chosen_start = start_number
    else:
        # Highest start maximizes points
        chosen_start = s_max

    placed: list[PlacedTile] = []
    joker_idx = 0
    for n in range(chosen_start, chosen_start + length):
        if n in by_num:
            placed.append(PlacedTile(by_num[n], run_color, n))
        else:
            placed.append(PlacedTile(jokers[joker_idx], run_color, n))
            joker_idx += 1

    meld = Meld(meld_type=MeldType.RUN, placed_tiles=placed)
    return meld if validate_meld(meld, joker_spec) else None


def build_set(tiles: Iterable[Tile], joker_spec: JokerSpec) -> Optional[Meld]:
    """Attempt to construct a valid SET (3 or 4 tiles) from `tiles`."""
    tile_list = list(tiles)
    length = len(tile_list)
    if length not in (3, 4):
        return None
    if len({t.tile_id for t in tile_list}) != length:
        return None

    jokers: list[Tile] = []
    non_jokers: list[tuple[Tile, Color, int]] = []
    for t in tile_list:
        eff = joker_spec.effective_color_number(t)
        if eff is None:
            jokers.append(t)
        else:
            non_jokers.append((t, eff[0], eff[1]))

    if not non_jokers:
        return None

    set_num = non_jokers[0][2]
    if any(n != set_num for _, _, n in non_jokers):
        return None

    used_colors: set[Color] = set()
    placed: list[PlacedTile] = []
    for t, c, n in non_jokers:
        if c in used_colors:
            return None
        used_colors.add(c)
        placed.append(PlacedTile(t, c, n))

    remaining_colors = [c for c in ALL_COLORS if c not in used_colors]
    if len(jokers) > len(remaining_colors):
        return None
    for j_tile, c in zip(jokers, remaining_colors):
        placed.append(PlacedTile(j_tile, c, set_num))

    placed.sort(key=lambda pt: pt.assigned_color.value)
    meld = Meld(meld_type=MeldType.SET, placed_tiles=placed)
    return meld if validate_meld(meld, joker_spec) else None


def build_pair(tiles: Iterable[Tile], joker_spec: JokerSpec) -> Optional[Meld]:
    """Attempt to construct a valid PAIR from 2 tiles."""
    tile_list = list(tiles)
    if len(tile_list) != 2 or tile_list[0].tile_id == tile_list[1].tile_id:
        return None

    e0 = joker_spec.effective_color_number(tile_list[0])
    e1 = joker_spec.effective_color_number(tile_list[1])
    if e0 is None and e1 is None:
        # Two real jokers: assign highest value (RED 13) as canonical representation
        c, n = Color.RED, 13
    elif e0 is None:
        assert e1 is not None
        c, n = e1
    elif e1 is None:
        c, n = e0
    else:
        if e0 != e1:
            return None
        c, n = e0

    meld = Meld(
        meld_type=MeldType.PAIR,
        placed_tiles=[
            PlacedTile(tile_list[0], c, n),
            PlacedTile(tile_list[1], c, n),
        ],
    )
    return meld if validate_meld(meld, joker_spec) else None


def can_lay_off(meld: Meld, tile: Tile, joker_spec: JokerSpec) -> list[str]:
    """Return legal lay-off positions for `tile` onto `meld`.

    For RUN: subset of ["left", "right"].
    For SET: ["add"] or [].
    For PAIR: [] (laying off onto pairs is illegal).
    """
    if meld.meld_type == MeldType.PAIR:
        return []

    if meld.meld_type == MeldType.RUN:
        pts = meld.placed_tiles
        if len(pts) >= 13:
            return []
        run_color = pts[0].assigned_color
        start = pts[0].assigned_number
        end = pts[-1].assigned_number

        if joker_spec.is_joker(tile):
            sides: list[str] = []
            if start > 1:
                sides.append("left")
            if end < 13:
                sides.append("right")
            return sides

        eff = joker_spec.effective_color_number(tile)
        assert eff is not None
        c, n = eff
        if c != run_color:
            return []

        sides = []
        if start > 1 and (
            n == start - 1
            or (n == start and joker_spec.is_joker(pts[0].tile))
        ):
            sides.append("left")
        if end < 13 and (
            n == end + 1
            or (n == end and joker_spec.is_joker(pts[-1].tile))
        ):
            sides.append("right")
        return sides

    if meld.meld_type == MeldType.SET:
        pts = meld.placed_tiles
        if len(pts) >= 4:
            return []
        set_num = pts[0].assigned_number
        if joker_spec.is_joker(tile):
            return ["add"]
        eff = joker_spec.effective_color_number(tile)
        assert eff is not None
        c, n = eff
        if n != set_num:
            return []
        fixed_colors = {
            pt.assigned_color for pt in pts if not joker_spec.is_joker(pt.tile)
        }
        return ["add"] if c not in fixed_colors else []

    return []


def apply_lay_off(
    meld: Meld,
    tile: Tile,
    joker_spec: JokerSpec,
    side: str = "auto",
) -> Meld:
    """Return a new Meld with `tile` laid off onto `meld`. Raises ValueError if illegal."""
    legal_sides = can_lay_off(meld, tile, joker_spec)
    if not legal_sides:
        raise ValueError(f"Tile {tile} cannot be laid off onto meld {meld}.")

    if side == "auto":
        # Prefer right for runs (higher value) or add for sets
        chosen = "right" if "right" in legal_sides else legal_sides[0]
    else:
        if side not in legal_sides:
            raise ValueError(
                f"Lay-off side '{side}' not legal (legal: {legal_sides})."
            )
        chosen = side

    pts = list(meld.placed_tiles)

    if meld.meld_type == MeldType.RUN:
        run_color = pts[0].assigned_color
        start = pts[0].assigned_number
        end = pts[-1].assigned_number

        if chosen == "left":
            if not joker_spec.is_joker(tile):
                eff = joker_spec.effective_color_number(tile)
                assert eff is not None
                if eff[1] == start and joker_spec.is_joker(pts[0].tile):
                    # Shift boundary Joker outward to start - 1
                    joker_tile = pts[0].tile
                    pts[0] = PlacedTile(tile, run_color, start)
                    pts.insert(0, PlacedTile(joker_tile, run_color, start - 1))
                else:
                    pts.insert(0, PlacedTile(tile, run_color, start - 1))
            else:
                pts.insert(0, PlacedTile(tile, run_color, start - 1))
        else:  # "right"
            if not joker_spec.is_joker(tile):
                eff = joker_spec.effective_color_number(tile)
                assert eff is not None
                if eff[1] == end and joker_spec.is_joker(pts[-1].tile):
                    joker_tile = pts[-1].tile
                    pts[-1] = PlacedTile(tile, run_color, end)
                    pts.append(PlacedTile(joker_tile, run_color, end + 1))
                else:
                    pts.append(PlacedTile(tile, run_color, end + 1))
            else:
                pts.append(PlacedTile(tile, run_color, end + 1))

        new_meld = Meld(meld_type=MeldType.RUN, placed_tiles=pts)
        assert validate_meld(new_meld, joker_spec)
        return new_meld

    if meld.meld_type == MeldType.SET:
        set_num = pts[0].assigned_number
        # Rebuild set cleanly so any existing Joker re-binds if needed
        new_tiles = [pt.tile for pt in pts] + [tile]
        new_meld = build_set(new_tiles, joker_spec)
        if new_meld is None:
            raise ValueError(f"Failed to lay off {tile} onto set {meld}.")
        return new_meld

    raise ValueError(f"Unsupported meld type for lay-off: {meld.meld_type}")

