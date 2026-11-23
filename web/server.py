"""FastAPI backend for playing or spectating Official Professional 101 Okey."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.base import Agent
from agents.heuristic_agent import HeuristicAgent
from agents.planner_agent import PlannerAgent
from engine import (
    Action,
    ActionType,
    GameConfig,
    NUM_PLAYERS,
    Okey101Engine,
    OpenType,
    TurnPhase,
)
from solver import solve_best_opening, solve_best_pairs, solve_min_deadwood
from tiles import JokerSpec, Meld, Tile, can_lay_off

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Official 101 Okey Web UI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _serialize_tile(tile: Tile, spec: JokerSpec) -> dict[str, Any]:
    is_joker = spec.is_joker(tile)
    eff = spec.effective_color_number(tile)
    if tile.is_false_joker:
        color_name = spec.joker_color.name.lower()
        display_num = spec.joker_number
        label = f"FJ({spec.joker_color.short}{spec.joker_number})"
    else:
        assert tile.color is not None
        color_name = tile.color.name.lower()
        display_num = tile.number
        label = f"{tile.color.short}{tile.number}"
    return {
        "tile_id": tile.tile_id,
        "color": color_name,
        "number": display_num,
        "is_joker": is_joker,
        "is_false_joker": tile.is_false_joker,
        "effective_color": eff[0].name.lower() if eff else None,
        "effective_number": eff[1] if eff else None,
        "label": label,
    }


def _serialize_meld(meld_id: int, owner_id: int, meld: Meld, spec: JokerSpec) -> dict[str, Any]:
    return {
        "meld_id": meld_id,
        "owner_id": owner_id,
        "meld_type": meld.meld_type.value,
        "points": meld.points,
        "placed_tiles": [
            {
                **_serialize_tile(pt.tile, spec),
                "assigned_color": pt.assigned_color.name.lower(),
                "assigned_number": pt.assigned_number,
            }
            for pt in meld.placed_tiles
        ],
    }


def _tile_short(tile: Tile, spec: JokerSpec) -> str:
    if spec.is_joker(tile):
        assert tile.color is not None
        return f"★{tile.color.short}{tile.number}"
    if tile.is_false_joker:
        return f"FJ({spec.joker_color.short}{spec.joker_number})"
    assert tile.color is not None
    return f"{tile.color.short}{tile.number}"


class GameSession:
    """Manages an interactive or spectator Official 101 Okey match."""

    def __init__(self) -> None:
        self.mode: Literal["play", "spectator"] = "play"
        self.bot_type: Literal["planner", "heuristic"] = "planner"
        self.katlamali: bool = False
        self.feed_penalty_multiplier: int = 0
        self.reveal_hands: bool = False
        self.round_number: int = 1
        self.cumulative_scores: list[int] = [0] * NUM_PLAYERS
        self.engine: Okey101Engine = Okey101Engine()
        self.agents: list[Optional[Agent]] = [None] * NUM_PLAYERS
        self.move_log: list[dict[str, Any]] = []
        self._round_scored: bool = False
        self._seed_counter: int = 100
        self.new_game(mode="play", bot_type="planner", katlamali=False, reset_match=True)

    def _make_agent(self, seat: int) -> Agent:
        seed = self._seed_counter * 31 + seat
        if self.bot_type == "planner":
            return PlannerAgent(num_scenarios=36, top_k=10, name=f"Planner P{seat}", seed=seed)
        return HeuristicAgent(name=f"Heuristic P{seat}", seed=seed)

    def _rebuild_agents(self) -> None:
        self.agents = []
        for s in range(NUM_PLAYERS):
            if self.mode == "play" and s == 0:
                self.agents.append(None)
            else:
                self.agents.append(self._make_agent(s))

    def _seat_name(self, seat: int) -> str:
        if self.mode == "play" and seat == 0:
            return "You (P0)"
        tag = "Planner" if self.bot_type == "planner" else "Heuristic"
        return f"{tag} P{seat}"

    def _log(self, text: str, kind: str = "info") -> None:
        self.move_log.append({"turn": self.engine.turn_count, "text": text, "kind": kind})
        if len(self.move_log) > 120:
            self.move_log = self.move_log[-120:]

    def new_game(
        self,
        mode: Literal["play", "spectator"] = "play",
        bot_type: Literal["planner", "heuristic"] = "planner",
        katlamali: bool = False,
        feed_penalty_multiplier: int = 0,
        reset_match: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.mode = mode
        self.bot_type = bot_type
        self.katlamali = katlamali
        self.feed_penalty_multiplier = feed_penalty_multiplier
        if reset_match:
            self.round_number = 1
            self.cumulative_scores = [0] * NUM_PLAYERS
        else:
            self.round_number += 1

        self._seed_counter += 1
        actual_seed = seed if seed is not None else self._seed_counter * 97 + self.round_number
        starting_player = (self.round_number - 1) % NUM_PLAYERS if not reset_match else 0

        self.engine = Okey101Engine(
            config=GameConfig(
                katlamali=katlamali,
                feed_penalty_multiplier=feed_penalty_multiplier,
            )
        )
        self.engine.reset(seed=actual_seed, starting_player=starting_player)
        self._rebuild_agents()
        self.move_log = []
        self._round_scored = False

        spec = self.engine.joker_spec
        self._log(
            f"Round {self.round_number} started. Indicator: {_tile_short(spec.indicator, spec)} → "
            f"Joker: ★{spec.joker_color.short}{spec.joker_number}. "
            f"{self._seat_name(starting_player)} starts with 22 tiles.",
            kind="system",
        )

    def _record_action_log(self, seat: int, action: Action, res_info: dict[str, Any], penalty: int) -> None:
        spec = self.engine.joker_spec
        name = self._seat_name(seat)
        at = action.action_type
        if at == ActionType.DRAW_DECK:
            self._log(f"{name} drew from center deck.")
        elif at == ActionType.TAKE_DISCARD:
            t = res_info.get("taken_tile")
            lbl = _tile_short(t, spec) if isinstance(t, Tile) else "tile"
            fp_amt = res_info.get("feed_penalty_amount")
            fp_plr = res_info.get("feed_penalty_player")
            if fp_amt and isinstance(fp_plr, int):
                self._log(
                    f"{name} took discard {lbl} ({self._seat_name(fp_plr)} +{fp_amt} feed penalty!).",
                    kind="penalty",
                )
            else:
                self._log(f"{name} took discard {lbl}.", kind="tactical")
        elif at == ActionType.OPEN_SERIES:
            pts = res_info.get("opened_points", sum(m.points for m in action.melds))
            self._log(f"{name} opened Series ({pts} pts)!", kind="open")
        elif at == ActionType.OPEN_PAIRS:
            cnt = res_info.get("opened_pairs", len(action.melds))
            self._log(f"{name} opened Pairs ({cnt} pairs)!", kind="open")
        elif at == ActionType.MELD:
            self._log(f"{name} melded {len(action.melds)} new meld(s).", kind="tactical")
        elif at == ActionType.LAY_OFF:
            self._log(f"{name} laid off (işlek) onto meld #{action.meld_id}.", kind="tactical")
        elif at == ActionType.DISCARD:
            t = res_info.get("discarded_tile")
            lbl = _tile_short(t, spec) if isinstance(t, Tile) else "tile"
            if penalty > 0:
                reason = "Joker discard" if res_info.get("penalty_reason") == "joker_discard" else "işlek discard"
                self._log(f"{name} discarded {lbl} (+{penalty} {reason} penalty!)", kind="penalty")
            else:
                self._log(f"{name} discarded {lbl}.")

    def _finalize_if_done(self) -> None:
        if self.engine.done and not self._round_scored and self.engine.round_scores is not None:
            self._round_scored = True
            for i in range(NUM_PLAYERS):
                self.cumulative_scores[i] += self.engine.round_scores[i]
            if self.engine.winner is not None:
                wf = self.engine.win_flags
                tags = []
                if wf.win_by_elden:
                    tags.append("Elden")
                if wf.win_by_okey:
                    tags.append("Okey")
                if wf.win_by_pairs:
                    tags.append("Pairs")
                tag_str = f" ({' + '.join(tags)}, x{wf.win_multiplier})" if tags else ""
                self._log(f"🏆 {self._seat_name(self.engine.winner)} won the round{tag_str}!", kind="win")
            else:
                self._log("Draw pile empty — round ended with no winner.", kind="system")

    def execute_action(self, action: Action) -> None:
        seat = self.engine.current_player
        res = self.engine.step(action)
        self._record_action_log(seat, action, res.info, res.penalty_incurred)
        self._finalize_if_done()

    def step_bot(self) -> bool:
        if self.engine.done:
            return False
        p = self.engine.current_player
        agent = self.agents[p]
        if agent is None:
            return False
        obs = self.engine.get_observation(p)
        legal = self.engine.get_legal_actions()
        action = agent.select_action(obs, legal)
        self.execute_action(action)
        return True

    def _build_solver_helper(self) -> dict[str, Any]:
        spec = self.engine.joker_spec
        hand = self.engine.hands[0]
        ot = self.engine.open_types[0]
        op_sol = solve_best_opening(
            hand,
            spec,
            min_points=self.engine.required_series_points,
            required_tile_id=self.engine.must_use_tile_id if self.engine.current_player == 0 else None,
        )
        pr_sol = solve_best_pairs(
            hand,
            spec,
            min_pairs=self.engine.required_pairs_count,
            required_tile_id=self.engine.must_use_tile_id if self.engine.current_player == 0 else None,
        )
        dw_sol = solve_min_deadwood(
            hand,
            spec,
            required_tile_id=self.engine.must_use_tile_id if self.engine.current_player == 0 else None,
        )

        # Build visual rack groups for "Auto-Group (Solver)"
        groups: list[list[int]] = []
        if ot == OpenType.PAIRS:
            pr_all = solve_best_pairs(hand, spec, min_pairs=1)
            for m in pr_all.pairs:
                groups.append([pt.tile.tile_id for pt in m.placed_tiles])
            rem = [t.tile_id for t in pr_all.deadwood_tiles]
        elif ot == OpenType.SERIES:
            for m in dw_sol.melds:
                groups.append([pt.tile.tile_id for pt in m.placed_tiles])
            rem = [t.tile_id for t in dw_sol.deadwood_tiles]
        else:
            for m in op_sol.melds:
                groups.append([pt.tile.tile_id for pt in m.placed_tiles])
            rem = [t.tile_id for t in op_sol.deadwood_tiles]

        return {
            "best_open_points": op_sol.total_points,
            "required_series_points": self.engine.required_series_points,
            "can_open_series": op_sol.meets_threshold,
            "best_pairs_count": pr_sol.pair_count,
            "required_pairs_count": self.engine.required_pairs_count,
            "can_open_pairs": pr_sol.meets_threshold,
            "min_deadwood": dw_sol.deadwood_penalty,
            "groups": groups,
            "deadwood_ids": rem,
        }

    def _build_human_legal(self) -> dict[str, Any]:
        if self.mode != "play" or self.engine.current_player != 0 or self.engine.done:
            return {
                "is_human_turn": False,
                "can_draw_deck": False,
                "can_take_discard": False,
                "can_open_series": False,
                "open_series_points": 0,
                "can_open_pairs": False,
                "open_pairs_count": 0,
                "can_meld": False,
                "layoffs": [],
                "discardable_tile_ids": [],
                "penalty_discards": {},
            }

        legal = self.engine.get_legal_actions()
        spec = self.engine.joker_spec
        hand = self.engine.hands[0]
        hand_map = {t.tile_id: t for t in hand}

        can_draw = any(a.action_type == ActionType.DRAW_DECK for a in legal)
        can_take = any(a.action_type == ActionType.TAKE_DISCARD for a in legal)
        open_s = next((a for a in legal if a.action_type == ActionType.OPEN_SERIES), None)
        open_p = next((a for a in legal if a.action_type == ActionType.OPEN_PAIRS), None)
        meld_a = next((a for a in legal if a.action_type == ActionType.MELD), None)

        layoffs = []
        for a in legal:
            if a.action_type == ActionType.LAY_OFF and a.tile_id is not None and a.meld_id is not None:
                layoffs.append(
                    {
                        "tile_id": a.tile_id,
                        "meld_id": a.meld_id,
                        "side": a.lay_off_side,
                        "tile_label": _tile_short(hand_map[a.tile_id], spec),
                    }
                )

        discardable: list[int] = []
        penalty_discards: dict[str, str] = {}
        for a in legal:
            if a.action_type == ActionType.DISCARD and a.tile_id is not None:
                discardable.append(a.tile_id)
                if len(hand) > 1:
                    t = hand_map[a.tile_id]
                    if spec.is_joker(t):
                        penalty_discards[str(a.tile_id)] = "joker"
                    elif self.engine.can_tile_lay_off_anywhere(t):
                        penalty_discards[str(a.tile_id)] = "islek"

        return {
            "is_human_turn": True,
            "can_draw_deck": can_draw,
            "can_take_discard": can_take,
            "can_open_series": open_s is not None,
            "open_series_points": sum(m.points for m in open_s.melds) if open_s else 0,
            "can_open_pairs": open_p is not None,
            "open_pairs_count": len(open_p.melds) if open_p else 0,
            "can_meld": meld_a is not None,
            "layoffs": layoffs,
            "discardable_tile_ids": discardable,
            "penalty_discards": penalty_discards,
        }

    def get_state(self) -> dict[str, Any]:
        spec = self.engine.joker_spec
        show_all = self.reveal_hands or self.engine.done or self.mode == "spectator"

        seats = []
        for s in range(NUM_PLAYERS):
            visible = (s == 0) or show_all
            seats.append(
                {
                    "seat_id": s,
                    "name": self._seat_name(s),
                    "is_human": self.mode == "play" and s == 0,
                    "hand_size": len(self.engine.hands[s]),
                    "open_type": self.engine.open_types[s].value,
                    "penalty": self.engine.penalties[s],
                    "cumulative_score": self.cumulative_scores[s],
                    "hand": [_serialize_tile(t, spec) for t in self.engine.hands[s]] if visible else None,
                    "discard_pile": [_serialize_tile(t, spec) for t in self.engine.discard_piles[s]],
                }
            )

        round_breakdown = None
        if self.engine.done and self.engine.round_scores is not None:
            round_breakdown = []
            wf = self.engine.win_flags
            for s in range(NUM_PLAYERS):
                base_hand = self.engine._base_hand_penalty(s)
                round_breakdown.append(
                    {
                        "seat_id": s,
                        "name": self._seat_name(s),
                        "open_type": self.engine.open_types[s].value,
                        "is_winner": self.engine.winner == s,
                        "hand_penalty": 0 if self.engine.winner == s else base_hand * wf.win_multiplier,
                        "winner_bonus": -101 * wf.win_multiplier if self.engine.winner == s else 0,
                        "in_round_penalty": self.engine.penalties[s],
                        "round_score": self.engine.round_scores[s],
                        "cumulative_score": self.cumulative_scores[s],
                    }
                )

        return {
            "mode": self.mode,
            "bot_type": self.bot_type,
            "katlamali": self.katlamali,
            "feed_penalty_multiplier": self.feed_penalty_multiplier,
            "reveal_hands": self.reveal_hands,
            "round_number": self.round_number,
            "current_player": self.engine.current_player,
            "phase": self.engine.phase.value,
            "done": self.engine.done,
            "winner": self.engine.winner,
            "win_flags": {
                "win_by_okey": self.engine.win_flags.win_by_okey,
                "win_by_elden": self.engine.win_flags.win_by_elden,
                "win_by_pairs": self.engine.win_flags.win_by_pairs,
                "win_multiplier": self.engine.win_flags.win_multiplier,
                "draw_pile_empty": self.engine.win_flags.draw_pile_empty,
            },
            "draw_pile_size": len(self.engine.draw_pile),
            "required_series_points": self.engine.required_series_points,
            "required_pairs_count": self.engine.required_pairs_count,
            "must_use_tile_id": self.engine.must_use_tile_id if self.engine.current_player == 0 else None,
            "indicator": _serialize_tile(spec.indicator, spec),
            "joker": {
                "color": spec.joker_color.name.lower(),
                "number": spec.joker_number,
                "label": f"{spec.joker_color.short}{spec.joker_number}",
            },
            "seats": seats,
            "table_melds": [
                _serialize_meld(tm.meld_id, tm.owner_id, tm.meld, spec)
                for tm in self.engine.table_melds
            ],
            "solver_helper": self._build_solver_helper(),
            "human_legal": self._build_human_legal(),
            "move_log": self.move_log[-60:],
            "round_breakdown": round_breakdown,
        }


session = GameSession()


# ---------------------------------------------------------------------------
# Request Schemas & Endpoints
# ---------------------------------------------------------------------------


class NewGameRequest(BaseModel):
    mode: Literal["play", "spectator"] = "play"
    bot_type: Literal["planner", "heuristic"] = "planner"
    katlamali: bool = False
    feed_penalty_multiplier: int = 0
    reset_match: bool = True
    seed: Optional[int] = None


class ConfigRequest(BaseModel):
    reveal_hands: Optional[bool] = None
    bot_type: Optional[Literal["planner", "heuristic"]] = None


class HumanActionRequest(BaseModel):
    action_type: str
    tile_id: Optional[int] = None
    meld_id: Optional[int] = None
    lay_off_side: str = "auto"


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    return session.get_state()


@app.post("/api/new_game")
def api_new_game(req: NewGameRequest) -> dict[str, Any]:
    session.new_game(
        mode=req.mode,
        bot_type=req.bot_type,
        katlamali=req.katlamali,
        feed_penalty_multiplier=req.feed_penalty_multiplier,
        reset_match=req.reset_match,
        seed=req.seed,
    )
    return session.get_state()


@app.post("/api/next_round")
def api_next_round() -> dict[str, Any]:
    session.new_game(
        mode=session.mode,
        bot_type=session.bot_type,
        katlamali=session.katlamali,
        feed_penalty_multiplier=session.feed_penalty_multiplier,
        reset_match=False,
    )
    return session.get_state()


@app.post("/api/config")
def api_config(req: ConfigRequest) -> dict[str, Any]:
    if req.reveal_hands is not None:
        session.reveal_hands = req.reveal_hands
    if req.bot_type is not None and req.bot_type != session.bot_type:
        session.bot_type = req.bot_type
        session._rebuild_agents()
    return session.get_state()


@app.post("/api/bot_step")
def api_bot_step() -> dict[str, Any]:
    session.step_bot()
    return session.get_state()


@app.post("/api/action")
def api_human_action(req: HumanActionRequest) -> dict[str, Any]:
    if session.mode != "play" or session.engine.current_player != 0 or session.engine.done:
        raise HTTPException(status_code=400, detail="Not human player's turn.")

    legal = session.engine.get_legal_actions()
    try:
        atype = ActionType(req.action_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown action_type: {req.action_type}") from exc

    chosen: Optional[Action] = None
    if atype in (ActionType.DRAW_DECK, ActionType.TAKE_DISCARD, ActionType.OPEN_SERIES, ActionType.OPEN_PAIRS, ActionType.MELD):
        chosen = next((a for a in legal if a.action_type == atype), None)
    elif atype == ActionType.DISCARD:
        chosen = next(
            (a for a in legal if a.action_type == ActionType.DISCARD and a.tile_id == req.tile_id),
            None,
        )
    elif atype == ActionType.LAY_OFF:
        chosen = next(
            (
                a
                for a in legal
                if a.action_type == ActionType.LAY_OFF
                and a.tile_id == req.tile_id
                and (req.meld_id is None or a.meld_id == req.meld_id)
                and (req.lay_off_side == "auto" or a.lay_off_side == req.lay_off_side)
            ),
            None,
        )

    if chosen is None:
        raise HTTPException(status_code=400, detail=f"Illegal action: {req.model_dump()}")

    try:
        session.execute_action(chosen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return session.get_state()

