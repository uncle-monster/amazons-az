from backend.nn.load import load_model
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.engine.board import Board, P1, P2
from backend.engine.rules import Action, Pos, apply_action, is_legal_action, winner_if_game_over
from backend.engine.mcts_puct import mcts_puct_search

router = APIRouter()
NET = load_model("checkpoints/model.pt", device="cpu")

GAME = {
    "board": Board.initial(),
    "turn": 1,         # 1=human(P1), 2=ai(P2)
    "game_over": False,
    "winner": None,    # 1 or 2
}


class PosIn(BaseModel):
    r: int
    c: int


class ActionIn(BaseModel):
    from_pos: PosIn
    to_pos: PosIn
    arrow_pos: PosIn


@router.get("/state")
def get_state():
    b: Board = GAME["board"]
    return {
        "turn": GAME["turn"],
        "board": b.to_list(),
        "game_over": GAME["game_over"],
        "winner": GAME["winner"],
    }


@router.post("/reset")
def reset():
    GAME["board"] = Board.initial()
    GAME["turn"] = 1
    GAME["game_over"] = False
    GAME["winner"] = None
    return {"ok": True}


@router.post("/move")
def human_move(action: ActionIn):
    if GAME["game_over"]:
        raise HTTPException(status_code=400, detail="Game is over.")
    if GAME["turn"] != 1:
        raise HTTPException(status_code=400, detail="Not human turn.")

    board: Board = GAME["board"]
    a = Action(
        from_pos=Pos(action.from_pos.r, action.from_pos.c),
        to_pos=Pos(action.to_pos.r, action.to_pos.c),
        arrow_pos=Pos(action.arrow_pos.r, action.arrow_pos.c),
    )

    ok, reason = is_legal_action(board, P1, a)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    new_board = apply_action(board, P1, a)

    # Next player is AI (P2)
    w = winner_if_game_over(new_board, next_player_to_move=P2)
    if w is not None:
        GAME["board"] = new_board
        GAME["turn"] = 2
        GAME["game_over"] = True
        GAME["winner"] = w
        return {
            "turn": GAME["turn"],
            "board": GAME["board"].to_list(),
            "game_over": True,
            "winner": w,
        }

    GAME["board"] = new_board
    GAME["turn"] = 2
    return {
        "turn": GAME["turn"],
        "board": GAME["board"].to_list(),
        "game_over": False,
        "winner": None,
    }


@router.post("/ai_move")
def ai_move(simulations: int = 200):
    if GAME["game_over"]:
        raise HTTPException(status_code=400, detail="Game is over.")
    if GAME["turn"] != 2:
        raise HTTPException(status_code=400, detail="Not AI turn.")

    board: Board = GAME["board"]

    a = mcts_puct_search(
    board=board,
    player_to_move=P2,
    net=NET,
    device="cpu",
    simulations=simulations,
    c_puct=1.5,
    max_actions_expand=400,
    seed=0,
)

    new_board = apply_action(board, P2, a)

    # Next player is human (P1)
    w = winner_if_game_over(new_board, next_player_to_move=P1)
    if w is not None:
        GAME["board"] = new_board
        GAME["turn"] = 1
        GAME["game_over"] = True
        GAME["winner"] = w
        return {
            "turn": GAME["turn"],
            "board": GAME["board"].to_list(),
            "game_over": True,
            "winner": w,
        }

    GAME["board"] = new_board
    GAME["turn"] = 1
    return {
        "turn": GAME["turn"],
        "board": GAME["board"].to_list(),
        "game_over": False,
        "winner": None,
    }