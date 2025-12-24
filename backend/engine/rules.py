from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from backend.engine.board import EMPTY, BLOCK, P1, P2, Board


@dataclass(frozen=True)
class Pos:
    r: int
    c: int


@dataclass(frozen=True)
class Action:
    from_pos: Pos
    to_pos: Pos
    arrow_pos: Pos


# --- existing helpers (keep yours if already present) ---

def in_bounds(p: Pos) -> bool:
    return 0 <= p.r < 10 and 0 <= p.c < 10


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def queen_direction(src: Pos, dst: Pos) -> Optional[Tuple[int, int]]:
    dr = dst.r - src.r
    dc = dst.c - src.c
    if dr == 0 and dc == 0:
        return None

    sdr = _sign(dr)
    sdc = _sign(dc)

    if dr == 0 and dc != 0:
        return (0, sdc)
    if dc == 0 and dr != 0:
        return (sdr, 0)
    if abs(dr) == abs(dc):
        return (sdr, sdc)
    return None


def iter_ray(src: Pos, step: Tuple[int, int], limit: int = 9) -> Iterable[Pos]:
    r, c = src.r, src.c
    dr, dc = step
    for _ in range(limit):
        r += dr
        c += dc
        yield Pos(r, c)


def is_clear_queen_path(grid: List[List[int]], src: Pos, dst: Pos) -> bool:
    if not (in_bounds(src) and in_bounds(dst)):
        return False

    step = queen_direction(src, dst)
    if step is None:
        return False

    cur = src
    while True:
        cur = Pos(cur.r + step[0], cur.c + step[1])
        if not in_bounds(cur):
            return False
        if cur == dst:
            return True
        if grid[cur.r][cur.c] != EMPTY:
            return False


# --- legality + apply (keep yours if already present) ---

def is_legal_action(board: Board, player: int, action: Action) -> Tuple[bool, str]:
    g = board.to_list()
    fr, to, ar = action.from_pos, action.to_pos, action.arrow_pos

    if player not in (P1, P2):
        return False, "Unknown player."
    if not (in_bounds(fr) and in_bounds(to) and in_bounds(ar)):
        return False, "Out of bounds."
    if g[fr.r][fr.c] != player:
        return False, "FROM does not contain current player's amazon."
    if g[to.r][to.c] != EMPTY:
        return False, "TO is not empty."
    if fr == to:
        return False, "FROM and TO cannot be the same."
    if not is_clear_queen_path(g, fr, to):
        return False, "Illegal move: FROM->TO is not a clear queen move."

    g2 = [row[:] for row in g]
    g2[fr.r][fr.c] = EMPTY
    g2[to.r][to.c] = player

    if g2[ar.r][ar.c] != EMPTY:
        return False, "ARROW is not empty."
    if to == ar:
        return False, "ARROW cannot be the same as TO."
    if not is_clear_queen_path(g2, to, ar):
        return False, "Illegal arrow: TO->ARROW is not a clear queen move."

    return True, ""


def apply_action(board: Board, player: int, action: Action) -> Board:
    ok, reason = is_legal_action(board, player, action)
    if not ok:
        raise ValueError(reason)

    g = board.to_list()
    fr, to, ar = action.from_pos, action.to_pos, action.arrow_pos
    g[fr.r][fr.c] = EMPTY
    g[to.r][to.c] = player
    g[ar.r][ar.c] = BLOCK
    return Board(g)


# =========================================================
# NEW: move generation for MCTS
# =========================================================

QUEEN_STEPS: List[Tuple[int, int]] = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]


def iter_queen_reachable_empty(grid: List[List[int]], src: Pos) -> Iterable[Pos]:
    """
    Yield all EMPTY squares reachable from src by queen move (8 dirs, sliding).
    """
    for dr, dc in QUEEN_STEPS:
        r, c = src.r, src.c
        while True:
            r += dr
            c += dc
            p = Pos(r, c)
            if not in_bounds(p):
                break
            if grid[r][c] != EMPTY:
                break
            yield p


def find_amazons(grid: List[List[int]], player: int) -> List[Pos]:
    out: List[Pos] = []
    for r in range(10):
        for c in range(10):
            if grid[r][c] == player:
                out.append(Pos(r, c))
    return out


def generate_legal_actions(board: Board, player: int) -> List[Action]:
    """
    Generate all legal full actions (from->to, then to->arrow) for player.

    Complexity can be large; OK for initial MCTS, later optimize.
    """
    g = board.to_list()
    amazons = find_amazons(g, player)

    actions: List[Action] = []

    for fr in amazons:
        # All possible TO squares
        for to in iter_queen_reachable_empty(g, fr):
            # Apply move on a temp grid
            g2 = [row[:] for row in g]
            g2[fr.r][fr.c] = EMPTY
            g2[to.r][to.c] = player

            # All possible ARROW squares from TO
            for ar in iter_queen_reachable_empty(g2, to):
                actions.append(Action(from_pos=fr, to_pos=to, arrow_pos=ar))

    return actions


def is_terminal(board: Board, player: int) -> bool:
    """
    Terminal for current player if they have no legal actions.
    """
    return len(generate_legal_actions(board, player)) == 0


def winner_if_game_over(board: Board, next_player_to_move: int) -> Optional[int]:
    """
    If next_player_to_move has no legal actions, they lose immediately.
    Returns winner (P1 or P2) or None if game continues.
    """
    if is_terminal(board, next_player_to_move):
        return P2 if next_player_to_move == P1 else P1
    return None