from __future__ import annotations

import numpy as np

from backend.engine.board import EMPTY, BLOCK, P1, P2, Board
from backend.engine.rules import Action, Pos


def encode_state(board: Board, player: int) -> np.ndarray:
    """
    Returns float32 array of shape (4, 10, 10):
      0: current player's amazons
      1: opponent amazons
      2: blocks
      3: player-to-move plane (all 1 if P1 else 0) - simple turn signal
    """
    g = board.to_list()
    x = np.zeros((4, 10, 10), dtype=np.float32)

    me = player
    opp = P2 if player == P1 else P1

    for r in range(10):
        for c in range(10):
            v = g[r][c]
            if v == me:
                x[0, r, c] = 1.0
            elif v == opp:
                x[1, r, c] = 1.0
            elif v == BLOCK:
                x[2, r, c] = 1.0

    x[3, :, :] = 1.0 if player == P1 else 0.0
    return x


def encode_action(action: Action) -> np.ndarray:
    """
    Returns float32 array of shape (3, 10, 10):
      0: from one-hot
      1: to one-hot
      2: arrow one-hot
    """
    a = np.zeros((3, 10, 10), dtype=np.float32)
    a[0, action.from_pos.r, action.from_pos.c] = 1.0
    a[1, action.to_pos.r, action.to_pos.c] = 1.0
    a[2, action.arrow_pos.r, action.arrow_pos.c] = 1.0
    return a