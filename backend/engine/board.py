from __future__ import annotations

from dataclasses import dataclass
from typing import List

EMPTY = 0
P1 = 1
P2 = 2
BLOCK = 3


@dataclass(frozen=True)
class Board:
    grid: List[List[int]]  # 10x10

    @staticmethod
    def initial() -> "Board":
        g = [[EMPTY for _ in range(10)] for _ in range(10)]
        # 0-based common opening
        for r, c in [(0, 3), (0, 6), (3, 0), (3, 9)]:
            g[r][c] = P1
        for r, c in [(6, 0), (6, 9), (9, 3), (9, 6)]:
            g[r][c] = P2
        return Board(g)

    def to_list(self) -> List[List[int]]:
        return [row[:] for row in self.grid]