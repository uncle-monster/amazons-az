from dataclasses import dataclass


@dataclass(frozen=True)
class Pos:
    r: int
    c: int


@dataclass(frozen=True)
class Action:
    from_pos: Pos
    to_pos: Pos
    arrow_pos: Pos