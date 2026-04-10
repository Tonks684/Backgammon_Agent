from enum import Enum
from typing import NamedTuple


class Player(Enum):
    WHITE = 0
    BLACK = 1

    def opponent(self) -> "Player":
        return Player.BLACK if self == Player.WHITE else Player.WHITE


class GameResult(Enum):
    IN_PROGRESS = 0
    WHITE_WIN = 1
    WHITE_GAMMON = 2
    WHITE_BACKGAMMON = 3
    BLACK_WIN = 4
    BLACK_GAMMON = 5
    BLACK_BACKGAMMON = 6

    def is_terminal(self) -> bool:
        return self != GameResult.IN_PROGRESS

    def winner(self) -> Player | None:
        if self in (GameResult.WHITE_WIN, GameResult.WHITE_GAMMON, GameResult.WHITE_BACKGAMMON):
            return Player.WHITE
        if self in (GameResult.BLACK_WIN, GameResult.BLACK_GAMMON, GameResult.BLACK_BACKGAMMON):
            return Player.BLACK
        return None


# Sentinel point values
BAR_POINT = 24
BEAROFF_POINT = 25


class Move(NamedTuple):
    from_point: int  # 0-23 = board points, 24 = bar, 25 = bearoff (unused as source)
    to_point: int    # 0-23 = board points, 24 = bar (unused as dest), 25 = bearoff


class DiceRoll(NamedTuple):
    d1: int
    d2: int

    @property
    def is_doubles(self) -> bool:
        return self.d1 == self.d2

    def die_values(self) -> list[int]:
        """Returns the list of die values available to use this turn."""
        if self.is_doubles:
            return [self.d1, self.d1, self.d1, self.d1]
        return [self.d1, self.d2]
