from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from backgammon.game.types import GameResult


# Output vector layout: [p_white_win, p_white_gammon, p_black_win, p_black_gammon]
_EQUITY_TARGETS: dict[GameResult, list[float]] = {
    GameResult.WHITE_WIN: [1.0, 0.0, 0.0, 0.0],
    GameResult.WHITE_GAMMON: [1.0, 1.0, 0.0, 0.0],
    GameResult.WHITE_BACKGAMMON: [1.0, 1.0, 0.0, 0.0],  # treated same as gammon
    GameResult.BLACK_WIN: [0.0, 0.0, 1.0, 0.0],
    GameResult.BLACK_GAMMON: [0.0, 0.0, 1.0, 1.0],
    GameResult.BLACK_BACKGAMMON: [0.0, 0.0, 1.0, 1.0],  # treated same as gammon
}


def compute_equity_target(result: GameResult) -> np.ndarray:
    """Convert a terminal GameResult to the 4-element training target vector.

    Layout: [p_white_win, p_white_gammon, p_black_win, p_black_gammon]
    Backgammon outcomes are folded into gammon (same equity value).
    """
    return np.array(_EQUITY_TARGETS[result], dtype=np.float32)


class WinRateTracker:
    """Rolling-window tracker for win/gammon/backgammon statistics.

    Parameters
    ----------
    window:
        Number of recent games to keep in the rolling window (default 1000).
    """

    def __init__(self, window: int = 1000) -> None:
        self._window = window
        self._results: Deque[GameResult] = deque(maxlen=window)

    def record(self, result: GameResult) -> None:
        self._results.append(result)

    @property
    def n_games(self) -> int:
        return len(self._results)

    @property
    def white_win_rate(self) -> float:
        if not self._results:
            return 0.0
        white_wins = sum(
            1
            for r in self._results
            if r in (
                GameResult.WHITE_WIN,
                GameResult.WHITE_GAMMON,
                GameResult.WHITE_BACKGAMMON,
            )
        )
        return white_wins / len(self._results)

    @property
    def gammon_rate(self) -> float:
        """Fraction of games that ended in a gammon or backgammon (either side)."""
        if not self._results:
            return 0.0
        gammons = sum(
            1
            for r in self._results
            if r in (
                GameResult.WHITE_GAMMON,
                GameResult.WHITE_BACKGAMMON,
                GameResult.BLACK_GAMMON,
                GameResult.BLACK_BACKGAMMON,
            )
        )
        return gammons / len(self._results)

    @property
    def white_gammon_rate(self) -> float:
        if not self._results:
            return 0.0
        return sum(
            1
            for r in self._results
            if r in (GameResult.WHITE_GAMMON, GameResult.WHITE_BACKGAMMON)
        ) / len(self._results)

    @property
    def black_gammon_rate(self) -> float:
        if not self._results:
            return 0.0
        return sum(
            1
            for r in self._results
            if r in (GameResult.BLACK_GAMMON, GameResult.BLACK_BACKGAMMON)
        ) / len(self._results)

    def summary(self) -> dict:
        return {
            "n_games": self.n_games,
            "white_win_rate": self.white_win_rate,
            "gammon_rate": self.gammon_rate,
            "white_gammon_rate": self.white_gammon_rate,
            "black_gammon_rate": self.black_gammon_rate,
        }
