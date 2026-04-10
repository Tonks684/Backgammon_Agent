"""Random agent — uniformly samples a legal move sequence each turn."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from backgammon.game.types import Move, Player

if TYPE_CHECKING:
    from backgammon.game.board import Board


class RandomAgent:
    """Selects a legal move sequence uniformly at random.

    Used as a baseline to verify that a trained agent has learned something
    meaningful (target: >99% win rate vs RandomAgent).
    """

    def select_move(
        self,
        board: Board,
        legal_move_sequences: list[list[Move]],
        player: Player,
    ) -> list[Move]:
        """Return a uniformly-sampled legal move sequence.

        Parameters
        ----------
        board:
            Current board state (unused — random selection needs no evaluation).
        legal_move_sequences:
            All legal move sequences for this turn, as returned by
            ``Board.get_legal_moves(dice_roll)``.  Each element is a list of
            ``Move`` objects representing one full sequence for the dice rolled.
        player:
            The player whose turn it is (unused for random selection).

        Returns
        -------
        list[Move]
            A randomly chosen move sequence, or ``[]`` if no moves are
            available (forced pass).
        """
        if not legal_move_sequences:
            return []
        return random.choice(legal_move_sequences)
