"""Self-play utilities for TD(λ) training.

Both sides of every game use the same agent (true self-play).  States are
always encoded from White's perspective so that the TD target is consistent
across the entire trajectory.
"""

from __future__ import annotations

import random
from typing import Type

import numpy as np

from backgammon.game.board import Board
from backgammon.game.encoder import encode
from backgammon.game.types import DiceRoll, GameResult, Player


def play_game(
    agent,
    board_cls: Type[Board] = Board,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], GameResult]:
    """Play one complete game using *agent* for both sides.

    Trajectory encoding convention:
        All states are encoded from ``Player.WHITE``'s perspective so that the
        value-network target is always expressed in the same coordinate frame.
        When it is BLACK's turn the board is still encoded from WHITE's viewpoint
        — the network sees a board where WHITE's equity has dropped after BLACK's
        move, which is exactly the signal needed for the TD update.

    Parameters
    ----------
    agent:
        Any agent implementing
        ``select_move(board, legal_move_sequences, player) -> list[Move]``.
    board_cls:
        Board class to instantiate (injectable for testing).

    Returns
    -------
    trajectory:
        List of ``(state_before, state_after)`` pairs, one per half-move.
        Each element is a pair of ``(54,)`` float32 numpy arrays encoded from
        White's perspective.
    result:
        Terminal ``GameResult`` (never ``IN_PROGRESS``).
    """
    board = board_cls()
    trajectory: list[tuple[np.ndarray, np.ndarray]] = []

    # Safety cap — a real backgammon game is almost never longer than ~300
    # half-moves; cap at 500 to guard against bugs/infinite loops.
    for _ in range(500):
        if board.is_terminal():
            break

        player = board.current_player
        d1 = random.randint(1, 6)
        d2 = random.randint(1, 6)
        dice = DiceRoll(d1, d2)

        legal_seqs = board.get_legal_moves(dice)

        # Encode state *before* the move (from WHITE's perspective)
        state_before = encode(board, Player.WHITE)

        seq = agent.select_move(board, legal_seqs, player)

        if seq:
            board.apply_move_sequence(seq)
        else:
            # Forced pass — just switch player
            board.current_player = board.current_player.opponent()

        # Encode state *after* the move (still from WHITE's perspective)
        state_after = encode(board, Player.WHITE)

        trajectory.append((state_before, state_after))

    result = board.get_result()
    # If somehow the loop ended without a terminal state, declare a draw-like
    # result as black win (should not happen in normal play).
    if result == GameResult.IN_PROGRESS:
        result = GameResult.BLACK_WIN

    return trajectory, result


def play_n_games(
    agent,
    n: int,
    board_cls: Type[Board] = Board,
) -> list[GameResult]:
    """Play *n* self-play games and return the list of results.

    Parameters
    ----------
    agent:
        Agent with a ``select_move`` method (same agent used for both sides).
    n:
        Number of games to play.
    board_cls:
        Board class to instantiate.

    Returns
    -------
    list[GameResult]
        Terminal results, one per game, in order.
    """
    return [play_game(agent, board_cls=board_cls)[1] for _ in range(n)]
