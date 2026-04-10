"""
Encodes a Board state into a 54-dimensional float32 vector.

Always encodes from the perspective of `player` — own checkers are always
in dims 0-23, opponent checkers in dims 24-47. This symmetry means the
network always sees the board the same way regardless of which colour it is.

Dimensions:
  0-23  : own checker counts per point, normalised by 15
  24-47 : opponent checker counts per point, normalised by 15
  48    : own bar count / 15
  49    : opponent bar count / 15
  50    : own borne-off count / 15
  51    : opponent borne-off count / 15
  52    : own pip count / 167  (167 = starting pip count)
  53    : game phase as float  (0.0 = contact, 0.5 = race, 1.0 = bearing off)
"""

import numpy as np

from backgammon.game.board import Board
from backgammon.game.types import Player

_MAX_CHECKERS = 15.0
_STARTING_PIPS = 167.0

STATE_SIZE = 54


def encode(board: Board, player: Player) -> np.ndarray:
    """
    Encode the board from `player`'s perspective.

    Returns:
        np.ndarray of shape (54,), dtype float32, all values in [0, 1].
    """
    vec = np.zeros(STATE_SIZE, dtype=np.float32)

    if player == Player.WHITE:
        own_sign = 1
        # own home: points 0-5, travels toward bearoff at low end
        own_points = range(24)
        opp_points = range(24)
    else:
        own_sign = -1
        # Flip board: BLACK sees the board mirrored so point 23 becomes 0
        own_points = range(23, -1, -1)
        opp_points = range(23, -1, -1)

    opp = player.opponent()

    # Own checker counts (dims 0-23)
    for out_idx, board_idx in enumerate(own_points):
        val = board.points[board_idx] * own_sign
        vec[out_idx] = max(val, 0) / _MAX_CHECKERS

    # Opponent checker counts (dims 24-47)
    opp_sign = -own_sign
    for out_idx, board_idx in enumerate(opp_points):
        val = board.points[board_idx] * opp_sign
        vec[24 + out_idx] = max(val, 0) / _MAX_CHECKERS

    # Bar (dims 48-49)
    vec[48] = board.bar[player] / _MAX_CHECKERS
    vec[49] = board.bar[opp] / _MAX_CHECKERS

    # Borne off (dims 50-51)
    vec[50] = board.borne_off[player] / _MAX_CHECKERS
    vec[51] = board.borne_off[opp] / _MAX_CHECKERS

    # Pip count (dim 52) — own pip count only, normalised
    vec[52] = board.pip_count(player) / _STARTING_PIPS

    # Game phase (dim 53)
    if board.all_checkers_in_home(player):
        vec[53] = 1.0   # bearing off
    elif not board.is_contact():
        vec[53] = 0.5   # pure race
    else:
        vec[53] = 0.0   # contact

    return vec
