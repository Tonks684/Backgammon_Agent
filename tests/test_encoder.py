"""Unit tests for backgammon/game/encoder.py"""

import numpy as np
import pytest
from backgammon.game.board import Board
from backgammon.game.encoder import encode, STATE_SIZE
from backgammon.game.types import DiceRoll, Player


class TestEncoderShape:
    def test_output_shape(self):
        b = Board()
        vec = encode(b, Player.WHITE)
        assert vec.shape == (STATE_SIZE,)
        assert STATE_SIZE == 54

    def test_dtype_float32(self):
        b = Board()
        vec = encode(b, Player.WHITE)
        assert vec.dtype == np.float32

    def test_all_values_in_unit_interval(self):
        b = Board()
        for player in (Player.WHITE, Player.BLACK):
            vec = encode(b, player)
            assert np.all(vec >= 0.0), f"Negative values for {player}"
            assert np.all(vec <= 1.0), f"Values > 1 for {player}"


class TestEncoderSymmetry:
    def test_starting_position_symmetric(self):
        """At the start, both players have identical pip counts → dims 52 should match."""
        b = Board()
        w = encode(b, Player.WHITE)
        bk = encode(b, Player.BLACK)
        # Pip count (dim 52) should be equal at start
        assert abs(float(w[52]) - float(bk[52])) < 1e-5

    def test_own_checker_count_matches(self):
        """Sum of own checker dims (0-23) should equal total own checkers on board / 15."""
        b = Board()
        vec = encode(b, Player.WHITE)
        own_checkers_on_board = sum(v for v in b.points if v > 0)
        encoded_sum = float(vec[0:24].sum()) * 15
        assert abs(encoded_sum - own_checkers_on_board) < 1e-3

    def test_perspective_flip(self):
        """Encoding from BLACK's perspective should mirror the board."""
        b = Board()
        # Place a single WHITE checker only
        b.points = [0] * 24
        b.points[20] = 1  # WHITE checker near WHITE's home end
        b.borne_off[Player.WHITE] = 14  # fill the rest

        w_vec = encode(b, Player.WHITE)
        bk_vec = encode(b, Player.BLACK)

        # From WHITE's view, own checker is at position 20 (dim 20)
        # From BLACK's view, it appears as opponent at mirrored position 3 (23-20=3, dim 24+3)
        assert w_vec[20] > 0
        assert bk_vec[24 + 3] > 0


class TestEncoderBarAndBearoff:
    def test_bar_encoded(self):
        b = Board()
        b.bar[Player.WHITE] = 3
        vec = encode(b, Player.WHITE)
        assert abs(float(vec[48]) - 3 / 15.0) < 1e-5

    def test_opponent_bar_encoded(self):
        b = Board()
        b.bar[Player.BLACK] = 2
        vec = encode(b, Player.WHITE)
        assert abs(float(vec[49]) - 2 / 15.0) < 1e-5

    def test_borne_off_encoded(self):
        b = Board()
        b.borne_off[Player.WHITE] = 5
        vec = encode(b, Player.WHITE)
        assert abs(float(vec[50]) - 5 / 15.0) < 1e-5


class TestEncoderPhase:
    def test_contact_phase(self):
        b = Board()  # starting position is contact
        vec = encode(b, Player.WHITE)
        assert float(vec[53]) == 0.0

    def test_bearing_off_phase(self):
        b = Board()
        b.points = [0] * 24
        b.points[2] = 15   # all WHITE in home board (points 0-5)
        b.points[23] = -15
        vec = encode(b, Player.WHITE)
        assert float(vec[53]) == 1.0

    def test_race_phase(self):
        b = Board()
        # WHITE only in points 0-5, BLACK only in points 18-23 → no contact
        b.points = [0] * 24
        b.points[5] = 15
        b.points[18] = -15
        vec = encode(b, Player.WHITE)
        # Race (0.5) or bearing off (1.0) — WHITE not all in home yet
        assert float(vec[53]) in (0.5, 1.0)


class TestEncoderPipCount:
    def test_starting_pip_normalised(self):
        b = Board()
        vec = encode(b, Player.WHITE)
        # 167 / 167 = 1.0
        assert abs(float(vec[52]) - 1.0) < 1e-5

    def test_pip_decreases_after_move(self):
        b = Board()
        initial_vec = encode(b, Player.WHITE)
        from backgammon.game.types import DiceRoll
        sequences = b.get_legal_moves(DiceRoll(1, 2))
        b.apply_move_sequence(sequences[0])
        # Now it's BLACK's turn; encode for WHITE still
        new_vec = encode(b, Player.WHITE)
        assert float(new_vec[52]) < float(initial_vec[52])
