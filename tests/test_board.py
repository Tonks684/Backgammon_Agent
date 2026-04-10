"""Unit tests for backgammon/game/board.py"""

import pytest
from backgammon.game.board import Board, STARTING_POINTS
from backgammon.game.types import (
    BAR_POINT, BEAROFF_POINT, DiceRoll, GameResult, Move, Player
)


# ---------------------------------------------------------------------------
# Starting position
# ---------------------------------------------------------------------------

class TestStartingPosition:
    def test_starting_pip_count_white(self):
        b = Board()
        assert b.pip_count(Player.WHITE) == 167

    def test_starting_pip_count_black(self):
        b = Board()
        assert b.pip_count(Player.BLACK) == 167

    def test_starting_no_bar(self):
        b = Board()
        assert b.bar[Player.WHITE] == 0
        assert b.bar[Player.BLACK] == 0

    def test_starting_no_borne_off(self):
        b = Board()
        assert b.borne_off[Player.WHITE] == 0
        assert b.borne_off[Player.BLACK] == 0

    def test_starting_player_is_white(self):
        b = Board()
        assert b.current_player == Player.WHITE

    def test_total_white_checkers(self):
        b = Board()
        total = sum(v for v in b.points if v > 0)
        assert total == 15

    def test_total_black_checkers(self):
        b = Board()
        total = sum(abs(v) for v in b.points if v < 0)
        assert total == 15


# ---------------------------------------------------------------------------
# Legal move generation — basic
# ---------------------------------------------------------------------------

class TestLegalMoves:
    def test_legal_moves_returns_list(self):
        b = Board()
        moves = b.get_legal_moves(DiceRoll(1, 2))
        assert isinstance(moves, list)
        assert len(moves) > 0

    def test_doubles_produce_four_moves(self):
        b = Board()
        sequences = b.get_legal_moves(DiceRoll(3, 3))
        # With doubles each sequence should use up to 4 dice
        max_len = max(len(s) for s in sequences)
        assert max_len == 4

    def test_non_doubles_produce_two_moves(self):
        b = Board()
        sequences = b.get_legal_moves(DiceRoll(1, 2))
        max_len = max(len(s) for s in sequences)
        assert max_len == 2

    def test_must_move_from_bar_first(self):
        b = Board()
        b.bar[Player.WHITE] = 1
        b.points[23] -= 1  # remove one white checker from point 23
        sequences = b.get_legal_moves(DiceRoll(1, 2))
        # All moves must start from bar
        for seq in sequences:
            assert seq[0].from_point == BAR_POINT

    def test_blot_can_be_hit(self):
        b = Board()
        # Clear the board for a controlled test
        b.points = [0] * 24
        b.points[20] = 1   # WHITE checker at 20
        b.points[19] = -1  # BLACK blot at 19 (one checker)
        b.current_player = Player.WHITE
        sequences = b.get_legal_moves(DiceRoll(1, 6))
        flat_moves = [m for seq in sequences for m in seq]
        hitting_moves = [m for m in flat_moves if m.from_point == 20 and m.to_point == 19]
        assert len(hitting_moves) > 0

    def test_cannot_land_on_opponent_prime(self):
        b = Board()
        b.points = [0] * 24
        b.points[22] = 1   # WHITE at 22
        b.points[20] = -2  # BLACK prime at 20
        b.current_player = Player.WHITE
        sequences = b.get_legal_moves(DiceRoll(2, 3))
        for seq in sequences:
            for m in seq:
                assert not (m.from_point == 22 and m.to_point == 20)


# ---------------------------------------------------------------------------
# Bearing off
# ---------------------------------------------------------------------------

class TestBearingOff:
    def _bearing_off_board(self) -> Board:
        """Board where WHITE has all 15 checkers in home (points 0-5)."""
        b = Board()
        b.points = [0] * 24
        b.points[5] = 15   # all WHITE checkers on point 5
        b.points[23] = -15 # all BLACK on point 23 (not interacting)
        return b

    def test_all_checkers_in_home_white(self):
        b = self._bearing_off_board()
        assert b.all_checkers_in_home(Player.WHITE)

    def test_bearoff_move_generated(self):
        b = self._bearing_off_board()
        b.current_player = Player.WHITE
        sequences = b.get_legal_moves(DiceRoll(5, 6))
        flat = [m for seq in sequences for m in seq]
        bearoff = [m for m in flat if m.to_point == BEAROFF_POINT]
        assert len(bearoff) > 0

    def test_bearoff_increments_borne_off(self):
        b = self._bearing_off_board()
        b.current_player = Player.WHITE
        sequences = b.get_legal_moves(DiceRoll(6, 6))
        # Apply first sequence — die 6 from point 5 → dest -1 → bearoff
        b.apply_move_sequence(sequences[0])
        assert b.borne_off[Player.WHITE] > 0


# ---------------------------------------------------------------------------
# Win / gammon / backgammon detection
# ---------------------------------------------------------------------------

class TestGameResult:
    def test_in_progress_at_start(self):
        b = Board()
        assert b.get_result() == GameResult.IN_PROGRESS

    def test_white_normal_win(self):
        b = Board()
        b.points = [0] * 24
        b.borne_off[Player.WHITE] = 15
        b.borne_off[Player.BLACK] = 1   # BLACK has borne off at least 1
        result = b.get_result()
        assert result == GameResult.WHITE_WIN

    def test_white_gammon(self):
        b = Board()
        b.points = [0] * 24
        b.borne_off[Player.WHITE] = 15
        b.borne_off[Player.BLACK] = 0
        b.bar[Player.BLACK] = 0
        # BLACK checkers are NOT in WHITE's home board
        b.points[23] = -15
        result = b.get_result()
        assert result == GameResult.WHITE_GAMMON

    def test_white_backgammon(self):
        b = Board()
        b.points = [0] * 24
        b.borne_off[Player.WHITE] = 15
        b.borne_off[Player.BLACK] = 0
        b.bar[Player.BLACK] = 1  # BLACK has checker on bar → backgammon
        result = b.get_result()
        assert result == GameResult.WHITE_BACKGAMMON

    def test_black_normal_win(self):
        b = Board()
        b.points = [0] * 24
        b.borne_off[Player.BLACK] = 15
        b.borne_off[Player.WHITE] = 1
        result = b.get_result()
        assert result == GameResult.BLACK_WIN

    def test_is_terminal_when_won(self):
        b = Board()
        b.points = [0] * 24
        b.borne_off[Player.WHITE] = 15
        b.borne_off[Player.BLACK] = 1
        assert b.is_terminal()

    def test_not_terminal_at_start(self):
        b = Board()
        assert not b.is_terminal()


# ---------------------------------------------------------------------------
# apply_move_sequence
# ---------------------------------------------------------------------------

class TestApplyMoveSequence:
    def test_player_switches_after_move(self):
        b = Board()
        sequences = b.get_legal_moves(DiceRoll(1, 2))
        b.apply_move_sequence(sequences[0])
        assert b.current_player == Player.BLACK

    def test_checker_moves_off_source(self):
        b = Board()
        # WHITE has 2 checkers on point 23 at start
        initial_count = b.points[23]
        sequences = b.get_legal_moves(DiceRoll(1, 2))
        b.apply_move_sequence(sequences[0])
        # At least one checker should have moved from 23
        assert b.points[23] <= initial_count

    def test_hitting_sends_to_bar(self):
        b = Board()
        b.points = [0] * 24
        b.points[20] = 1   # WHITE
        b.points[19] = -1  # BLACK blot
        b.current_player = Player.WHITE
        b.apply_move_sequence([Move(20, 19)])
        assert b.bar[Player.BLACK] == 1
        assert b.points[19] == 1  # WHITE now owns it


# ---------------------------------------------------------------------------
# Board copy
# ---------------------------------------------------------------------------

class TestCopy:
    def test_copy_is_independent(self):
        b = Board()
        c = b.copy()
        c.points[0] = 99
        assert b.points[0] != 99

    def test_copy_bar_independent(self):
        b = Board()
        c = b.copy()
        c.bar[Player.WHITE] = 5
        assert b.bar[Player.WHITE] == 0
