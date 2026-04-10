"""
Backgammon board state and rules engine.

Board orientation:
- Points 0-23 are numbered from WHITE's perspective.
- WHITE moves from high to low (point 23 → 0), bearing off past point 0.
- BLACK moves from low to high (point 0 → 23), bearing off past point 23.
- points[i] > 0  →  WHITE has that many checkers on point i
- points[i] < 0  →  BLACK has that many checkers on point i
- points[i] == 0 →  point is empty

Home boards:
- WHITE home: points 0-5
- BLACK home: points 18-23
"""

import copy
import random
from itertools import permutations
from typing import Optional

from backgammon.game.types import (
    BAR_POINT,
    BEAROFF_POINT,
    DiceRoll,
    GameResult,
    Move,
    Player,
)

# Starting position: positive = WHITE, negative = BLACK
STARTING_POINTS = [
    -2, 0, 0, 0, 0, 5,   # points 0-5:   BLACK×2 at pt0, WHITE×5 at pt5
    0, 3, 0, 0, 0, -5,   # points 6-11:  WHITE×3 at pt7, BLACK×5 at pt11
    5, 0, 0, 0, -3, 0,   # points 12-17: WHITE×5 at pt12, BLACK×3 at pt16
    -5, 0, 0, 0, 0, 2,   # points 18-23: BLACK×5 at pt18, WHITE×2 at pt23
]
# WHITE pip count = 5×6 + 3×8 + 5×13 + 2×24 = 167
# BLACK pip count = 2×24 + 5×13 + 3×8 + 5×6 = 167


class Board:
    __slots__ = ("points", "bar", "borne_off", "current_player")

    def __init__(self) -> None:
        # points[i]: positive = WHITE checkers, negative = BLACK checkers
        self.points: list[int] = list(STARTING_POINTS)
        # bar[Player] = number of checkers on the bar
        self.bar: dict[Player, int] = {Player.WHITE: 0, Player.BLACK: 0}
        # borne_off[Player] = number of checkers borne off
        self.borne_off: dict[Player, int] = {Player.WHITE: 0, Player.BLACK: 0}
        self.current_player: Player = Player.WHITE

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.points = list(self.points)
        b.bar = dict(self.bar)
        b.borne_off = dict(self.borne_off)
        b.current_player = self.current_player
        return b

    # ------------------------------------------------------------------
    # Pip counts
    # ------------------------------------------------------------------

    def pip_count(self, player: Player) -> int:
        """Total pip distance remaining for player to bear all checkers off."""
        if player == Player.WHITE:
            total = sum(
                count * (i + 1)
                for i, count in enumerate(self.points)
                if count > 0
            )
            total += self.bar[Player.WHITE] * 25
        else:
            total = sum(
                abs(count) * (24 - i)
                for i, count in enumerate(self.points)
                if count < 0
            )
            total += self.bar[Player.BLACK] * 25
        return total

    # ------------------------------------------------------------------
    # Game phase helpers
    # ------------------------------------------------------------------

    def all_checkers_in_home(self, player: Player) -> bool:
        """True if all of player's remaining checkers are in their home board."""
        if self.bar[player] > 0:
            return False
        if player == Player.WHITE:
            return all(self.points[i] <= 0 for i in range(6, 24))
        else:
            return all(self.points[i] >= 0 for i in range(0, 18))

    def is_contact(self) -> bool:
        """True if both players have checkers that can still interact."""
        white_max = max((i for i, v in enumerate(self.points) if v > 0), default=-1)
        if self.bar[Player.WHITE] > 0:
            white_max = 23
        black_min = min((i for i, v in enumerate(self.points) if v < 0), default=24)
        if self.bar[Player.BLACK] > 0:
            black_min = 0
        return white_max > black_min

    # ------------------------------------------------------------------
    # Legal move generation
    # ------------------------------------------------------------------

    def _checker_count(self, player: Player) -> int:
        sign = 1 if player == Player.WHITE else -1
        return (
            sum(v * sign for v in self.points if v * sign > 0)
            + self.bar[player]
            + self.borne_off[player]
        )

    def _point_owner(self, point: int) -> Optional[Player]:
        if self.points[point] > 0:
            return Player.WHITE
        if self.points[point] < 0:
            return Player.BLACK
        return None

    def _can_land(self, player: Player, point: int) -> bool:
        """Can player land on this point (empty, own checker, or single opponent blot)?"""
        owner = self._point_owner(point)
        if owner is None:
            return True
        if owner == player:
            return True
        # Opponent blot: can hit if only 1
        opp_count = abs(self.points[point])
        return opp_count == 1

    def _apply_single_move(self, board: "Board", player: Player, move: Move) -> "Board":
        """Return a new board with one move applied. Does not validate legality."""
        b = board.copy()
        sign = 1 if player == Player.WHITE else -1

        # Lift checker from source
        if move.from_point == BAR_POINT:
            b.bar[player] -= 1
        else:
            b.points[move.from_point] -= sign

        # Place on destination
        if move.to_point == BEAROFF_POINT:
            b.borne_off[player] += 1
        else:
            dest = move.to_point
            # Hit blot if present
            opp = player.opponent()
            opp_sign = -sign
            if b.points[dest] * opp_sign > 0:
                # There's an opponent blot here
                b.points[dest] = 0
                b.bar[opp] += 1
            b.points[dest] += sign

        return b

    def _moves_for_die(
        self, board: "Board", player: Player, die: int
    ) -> list[Move]:
        """All single-checker moves legal for this die value on the given board."""
        moves = []
        sign = 1 if player == Player.WHITE else -1
        direction = -1 if player == Player.WHITE else 1  # WHITE goes down, BLACK up

        # Must enter from bar first
        if board.bar[player] > 0:
            if player == Player.WHITE:
                entry = 24 - die  # die=1 → point 23, die=6 → point 18
            else:
                entry = die - 1  # die=1 → point 0, die=6 → point 5
            if 0 <= entry <= 23 and board._can_land(player, entry):
                moves.append(Move(BAR_POINT, entry))
            return moves

        # Normal moves from board points
        all_home = board.all_checkers_in_home(player)
        for i in range(24):
            if board.points[i] * sign <= 0:
                continue  # no own checker here
            dest = i + direction * die
            if 0 <= dest <= 23:
                if board._can_land(player, dest):
                    moves.append(Move(i, dest))
            elif all_home:
                # Bearing off
                if player == Player.WHITE:
                    # Moving off the low end: dest < 0
                    if dest < 0:
                        # Exact bear-off or highest checker rule
                        if dest == -1 or not any(
                            board.points[j] > 0 for j in range(i + 1, 6)
                        ):
                            moves.append(Move(i, BEAROFF_POINT))
                else:
                    # BLACK bearing off: dest > 23
                    if dest > 23:
                        if dest == 24 or not any(
                            board.points[j] < 0 for j in range(18, i)
                        ):
                            moves.append(Move(i, BEAROFF_POINT))
        return moves

    def _generate_sequences(
        self, board: "Board", player: Player, dice: list[int], used: list[Move]
    ) -> list[list[Move]]:
        """Recursively generate all legal move sequences consuming dice."""
        if not dice:
            return [used]

        results = []
        tried: set[tuple[int, int, int]] = set()

        for idx, die in enumerate(dice):
            if die in (d for d in dice[:idx]):
                continue  # skip duplicate die values already tried
            possible = self._moves_for_die(board, player, die)
            if not possible:
                continue
            for move in possible:
                key = (die, move.from_point, move.to_point)
                if key in tried:
                    continue
                tried.add(key)
                new_board = self._apply_single_move(board, player, move)
                remaining = list(dice)
                remaining.pop(idx)
                sub = self._generate_sequences(new_board, player, remaining, used + [move])
                results.extend(sub)

        if not results:
            # Could not use all dice; return what we have
            return [used] if used else []
        return results

    def get_legal_moves(self, dice_roll: DiceRoll) -> list[list[Move]]:
        """
        Return all legal move sequences for the current player given dice_roll.
        Each sequence is a list of Move objects (1-4 moves).
        Enforces the must-use-maximum-dice rule.
        Returns [[]] (one empty sequence) if no moves are possible (player is forced to pass).
        """
        player = self.current_player
        dice = dice_roll.die_values()

        sequences = self._generate_sequences(self, player, dice, [])

        if not sequences:
            return [[]]  # forced pass

        # Must use as many dice as possible
        max_len = max(len(s) for s in sequences)
        sequences = [s for s in sequences if len(s) == max_len]

        # If two dice and only one can be used, must use the higher die if possible
        if not dice_roll.is_doubles and max_len == 1:
            higher = max(dice_roll.d1, dice_roll.d2)
            lower = min(dice_roll.d1, dice_roll.d2)
            high_die_moves = self._moves_for_die(self, player, higher)
            if high_die_moves:
                low_die_moves = self._moves_for_die(self, player, lower)
                if low_die_moves:
                    # Both dice usable for 1 move; keep all
                    pass
                else:
                    sequences = [[m] for m in high_die_moves]
            # else: only lower die usable, keep all sequences

        # Deduplicate sequences
        seen: set[tuple[tuple[int, int], ...]] = set()
        unique = []
        for seq in sequences:
            key = tuple((m.from_point, m.to_point) for m in seq)
            if key not in seen:
                seen.add(key)
                unique.append(seq)
        return unique

    def apply_move_sequence(self, moves: list[Move]) -> None:
        """Apply a full move sequence in-place and switch current player."""
        player = self.current_player
        for move in moves:
            self._apply_move_inplace(player, move)
        self.current_player = player.opponent()

    def _apply_move_inplace(self, player: Player, move: Move) -> None:
        sign = 1 if player == Player.WHITE else -1

        if move.from_point == BAR_POINT:
            self.bar[player] -= 1
        else:
            self.points[move.from_point] -= sign

        if move.to_point == BEAROFF_POINT:
            self.borne_off[player] += 1
        else:
            dest = move.to_point
            opp = player.opponent()
            opp_sign = -sign
            if self.points[dest] * opp_sign > 0:
                self.points[dest] = 0
                self.bar[opp] += 1
            self.points[dest] += sign

    # ------------------------------------------------------------------
    # Terminal state detection
    # ------------------------------------------------------------------

    def get_result(self) -> GameResult:
        """Return current game result (may be IN_PROGRESS)."""
        for player, sign in ((Player.WHITE, 1), (Player.BLACK, -1)):
            if self.borne_off[player] == 15:
                opp = player.opponent()
                opp_borne_off = self.borne_off[opp]
                opp_on_bar = self.bar[opp]

                if opp_borne_off > 0:
                    return GameResult.WHITE_WIN if player == Player.WHITE else GameResult.BLACK_WIN

                # Check backgammon: opponent has checker on bar or in winner's home
                if player == Player.WHITE:
                    opp_in_home = any(self.points[i] < 0 for i in range(0, 6))
                    if opp_on_bar or opp_in_home:
                        return GameResult.WHITE_BACKGAMMON
                else:
                    opp_in_home = any(self.points[i] > 0 for i in range(18, 24))
                    if opp_on_bar or opp_in_home:
                        return GameResult.BLACK_BACKGAMMON

                return GameResult.WHITE_GAMMON if player == Player.WHITE else GameResult.BLACK_GAMMON

        return GameResult.IN_PROGRESS

    def is_terminal(self) -> bool:
        return self.get_result().is_terminal()

    def __repr__(self) -> str:
        lines = [f"Player to move: {self.current_player.name}"]
        lines.append(f"Bar  W:{self.bar[Player.WHITE]}  B:{self.bar[Player.BLACK]}")
        lines.append(f"Off  W:{self.borne_off[Player.WHITE]}  B:{self.borne_off[Player.BLACK]}")
        row = " ".join(f"{v:+3d}" for v in self.points)
        lines.append(f"Points: {row}")
        return "\n".join(lines)
