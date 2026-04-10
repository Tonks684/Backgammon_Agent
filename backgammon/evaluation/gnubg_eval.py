"""
Interface to GNU Backgammon (gnubg) for agent evaluation.

Runs gnubg in ``--tty`` mode via subprocess (stdin/stdout).
Designed for the Oracle VM (Linux) where gnubg is installed via apt.

Usage
-----
    evaluator = GnubgEvaluator(skill_level="expert")
    results = evaluator.evaluate_match(agent, n_matches=100)
    print(results)  # {'win_rate': 0.54, 'gammon_rate': 0.12, 'backgammon_rate': 0.01}

gnubg Position ID format
-------------------------
The 14-character base-64 position ID used by gnubg encodes 80 bits:
  bits  0-39 : on-move player's checkers (unary / "transit" encoding)
  bits 40-79 : off-move player's checkers (from that player's own perspective)

For each player (40 bits):
  Iterate through positions 0-24 (0 = own ace-point, 23 = own 24-point, 24 = bar).
  For each position: emit (count_of_checkers_here) 1-bits, then one 0-bit separator.
  Maximum: 15 ones + 25 zeros = 40 bits exactly.

Board coordinate mapping (from board.py conventions):
  WHITE: moves 23→0.  board index j == gnubg position j   (ace-point at j=0).
  BLACK: moves 0→23.  board index j == gnubg position 23-j (ace-point at j=23).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

from backgammon.game.board import Board
from backgammon.game.types import BAR_POINT, BEAROFF_POINT, DiceRoll, GameResult, Move, Player

if TYPE_CHECKING:
    pass

_GNUBG_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

_SKILL_LEVELS = {
    "beginner": "beginner",
    "intermediate": "intermediate",
    "advanced": "advanced",
    "expert": "expert",
    "world_class": "world class",
    "grandmaster": "grandmaster",
}

# Patterns for parsing gnubg --tty output
_RE_DICE_WHITE = re.compile(r"(?:You|White)\s+rolled\s+(\d)\s+(?:and\s+)?(\d)", re.IGNORECASE)
_RE_DICE_BLACK = re.compile(r"(?:Black|GNUBG|Computer)\s+rolled\s*[:\s]\s*(\d)\s+(?:and\s+)?(\d)", re.IGNORECASE)
_RE_GNUBG_MOVE = re.compile(r"(?:Black|GNUBG|Computer)\s+moves?\s+(.*?)[\.\n]", re.IGNORECASE)
_RE_WHITE_WINS = re.compile(r"White\s+wins\s+(\d+)\s+point", re.IGNORECASE)
_RE_BLACK_WINS = re.compile(r"Black\s+wins\s+(\d+)\s+point", re.IGNORECASE)
_RE_GAMMON = re.compile(r"gammon", re.IGNORECASE)
_RE_BACKGAMMON = re.compile(r"backgammon", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Position ID encoding
# ---------------------------------------------------------------------------

def _encode_player_bits(positions: list[int], bar: int) -> list[int]:
    """Encode one player's checkers into 40 bits (unary per position).

    Parameters
    ----------
    positions:
        Checker counts at gnubg positions 0-23 (24 board points, ace-point first).
    bar:
        Checker count on the bar (gnubg position 24).
    """
    bits: list[int] = []
    all_positions = positions + [bar]  # indices 0-24
    for count in all_positions:
        bits.extend([1] * count)
        bits.append(0)  # separator
    # Trim or pad to exactly 40 bits
    bits = bits[:40]
    while len(bits) < 40:
        bits.append(0)
    return bits


def board_to_position_id(board: Board, player: Player) -> str:
    """Encode the board into a 14-character gnubg position ID.

    The on-move player's checkers are encoded first (bits 0-39) from that
    player's own perspective (gnubg position 0 = own ace-point).  The
    opponent's checkers follow (bits 40-79) from the opponent's perspective.

    Parameters
    ----------
    board:
        The current board state.
    player:
        The player whose turn it is (determines which player's bits come first).
    """
    if player == Player.WHITE:
        # WHITE: board index j → gnubg position j (ace-point = index 0)
        current_pos = [max(board.points[j], 0) for j in range(24)]
        current_bar = board.bar[Player.WHITE]
        # BLACK opponent from BLACK's perspective: ace-point = board index 23
        opp_pos = [max(-board.points[23 - j], 0) for j in range(24)]
        opp_bar = board.bar[Player.BLACK]
    else:
        # BLACK: board index (23-j) → gnubg position j (ace-point = index 23)
        current_pos = [max(-board.points[23 - j], 0) for j in range(24)]
        current_bar = board.bar[Player.BLACK]
        # WHITE opponent from WHITE's perspective
        opp_pos = [max(board.points[j], 0) for j in range(24)]
        opp_bar = board.bar[Player.WHITE]

    bits = _encode_player_bits(current_pos, current_bar)
    bits += _encode_player_bits(opp_pos, opp_bar)
    # bits is now exactly 80 elements

    # Convert to an 80-bit integer (bit[0] = LSB)
    value = 0
    for i, b in enumerate(bits):
        if b:
            value |= (1 << i)

    # Produce 14 base-64 characters (6 bits each, reading from LSB)
    chars = []
    for _ in range(14):
        chars.append(_GNUBG_B64[value & 0x3F])
        value >>= 6
    return "".join(chars)


# ---------------------------------------------------------------------------
# Move notation conversion
# ---------------------------------------------------------------------------

def _point_to_gnubg(point: int, player: Player) -> str:
    """Convert an internal board point index to gnubg human-readable notation.

    gnubg always uses the *current player's* perspective (1 = ace-point):
      WHITE: index j → notation str(j+1)
      BLACK: index j → notation str(24-j)
    """
    if point == BAR_POINT:
        return "bar"
    if point == BEAROFF_POINT:
        return "off"
    if player == Player.WHITE:
        return str(point + 1)
    else:
        return str(24 - point)


def _seq_to_gnubg_move(seq: list[Move], player: Player) -> str:
    """Convert a move sequence to gnubg ``move`` command string.

    Example: ``move 24/21 13/11``
    """
    parts = [
        f"{_point_to_gnubg(m.from_point, player)}/{_point_to_gnubg(m.to_point, player)}"
        for m in seq
    ]
    return "move " + " ".join(parts)


def _parse_gnubg_move(move_str: str, player: Player) -> list[Move]:
    """Parse a gnubg move string (e.g. ``13/11 24/22``) into Move objects.

    Returns an empty list if parsing fails.
    """
    moves = []
    for token in move_str.strip().split():
        m = re.match(r"(bar|\d+)/(off|\d+)\*?", token, re.IGNORECASE)
        if not m:
            continue
        src_s, dst_s = m.group(1), m.group(2)

        def _from_notation(s: str, is_src: bool) -> int:
            if s.lower() == "bar":
                return BAR_POINT
            if s.lower() == "off":
                return BEAROFF_POINT
            n = int(s)
            if player == Player.WHITE:
                return n - 1
            else:
                return 24 - n

        try:
            moves.append(Move(_from_notation(src_s, True), _from_notation(dst_s, False)))
        except (ValueError, IndexError):
            continue
    return moves


# ---------------------------------------------------------------------------
# GnubgEvaluator
# ---------------------------------------------------------------------------

class GnubgEvaluator:
    """Evaluate an agent by playing full matches against GNU Backgammon.

    gnubg is launched once per ``evaluate_match`` call in ``--tty`` mode.
    Our agent always plays as White; gnubg plays as Black.

    Parameters
    ----------
    gnubg_path:
        Path to the gnubg binary (default ``"gnubg"``, relies on PATH).
    readline_timeout:
        Seconds to wait for a line from gnubg before giving up (default 30).
    """

    def __init__(
        self,
        gnubg_path: str = "gnubg",
        readline_timeout: float = 30.0,
    ) -> None:
        self._gnubg_path = gnubg_path
        self._timeout = readline_timeout
        self._check_gnubg_available()

    def _check_gnubg_available(self) -> None:
        if shutil.which(self._gnubg_path) is None:
            raise RuntimeError(
                f"gnubg not found on PATH (searched for '{self._gnubg_path}').\n"
                "Install it on the Oracle VM with:\n"
                "    sudo apt-get install gnubg\n"
                "Then re-run this evaluation."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_match(
        self,
        agent,
        skill_level: str = "expert",
        n_matches: int = 100,
    ) -> dict:
        """Play ``n_matches`` games against gnubg at the given skill level.

        Our agent plays as White; gnubg plays as Black.

        Parameters
        ----------
        agent:
            Any agent implementing ``select_move(board, legal_move_sequences, player)``.
        skill_level:
            One of: ``"beginner"``, ``"intermediate"``, ``"advanced"``,
            ``"expert"``, ``"world_class"``, ``"grandmaster"``.
        n_matches:
            Number of games to play.

        Returns
        -------
        dict with keys:
            ``win_rate``         – fraction of games White (our agent) won
            ``gammon_rate``      – fraction of games ending in a gammon (either side)
            ``backgammon_rate``  – fraction of games ending in a backgammon (either side)
        """
        if skill_level not in _SKILL_LEVELS:
            raise ValueError(
                f"Unknown skill level '{skill_level}'. "
                f"Choose from: {list(_SKILL_LEVELS)}"
            )
        gnubg_skill = _SKILL_LEVELS[skill_level]

        results: list[GameResult] = []
        for _ in range(n_matches):
            result = self._play_one_game(agent, gnubg_skill)
            results.append(result)

        n = len(results)
        wins = sum(
            1 for r in results
            if r in (GameResult.WHITE_WIN, GameResult.WHITE_GAMMON, GameResult.WHITE_BACKGAMMON)
        )
        gammons = sum(
            1 for r in results
            if r in (
                GameResult.WHITE_GAMMON, GameResult.WHITE_BACKGAMMON,
                GameResult.BLACK_GAMMON, GameResult.BLACK_BACKGAMMON,
            )
        )
        backgammons = sum(
            1 for r in results
            if r in (GameResult.WHITE_BACKGAMMON, GameResult.BLACK_BACKGAMMON)
        )
        return {
            "win_rate": wins / n if n else 0.0,
            "gammon_rate": gammons / n if n else 0.0,
            "backgammon_rate": backgammons / n if n else 0.0,
        }

    # ------------------------------------------------------------------
    # Single-game loop
    # ------------------------------------------------------------------

    def _play_one_game(self, agent, gnubg_skill: str) -> GameResult:
        """Launch gnubg, play one complete game, return the GameResult."""
        proc = subprocess.Popen(
            [self._gnubg_path, "--tty"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        try:
            return self._run_game(proc, agent, gnubg_skill)
        finally:
            try:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            proc.wait(timeout=5)

    def _send(self, proc: subprocess.Popen, cmd: str) -> None:
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def _readline(self, proc: subprocess.Popen) -> str:
        """Read one line from gnubg stdout (with timeout guard)."""
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if line:
                return line
        return ""

    def _read_until(self, proc: subprocess.Popen, patterns: list[re.Pattern]) -> tuple[str, re.Match | None]:
        """Read lines until one matches any pattern; return (line, match)."""
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                continue
            for pat in patterns:
                m = pat.search(line)
                if m:
                    return line, m
        return "", None

    def _run_game(self, proc: subprocess.Popen, agent, gnubg_skill: str) -> GameResult:
        """Drive a full game via gnubg tty commands.

        Protocol:
          - We always play White (human); gnubg plays Black (computer).
          - We poll gnubg output line by line and react to recognised patterns.
          - On our turn: roll → enumerate legal moves → agent selects → send move.
          - On gnubg's turn: read until gnubg announces its move.
          - Detect game-over by "White wins" / "Black wins" patterns.

        Note: gnubg output format is version-dependent.  The patterns here
        were tested against gnubg 1.06.x.  If your gnubg version produces
        different output, adjust the ``_RE_*`` constants at the top of this
        module.
        """
        board = Board()

        # --- Setup ---
        # Wait for gnubg startup prompt
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if ">" in line or "gnubg" in line.lower():
                break

        self._send(proc, "new game")
        self._send(proc, "set player 0 human")
        self._send(proc, "set player 1 gnubg")
        self._send(proc, f"set player 1 skill {gnubg_skill}")

        _game_over_pats = [_RE_WHITE_WINS, _RE_BLACK_WINS]
        _all_pats = [
            _RE_DICE_WHITE,
            _RE_DICE_BLACK,
            _RE_GNUBG_MOVE,
            _RE_WHITE_WINS,
            _RE_BLACK_WINS,
        ]

        # Maximum turns guard (a full game rarely exceeds 300 half-moves)
        for _ in range(600):
            line, match = self._read_until(proc, _all_pats)
            if not line:
                break  # timeout — treat as draw / incomplete

            # --- Game over ---
            if _RE_WHITE_WINS.search(line):
                win_line = line
                # Check if subsequent lines mention gammon/backgammon
                if _RE_BACKGAMMON.search(win_line):
                    return GameResult.WHITE_BACKGAMMON
                if _RE_GAMMON.search(win_line):
                    return GameResult.WHITE_GAMMON
                return GameResult.WHITE_WIN

            if _RE_BLACK_WINS.search(line):
                win_line = line
                if _RE_BACKGAMMON.search(win_line):
                    return GameResult.BLACK_BACKGAMMON
                if _RE_GAMMON.search(win_line):
                    return GameResult.BLACK_GAMMON
                return GameResult.BLACK_WIN

            # --- White's (our) dice ---
            m = _RE_DICE_WHITE.search(line)
            if m:
                d1, d2 = int(m.group(1)), int(m.group(2))
                dice = DiceRoll(d1, d2)
                legal = board.get_legal_moves(dice)
                seq = agent.select_move(board, legal, Player.WHITE)
                if seq:
                    self._send(proc, _seq_to_gnubg_move(seq, Player.WHITE))
                    board.apply_move_sequence(seq)
                else:
                    # Forced pass
                    self._send(proc, "move")
                    board.current_player = board.current_player.opponent()
                continue

            # --- Black's (gnubg) dice ---
            m_bdice = _RE_DICE_BLACK.search(line)
            if m_bdice:
                # gnubg will automatically make its move; we just wait
                continue

            # --- gnubg's move announcement ---
            m_move = _RE_GNUBG_MOVE.search(line)
            if m_move:
                move_str = m_move.group(1)
                gnubg_moves = _parse_gnubg_move(move_str, Player.BLACK)
                if gnubg_moves:
                    # Apply each individual move in place without switching player,
                    # then switch player at the end (apply_move_sequence handles this).
                    # We construct DiceRoll-independent application here:
                    player = Player.BLACK
                    for mv in gnubg_moves:
                        board._apply_move_inplace(player, mv)
                    board.current_player = Player.WHITE
                continue

        # If we exit the loop without a terminal state, return in-progress
        # (shouldn't happen in normal play)
        result = board.get_result()
        if result == GameResult.IN_PROGRESS:
            # Fallback: treat as black win (we timed out / something went wrong)
            return GameResult.BLACK_WIN
        return result
