from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from backgammon.config import Config
from backgammon.evaluation.metrics import compute_equity_target
from backgammon.game.encoder import encode
from backgammon.game.types import GameResult, Move, Player
from backgammon.models.mlp import ValueNetwork

if TYPE_CHECKING:
    from backgammon.game.board import Board


class TDLambdaAgent:
    """TD(λ) agent for backgammon.

    Uses a ValueNetwork to evaluate positions and trains via temporal-difference
    learning with eligibility traces (Tesauro 1992/1995 convention).

    The network output is a 4-element probability vector:
        [p_white_win, p_white_gammon, p_black_win, p_black_gammon]

    Move selection: enumerate all legal move sequences after a dice roll,
    apply each to a copy of the board, encode the resulting position, evaluate
    with the network, and return the sequence with highest equity (from the
    current player's perspective).

    TD(λ) update (off-policy, full-game trajectory):
        δ_t  = V(s_{t+1}) - V(s_t)
        e_t  = λ · e_{t-1} + ∇V(s_t)
        Δw   = α · δ_t · e_t

    At the terminal step, V(s_T) is replaced by compute_equity_target(result).
    """

    def __init__(self, network: ValueNetwork, config: Config) -> None:
        self.network = network
        self.config = config
        self._alpha = config.alpha
        self._lambda = config.lambda_

    # ------------------------------------------------------------------
    # Move selection
    # ------------------------------------------------------------------

    def select_move(
        self,
        board: Board,
        legal_move_sequences: list[list[Move]],
        player: Player,
    ) -> list[Move]:
        """Return the move sequence that maximises equity for *player*.

        Parameters
        ----------
        board:
            Current board state (will not be mutated).
        legal_move_sequences:
            All legal move sequences available this turn. Each element is a
            list of Move objects representing one full sequence of moves for
            the rolled dice.
        player:
            The player whose turn it is.

        Returns
        -------
        list[Move]
            The best move sequence according to the value network.
        """
        if not legal_move_sequences:
            return []

        best_seq: list[Move] = []
        best_equity = float("-inf")

        self.network.eval()
        with torch.no_grad():
            for seq in legal_move_sequences:
                # Apply move sequence to a scratch copy of the board
                next_board = board.copy()
                next_board.apply_move_sequence(seq)

                # Always encode from WHITE's perspective to match training convention
                state_vec = encode(next_board, Player.WHITE)
                x = torch.tensor(state_vec, dtype=torch.float32)
                output = self.network(x)
                eq = ValueNetwork.equity(output).item()

                # White maximises equity; black minimises it
                signed_eq = eq if player == Player.WHITE else -eq
                if signed_eq > best_equity:
                    best_equity = signed_eq
                    best_seq = seq

        return best_seq

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(
        self,
        trajectory: list[tuple[np.ndarray, np.ndarray]],
        result: GameResult,
    ) -> None:
        """Run a TD(λ) weight update over one completed game trajectory.

        Parameters
        ----------
        trajectory:
            List of (state_vec, next_state_vec) pairs recorded during the game.
            Each element is a pair of 54-dimensional float32 numpy arrays.
            The final next_state_vec is ignored — the terminal target comes
            from *result* instead.
        result:
            The game outcome, used to build the terminal training target.
        """
        if not trajectory:
            return

        self.network.train()

        # Initialise eligibility traces (same shape as parameters)
        traces = [torch.zeros_like(p) for p in self.network.parameters()]
        terminal_target = torch.tensor(
            compute_equity_target(result), dtype=torch.float32
        )

        for t, (state_vec, next_state_vec) in enumerate(trajectory):
            is_terminal = t == len(trajectory) - 1

            x_t = torch.tensor(state_vec, dtype=torch.float32)

            # --- Forward pass for V(s_t) with gradient ---
            self.network.zero_grad()
            v_t = self.network(x_t)  # shape (4,)
            # Retain graph so we can call backward after computing δ
            v_t.sum().backward()

            # --- Compute TD target V(s_{t+1}) ---
            if is_terminal:
                v_next = terminal_target
            else:
                x_next = torch.tensor(next_state_vec, dtype=torch.float32)
                with torch.no_grad():
                    v_next = self.network(x_next)

            # --- TD error (vector, one per output head) ---
            with torch.no_grad():
                delta = v_next - v_t.detach()  # shape (4,)

            # --- Update eligibility traces and apply weight update ---
            with torch.no_grad():
                for param, trace, grad in zip(
                    self.network.parameters(),
                    traces,
                    (p.grad for p in self.network.parameters()),
                ):
                    # grad shape: (*param.shape)
                    # delta shape: (4,) — one scalar per output unit
                    # For a multi-output network we sum TD errors across heads
                    # (equivalent to treating each output independently and
                    # summing their contributions, as in Tesauro 1992).
                    delta_scalar = delta.sum().item()

                    # Accumulate trace: e_t = λ·e_{t-1} + ∇V(s_t)
                    trace.mul_(self._lambda).add_(grad)

                    # Weight update: Δw = α · δ · e
                    param.add_(self._alpha * delta_scalar * trace)

                    # Zero grad for next iteration
                    if param.grad is not None:
                        param.grad.zero_()
