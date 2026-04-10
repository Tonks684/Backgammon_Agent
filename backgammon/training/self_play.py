"""Self-play utilities for TD(λ) training.

Both sides of every game use the same agent (true self-play).  States are
always encoded from White's perspective so that the TD target is consistent
across the entire trajectory.

Parallelism
-----------
play_batch() distributes game generation across n_workers CPU processes.
Each worker receives a snapshot of the current network weights (CPU tensors),
plays n games independently, and returns the trajectories.  The main process
then runs the TD(λ) update on GPU sequentially over the collected batch.

Scaling up:
  - More CPU cores  → increase n_workers in Config
  - Bigger GPU      → increase batch_size in Config
  - Multiple GPUs   → increase n_gpus in Config (triggers DDP in trainer)
"""

from __future__ import annotations

import random
from multiprocessing import Pool
from typing import Type

import numpy as np
import torch

from backgammon.game.board import Board
from backgammon.game.encoder import encode
from backgammon.game.types import DiceRoll, GameResult, Player


# ---------------------------------------------------------------------------
# Single-game loop (used by workers and directly in tests)
# ---------------------------------------------------------------------------

def play_game(
    agent,
    board_cls: Type[Board] = Board,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], GameResult]:
    """Play one complete game using *agent* for both sides.

    All states encoded from WHITE's perspective — see module docstring.
    """
    board = board_cls()
    trajectory: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(500):
        if board.is_terminal():
            break

        player = board.current_player
        dice = DiceRoll(random.randint(1, 6), random.randint(1, 6))
        legal_seqs = board.get_legal_moves(dice)

        state_before = encode(board, Player.WHITE)
        seq = agent.select_move(board, legal_seqs, player)

        if seq:
            board.apply_move_sequence(seq)
        else:
            board.current_player = board.current_player.opponent()

        state_after = encode(board, Player.WHITE)
        trajectory.append((state_before, state_after))

    result = board.get_result()
    if result == GameResult.IN_PROGRESS:
        result = GameResult.BLACK_WIN

    return trajectory, result


def play_n_games(
    agent,
    n: int,
    board_cls: Type[Board] = Board,
) -> list[GameResult]:
    """Play n self-play games sequentially, return results."""
    return [play_game(agent, board_cls=board_cls)[1] for _ in range(n)]


# ---------------------------------------------------------------------------
# Parallel batch generation
# ---------------------------------------------------------------------------

def _worker_fn(args: tuple) -> list[tuple]:
    """Worker entry point — runs in a separate process.

    Receives network weights as CPU state-dict (picklable), reconstructs
    a local CPU agent, plays n_games, returns trajectories + results.
    """
    state_dict, n_games, hidden_size, n_hidden_layers, alpha, lambda_ = args

    # Import locally — avoids issues with CUDA handles being inherited
    from backgammon.agents.td_lambda import TDLambdaAgent
    from backgammon.config import Config
    from backgammon.models.mlp import ValueNetwork

    network = ValueNetwork(hidden_size=hidden_size, n_hidden_layers=n_hidden_layers)
    network.load_state_dict(state_dict)

    cfg = Config(alpha=alpha, lambda_=lambda_,
                 hidden_size=hidden_size, n_hidden_layers=n_hidden_layers)
    agent = TDLambdaAgent(network, cfg, device=torch.device("cpu"))

    return [play_game(agent) for _ in range(n_games)]


def play_batch(
    agent,
    total_games: int,
    n_workers: int,
    pool: Pool | None = None,
) -> list[tuple[list[tuple[np.ndarray, np.ndarray]], GameResult]]:
    """Play total_games games across n_workers parallel CPU processes.

    Parameters
    ----------
    agent:
        The current TDLambdaAgent.  Its network weights are snapshotted and
        sent to each worker — the agent itself is NOT mutated.
    total_games:
        Total number of games to play across all workers.
    n_workers:
        Number of parallel CPU processes.
    pool:
        Optional persistent Pool. Pass a long-lived Pool from the Trainer
        to avoid the per-batch process spawn overhead (~2s per batch).
        If None, a temporary Pool is created and destroyed for this call.

    Scaling
    -------
    Increase n_workers (= Config.n_workers) to use more CPU cores.
    On Lightning.ai free plan: n_workers=32 matches the 32-core CPU Studio.
    On a beefier VM: n_workers=64 or higher.
    """
    # Snapshot weights as CPU tensors — safe to pickle across processes
    state_dict = {k: v.cpu() for k, v in agent.network.state_dict().items()}

    games_per_worker = max(1, total_games // n_workers)
    worker_counts = [games_per_worker] * n_workers
    remainder = total_games - games_per_worker * n_workers
    for i in range(remainder):
        worker_counts[i] += 1

    args = [
        (
            state_dict,
            count,
            agent.config.hidden_size,
            agent.config.n_hidden_layers,
            agent.config.alpha,
            agent.config.lambda_,
        )
        for count in worker_counts
        if count > 0
    ]

    if pool is not None:
        nested = pool.map(_worker_fn, args)
    else:
        with Pool(processes=len(args)) as p:
            nested = p.map(_worker_fn, args)

    return [item for batch in nested for item in batch]
