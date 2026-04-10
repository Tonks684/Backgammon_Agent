"""
autoresearch/trial.py — single experiment runner.

Contains the training logic as a callable function so agent.py can run
multiple trials in parallel (ProcessPoolExecutor) without file-patching.

train.py is kept as a thin wrapper around run_trial() for standalone use.
"""

from __future__ import annotations

import random
import time
from multiprocessing import Pool
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from backgammon.agents.random_agent import RandomAgent
from backgammon.game.board import Board
from backgammon.game.types import DiceRoll, GameResult, Player


BUDGET_SECONDS = 300  # fixed — do not change


def evaluate_vs_random(agent, n_games: int = 500) -> float:
    """Win rate of agent (WHITE) vs random opponent (BLACK). Frozen."""
    random_agent = RandomAgent()
    wins = 0
    for _ in range(n_games):
        board = Board()
        for _ in range(500):
            if board.is_terminal():
                break
            player = board.current_player
            dice = DiceRoll(random.randint(1, 6), random.randint(1, 6))
            legal = board.get_legal_moves(dice)
            seq = (agent.select_move(board, legal, player)
                   if player == Player.WHITE
                   else random_agent.select_move(board, legal, player))
            if seq:
                board.apply_move_sequence(seq)
            else:
                board.current_player = board.current_player.opponent()
        result = board.get_result()
        if result in (GameResult.WHITE_WIN, GameResult.WHITE_GAMMON,
                      GameResult.WHITE_BACKGAMMON):
            wins += 1
    return wins / n_games


def run_trial(params: dict, n_workers: int = 32) -> dict | None:
    """Run one training experiment and return metrics.

    Parameters
    ----------
    params:
        Dict with keys: ALPHA, LAMBDA, HIDDEN_SIZE, N_HIDDEN_LAYERS, BATCH_SIZE.
    n_workers:
        CPU workers available to this trial. When running N_PARALLEL trials
        simultaneously, pass total_cpu_cores // N_PARALLEL.

    Returns
    -------
    dict with val_bpb, win_rate, episodes, etc. — or None on failure.
    """
    try:
        from backgammon.agents.td_lambda import TDLambdaAgent
        from backgammon.config import Config
        from backgammon.models.mlp import ValueNetwork
        from backgammon.training.self_play import play_batch

        alpha          = params["ALPHA"]
        lambda_        = params["LAMBDA"]
        hidden_size    = params["HIDDEN_SIZE"]
        n_hidden_layers = params["N_HIDDEN_LAYERS"]
        batch_size     = params["BATCH_SIZE"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = Config(
            alpha=alpha,
            lambda_=lambda_,
            hidden_size=hidden_size,
            n_hidden_layers=n_hidden_layers,
            batch_size=batch_size,
            n_workers=n_workers,
            checkpoint_dir="data/autoresearch_checkpoints/",
            eval_dir="data/autoresearch_evals/",
        )

        network = ValueNetwork(hidden_size=hidden_size, n_hidden_layers=n_hidden_layers)
        if hasattr(torch, "compile"):
            network = torch.compile(network)

        agent = TDLambdaAgent(network=network, config=config, device=device)

        t0 = time.time()
        episode = 0

        with Pool(processes=n_workers) as pool:
            while time.time() - t0 < BUDGET_SECONDS:
                batch = play_batch(agent, batch_size, n_workers, pool=pool)
                for trajectory, result in batch:
                    agent.update(trajectory, result)
                    episode += 1

        training_seconds = time.time() - t0
        win_rate = evaluate_vs_random(agent, n_games=500)
        total_seconds = time.time() - t0

        peak_vram_mb = (torch.cuda.max_memory_allocated() / 1e6
                        if device.type == "cuda" else 0.0)

        return {
            "val_bpb":          round(1.0 - win_rate, 6),
            "win_rate":         round(win_rate, 6),
            "episodes":         episode,
            "training_seconds": round(training_seconds, 1),
            "total_seconds":    round(total_seconds, 1),
            "peak_vram_mb":     round(peak_vram_mb, 1),
        }

    except Exception as e:
        print(f"  trial ERROR: {e}")
        return None
