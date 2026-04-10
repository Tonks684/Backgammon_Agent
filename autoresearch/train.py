"""
autoresearch experiment script — backgammon TD(λ) hyperparameter search.

The autoresearch agent modifies ONLY this file (below the TUNABLE PARAMETERS
comment). It runs for exactly BUDGET_SECONDS, then prints metrics in the
format the agent greps for.

The agent reads:  grep "^val_bpb:" run.log
We output:        val_bpb: X.XXXX   (lower is better — we use 1 - win_rate)
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from multiprocessing import Pool

from backgammon.agents.random_agent import RandomAgent
from backgammon.agents.td_lambda import TDLambdaAgent
from backgammon.config import Config
from backgammon.game.board import Board
from backgammon.game.types import DiceRoll, GameResult, Player
from backgammon.models.mlp import ValueNetwork
from backgammon.training.self_play import play_batch

# ---------------------------------------------------------------------------
# Fixed — do not change
# ---------------------------------------------------------------------------
BUDGET_SECONDS = 300

# ---------------------------------------------------------------------------
# TUNABLE PARAMETERS — autoresearch agent will modify these
# ---------------------------------------------------------------------------
ALPHA         = 0.01    # TD learning rate
LAMBDA        = 0.7     # eligibility trace decay
HIDDEN_SIZE   = 128     # neurons per hidden layer
N_HIDDEN_LAYERS = 2     # number of hidden layers
BATCH_SIZE    = 32      # games collected before each weight update step
N_WORKERS     = 32      # parallel CPU workers (max = CPU core count)
# ---------------------------------------------------------------------------


def evaluate_vs_random(agent, n_games: int = 500) -> float:
    """Win rate of agent (WHITE) vs random opponent (BLACK). Frozen — do not modify."""
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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(
        alpha=ALPHA,
        lambda_=LAMBDA,
        hidden_size=HIDDEN_SIZE,
        n_hidden_layers=N_HIDDEN_LAYERS,
        batch_size=BATCH_SIZE,
        n_workers=N_WORKERS,
        checkpoint_dir="data/autoresearch_checkpoints/",
        eval_dir="data/autoresearch_evals/",
    )

    agent = TDLambdaAgent(
        network=ValueNetwork(hidden_size=HIDDEN_SIZE, n_hidden_layers=N_HIDDEN_LAYERS),
        config=config,
        device=device,
    )

    t0 = time.time()
    episode = 0
    step = 0

    # Persistent pool — created once, eliminates ~2s spawn overhead per batch
    with Pool(processes=N_WORKERS) as pool:
        while time.time() - t0 < BUDGET_SECONDS:
            batch = play_batch(agent, BATCH_SIZE, N_WORKERS, pool=pool)
            for trajectory, result in batch:
                agent.update(trajectory, result)
                episode += 1
            step += 1

    training_seconds = time.time() - t0
    win_rate = evaluate_vs_random(agent, n_games=500)
    total_seconds = time.time() - t0

    # val_bpb convention: lower is better — report 1 - win_rate
    val_bpb = 1.0 - win_rate

    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1e6
                    if device.type == "cuda" else 0.0)

    # Metrics block — agent greps for these exact prefixes
    print("---")
    print(f"val_bpb: {val_bpb:.6f}")
    print(f"win_rate: {win_rate:.6f}")
    print(f"episodes: {episode}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds: {total_seconds:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print(f"hidden_size: {HIDDEN_SIZE}")
    print(f"n_hidden_layers: {N_HIDDEN_LAYERS}")
    print(f"alpha: {ALPHA}")
    print(f"lambda: {LAMBDA}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"n_workers: {N_WORKERS}")


if __name__ == "__main__":
    main()
