"""
autoresearch experiment script — backgammon TD(λ) hyperparameter search.

This script is modified autonomously by the autoresearch agent.
It runs for exactly BUDGET_SECONDS, then prints a single score line.

autoresearch reads the last line matching "score: X.XXX" as the metric.
Higher is better (win rate vs random opponent, 0.0–1.0).

DO NOT change: Board, encoder, types (game rules are fixed).
The agent MAY change: anything below the "TUNABLE PARAMETERS" comment.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

# Make sure the package is importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from backgammon.agents.td_lambda import TDLambdaAgent
from backgammon.agents.random_agent import RandomAgent
from backgammon.config import Config
from backgammon.game.board import Board
from backgammon.game.encoder import encode
from backgammon.game.types import DiceRoll, GameResult, Player
from backgammon.models.mlp import ValueNetwork
from backgammon.training.self_play import play_batch, play_game

# ---------------------------------------------------------------------------
# Budget — do not change (autoresearch uses a fixed 5-minute window)
# ---------------------------------------------------------------------------
BUDGET_SECONDS = 300

# ---------------------------------------------------------------------------
# TUNABLE PARAMETERS — autoresearch will modify these
# ---------------------------------------------------------------------------
ALPHA = 0.01          # TD learning rate
LAMBDA = 0.7          # eligibility trace decay
HIDDEN_SIZE = 128     # neurons per hidden layer
N_HIDDEN_LAYERS = 2   # number of hidden layers
BATCH_SIZE = 32       # games collected before each weight update step
N_WORKERS = 32        # parallel CPU workers for game generation
# ---------------------------------------------------------------------------


def evaluate_vs_random(agent, n_games: int = 500) -> float:
    """Win rate of agent (WHITE) vs random opponent (BLACK)."""
    random_agent = RandomAgent()
    wins = 0
    board = Board()

    for _ in range(n_games):
        board = Board()
        for _ in range(500):
            if board.is_terminal():
                break
            player = board.current_player
            dice = DiceRoll(random.randint(1, 6), random.randint(1, 6))
            legal = board.get_legal_moves(dice)
            if player == Player.WHITE:
                seq = agent.select_move(board, legal, player)
            else:
                seq = random_agent.select_move(board, legal, player)
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
    )

    network = ValueNetwork(
        hidden_size=HIDDEN_SIZE,
        n_hidden_layers=N_HIDDEN_LAYERS,
    )
    agent = TDLambdaAgent(network=network, config=config, device=device)

    t0 = time.time()
    episode = 0
    step = 0

    # Training loop — runs until budget expires
    while time.time() - t0 < BUDGET_SECONDS:
        batch = play_batch(agent, BATCH_SIZE, N_WORKERS)
        for trajectory, result in batch:
            agent.update(trajectory, result)
            episode += 1
        step += 1

        elapsed = time.time() - t0
        remaining = BUDGET_SECONDS - elapsed
        # Print progress every ~30s
        if step % max(1, int(30 / (elapsed / step + 1e-9))) == 0:
            print(f"  step={step} episode={episode} elapsed={elapsed:.0f}s remaining={remaining:.0f}s",
                  flush=True)

    # Evaluate
    print(f"\nTraining complete: {episode} episodes in {time.time() - t0:.0f}s")
    score = evaluate_vs_random(agent, n_games=500)
    print(f"score: {score:.4f}")


if __name__ == "__main__":
    main()
