"""Trainer: orchestrates TD(λ) self-play training with optional gnubg evaluation.

wandb logging is built in (Task 15).  Set WANDB_MODE=disabled to skip it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

from backgammon.config import Config
from backgammon.evaluation.metrics import WinRateTracker
from backgammon.game.types import GameResult, Player
from backgammon.training.self_play import play_game

# Optional wandb import — gracefully degrade if not installed or disabled
_WANDB_DISABLED = os.environ.get("WANDB_MODE", "").lower() == "disabled"
try:
    if not _WANDB_DISABLED:
        import wandb
        _WANDB_AVAILABLE = True
    else:
        _WANDB_AVAILABLE = False
except ImportError:
    _WANDB_AVAILABLE = False


class Trainer:
    """Runs the TD(λ) self-play training loop.

    Parameters
    ----------
    agent:
        A ``TDLambdaAgent`` (or any object with ``select_move`` and ``update``).
    config:
        ``Config`` dataclass with all hyperparameters and paths.
    evaluator:
        Optional ``GnubgEvaluator``.  If provided, ``evaluate_match`` is called
        every ``eval_every`` episodes.
    """

    def __init__(
        self,
        agent,
        config: Config,
        evaluator=None,
    ) -> None:
        self.agent = agent
        self.config = config
        self.evaluator = evaluator

        self._checkpoint_dir = Path(config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._eval_dir = Path(config.eval_dir)
        self._eval_dir.mkdir(parents=True, exist_ok=True)

        # Rolling trackers
        self._win_tracker = WinRateTracker(window=1000)
        self._recent_td_errors: deque[float] = deque(maxlen=1000)
        self._recent_game_lengths: deque[int] = deque(maxlen=1000)

        # wandb setup
        self._wandb_run = None
        if _WANDB_AVAILABLE:
            self._wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.run_name,
                config=dataclasses.asdict(config),
                resume="allow",
            )

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(
        self,
        n_episodes: Optional[int] = None,
        eval_every: Optional[int] = None,
        checkpoint_every: Optional[int] = None,
    ) -> None:
        """Run the self-play training loop.

        Parameters
        ----------
        n_episodes:
            Total number of self-play games to run.  Defaults to
            ``config.n_episodes``.
        eval_every:
            Run gnubg evaluation (if evaluator is set) every this many
            episodes.  Defaults to ``config.eval_every``.
        checkpoint_every:
            Save a model checkpoint every this many episodes.  Defaults to
            ``config.checkpoint_every``.
        """
        n_episodes = n_episodes if n_episodes is not None else self.config.n_episodes
        eval_every = eval_every if eval_every is not None else self.config.eval_every
        checkpoint_every = checkpoint_every if checkpoint_every is not None else self.config.checkpoint_every

        print(f"Training for {n_episodes:,} episodes "
              f"(eval every {eval_every:,}, checkpoint every {checkpoint_every:,})")

        t0 = time.time()

        for episode in range(1, n_episodes + 1):
            # --- Self-play ---
            trajectory, result = play_game(self.agent)
            game_length = len(trajectory)

            # --- TD(λ) update ---
            self.agent.update(trajectory, result)

            # --- Metric tracking ---
            self._win_tracker.record(result)
            self._recent_game_lengths.append(game_length)

            # Approximate TD error: mean absolute change across trajectory
            if trajectory:
                td_err = self._estimate_td_error(trajectory, result)
                self._recent_td_errors.append(td_err)

            # --- Per-episode wandb log ---
            if self._wandb_run is not None:
                metrics: dict = {
                    "episode": episode,
                    "game_length": game_length,
                }
                if self._recent_td_errors:
                    metrics["td_loss"] = sum(self._recent_td_errors) / len(self._recent_td_errors)
                self._wandb_run.log(metrics, step=episode)

            # --- eval_every: self-play win-rate log + optional gnubg eval ---
            if episode % eval_every == 0:
                summary = self._win_tracker.summary()
                elapsed = time.time() - t0
                speed = episode / elapsed
                print(
                    f"Ep {episode:>8,} | "
                    f"white_wr={summary['white_win_rate']:.3f} | "
                    f"gammon_rate={summary['gammon_rate']:.3f} | "
                    f"avg_len={_mean(self._recent_game_lengths):.1f} | "
                    f"td_loss={_mean(self._recent_td_errors):.4f} | "
                    f"{speed:.0f} ep/s"
                )

                if self._wandb_run is not None:
                    self._wandb_run.log(
                        {
                            "self_play_win_rate": summary["white_win_rate"],
                            "gammon_rate": summary["gammon_rate"],
                        },
                        step=episode,
                    )

                if self.evaluator is not None:
                    self._run_gnubg_eval(episode)

            # --- Checkpoint ---
            if episode % checkpoint_every == 0:
                self._save_checkpoint(episode)

        # Final checkpoint
        self._save_checkpoint(n_episodes)
        if self._wandb_run is not None:
            self._wandb_run.finish()
        print("Training complete.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_td_error(
        self,
        trajectory: list[tuple],
        result: GameResult,
    ) -> float:
        """Rough mean absolute TD error over the trajectory (for logging).

        Uses the first state's network value vs the final terminal target
        as a proxy — avoids a full forward pass through every step.
        """
        import numpy as np
        import torch
        from backgammon.evaluation.metrics import compute_equity_target
        from backgammon.models.mlp import ValueNetwork

        terminal_eq = float(
            ValueNetwork.equity(
                torch.tensor(compute_equity_target(result), dtype=torch.float32)
            )
        )
        # Use the first state's equity as a proxy for V(s_0)
        s0, _ = trajectory[0]
        with torch.no_grad():
            v0 = self.agent.network(torch.tensor(s0, dtype=torch.float32))
            eq0 = float(ValueNetwork.equity(v0))
        return abs(terminal_eq - eq0)

    def _save_checkpoint(self, episode: int) -> None:
        ckpt_path = self._checkpoint_dir / f"checkpoint_{episode:08d}.pt"
        self.agent.network.save_checkpoint(ckpt_path)

        # Also write a "latest" symlink/file so main.py --resume can find it
        latest_path = self._checkpoint_dir / "latest.pt"
        self.agent.network.save_checkpoint(latest_path)

        # Save training metadata alongside checkpoint
        meta = {
            "episode": episode,
            "white_win_rate": self._win_tracker.white_win_rate,
            "gammon_rate": self._win_tracker.gammon_rate,
        }
        with open(self._checkpoint_dir / f"checkpoint_{episode:08d}.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Checkpoint saved -> {ckpt_path}")

    def _run_gnubg_eval(self, episode: int) -> None:
        """Run a gnubg evaluation and log results."""
        skill = getattr(self.config, "gnubg_skill", "expert")
        n_matches = getattr(self.config, "gnubg_eval_matches", 100)

        print(f"  Running gnubg eval ({n_matches} matches, skill={skill}) …")
        results = self.evaluator.evaluate_match(
            self.agent, skill_level=skill, n_matches=n_matches
        )
        print(
            f"  gnubg [{skill}]: win_rate={results['win_rate']:.3f} "
            f"gammon_rate={results['gammon_rate']:.3f} "
            f"backgammon_rate={results['backgammon_rate']:.3f}"
        )

        # Save to eval dir
        eval_path = self._eval_dir / f"gnubg_{skill}_{episode:08d}.json"
        with open(eval_path, "w") as f:
            json.dump({"episode": episode, "skill": skill, **results}, f, indent=2)

        if self._wandb_run is not None:
            self._wandb_run.log(
                {f"gnubg_{skill}_win_rate": results["win_rate"]},
                step=episode,
            )


def _mean(collection) -> float:
    items = list(collection)
    return sum(items) / len(items) if items else 0.0
