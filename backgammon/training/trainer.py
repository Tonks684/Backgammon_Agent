"""Trainer: orchestrates TD(λ) self-play training.

Parallelism model
-----------------
Each training step:
  1. Dispatch config.batch_size games across config.n_workers CPU processes
     (play_batch — CPU-bound game logic runs in parallel)
  2. Collect all trajectories back on the main process
  3. Run TD(λ) update on GPU sequentially over each trajectory

Scaling levers (all in Config):
  n_workers  — CPU workers for game generation  (match CPU core count)
  batch_size — games per update step            (match GPU memory)
  n_gpus     — 1 = single GPU, >1 = DDP        (match GPU count)

Upgrading:
  Free plan  : n_workers=32, batch_size=32,  n_gpus=1
  Paid plan  : n_workers=64, batch_size=128, n_gpus=2  (just change config)
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from multiprocessing import Pool

from backgammon.config import Config
from backgammon.evaluation.metrics import WinRateTracker
from backgammon.game.types import GameResult
from backgammon.training.self_play import play_batch, play_game

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
        TDLambdaAgent with select_move and update methods.
    config:
        Config dataclass — controls parallelism, schedule, and paths.
    evaluator:
        Optional GnubgEvaluator. Called every eval_every episodes.
    """

    def __init__(self, agent, config: Config, evaluator=None) -> None:
        self.agent = agent
        self.config = config
        self.evaluator = evaluator

        self._checkpoint_dir = Path(config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._eval_dir = Path(config.eval_dir)
        self._eval_dir.mkdir(parents=True, exist_ok=True)

        self._win_tracker = WinRateTracker(window=1000)
        self._recent_td_errors: deque[float] = deque(maxlen=1000)
        self._recent_game_lengths: deque[int] = deque(maxlen=1000)

        # Persistent worker pool — created once, reused every batch
        # Eliminates ~2s process-spawn overhead per batch call
        self._pool = Pool(processes=config.n_workers)
        print(f"Worker pool started: {config.n_workers} processes")

        # Multi-GPU: wrap network with DDP if n_gpus > 1
        self._ddp = False
        if config.n_gpus > 1 and torch.cuda.device_count() >= config.n_gpus:
            self._setup_ddp()

        self._wandb_run = None
        if _WANDB_AVAILABLE:
            self._wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.run_name,
                config=dataclasses.asdict(config),
                resume="allow",
            )

    # ------------------------------------------------------------------
    # DDP setup (multi-GPU — plug in when upgrading plan)
    # ------------------------------------------------------------------

    def _setup_ddp(self) -> None:
        """Wrap the network with DistributedDataParallel for multi-GPU training.

        Called automatically when config.n_gpus > 1.
        To use: set n_gpus=2 in config — no other changes needed.

        Note: TD(λ) does manual weight updates (not via optimizer), so after
        each update we broadcast rank-0 weights to all other ranks to keep
        them in sync. This is handled in _sync_ddp_weights().
        """
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank() % self.config.n_gpus
        torch.cuda.set_device(local_rank)
        self.agent.network = nn.parallel.DistributedDataParallel(
            self.agent.network,
            device_ids=[local_rank],
        )
        self._ddp = True
        print(f"DDP enabled across {self.config.n_gpus} GPUs")

    def _sync_ddp_weights(self) -> None:
        """Broadcast rank-0 weights to all ranks after manual TD update."""
        if not self._ddp:
            return
        for param in self.agent.network.parameters():
            dist.broadcast(param.data, src=0)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        n_episodes: Optional[int] = None,
        eval_every: Optional[int] = None,
        checkpoint_every: Optional[int] = None,
    ) -> None:
        n_episodes = n_episodes or self.config.n_episodes
        eval_every = eval_every or self.config.eval_every
        checkpoint_every = checkpoint_every or self.config.checkpoint_every
        batch_size = self.config.batch_size
        n_workers = self.config.n_workers

        total_steps = (n_episodes + batch_size - 1) // batch_size
        print(
            f"Training for {n_episodes:,} episodes | "
            f"batch={batch_size} | workers={n_workers} | "
            f"eval every {eval_every:,} | checkpoint every {checkpoint_every:,}"
        )

        episode = 0
        t0 = time.time()

        for step in range(1, total_steps + 1):
            # --- Parallel game generation (CPU, persistent pool) ---
            games_this_step = min(batch_size, n_episodes - episode)
            if games_this_step <= 0:
                break

            batch = play_batch(self.agent, games_this_step, n_workers, pool=self._pool)

            # --- Sequential TD(λ) updates (GPU) ---
            for trajectory, result in batch:
                self.agent.update(trajectory, result)
                self._win_tracker.record(result)
                self._recent_game_lengths.append(len(trajectory))

                if trajectory:
                    td_err = self._estimate_td_error(trajectory, result)
                    self._recent_td_errors.append(td_err)

                episode += 1

            # Sync weights across GPUs after batch updates
            self._sync_ddp_weights()

            # --- Logging ---
            if self._wandb_run is not None:
                metrics: dict = {
                    "episode": episode,
                    "game_length": _mean(self._recent_game_lengths),
                    "td_loss": _mean(self._recent_td_errors),
                }
                self._wandb_run.log(metrics, step=episode)

            # --- eval_every ---
            if episode % eval_every < batch_size:
                summary = self._win_tracker.summary()
                elapsed = time.time() - t0
                ep_per_s = episode / elapsed if elapsed > 0 else 0
                print(
                    f"Ep {episode:>8,} | "
                    f"white_wr={summary['white_win_rate']:.3f} | "
                    f"gammon_rate={summary['gammon_rate']:.3f} | "
                    f"avg_len={_mean(self._recent_game_lengths):.1f} | "
                    f"td_loss={_mean(self._recent_td_errors):.4f} | "
                    f"{ep_per_s:.0f} ep/s"
                )
                if self._wandb_run is not None:
                    self._wandb_run.log(
                        {"self_play_win_rate": summary["white_win_rate"],
                         "gammon_rate": summary["gammon_rate"]},
                        step=episode,
                    )
                if self.evaluator is not None:
                    self._run_gnubg_eval(episode)

            # --- Checkpoint ---
            if episode % checkpoint_every < batch_size:
                self._save_checkpoint(episode)

        # Final checkpoint (only if not already saved this step)
        if episode % checkpoint_every >= batch_size:
            self._save_checkpoint(episode)

        self._pool.close()
        self._pool.join()

        if self._wandb_run is not None:
            self._wandb_run.finish()
        print("Training complete.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_td_error(self, trajectory, result) -> float:
        from backgammon.evaluation.metrics import compute_equity_target
        from backgammon.models.mlp import ValueNetwork

        device = self.agent.device
        terminal_eq = float(ValueNetwork.equity(
            torch.tensor(compute_equity_target(result), dtype=torch.float32, device=device)
        ))
        s0, _ = trajectory[0]
        with torch.no_grad():
            v0 = self.agent.network(torch.tensor(s0, dtype=torch.float32, device=device))
            eq0 = float(ValueNetwork.equity(v0))
        return abs(terminal_eq - eq0)

    def _save_checkpoint(self, episode: int) -> None:
        ckpt_path = self._checkpoint_dir / f"checkpoint_{episode:08d}.pt"
        # Unwrap DDP if active
        network = (self.agent.network.module
                   if self._ddp else self.agent.network)
        network.save_checkpoint(ckpt_path)
        network.save_checkpoint(self._checkpoint_dir / "latest.pt")
        meta = {
            "episode": episode,
            "white_win_rate": self._win_tracker.white_win_rate,
            "gammon_rate": self._win_tracker.gammon_rate,
        }
        with open(self._checkpoint_dir / f"checkpoint_{episode:08d}.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Checkpoint saved -> {ckpt_path}")

    def _run_gnubg_eval(self, episode: int) -> None:
        skill = getattr(self.config, "gnubg_skill", "expert")
        n_matches = getattr(self.config, "gnubg_eval_matches", 100)
        print(f"  Running gnubg eval ({n_matches} matches, skill={skill}) ...")
        results = self.evaluator.evaluate_match(
            self.agent, skill_level=skill, n_matches=n_matches
        )
        print(
            f"  gnubg [{skill}]: win_rate={results['win_rate']:.3f} "
            f"gammon_rate={results['gammon_rate']:.3f}"
        )
        eval_path = self._eval_dir / f"gnubg_{skill}_{episode:08d}.json"
        with open(eval_path, "w") as f:
            json.dump({"episode": episode, "skill": skill, **results}, f, indent=2)
        if self._wandb_run is not None:
            self._wandb_run.log(
                {f"gnubg_{skill}_win_rate": results["win_rate"]}, step=episode
            )


def _mean(collection) -> float:
    items = list(collection)
    return sum(items) / len(items) if items else 0.0
