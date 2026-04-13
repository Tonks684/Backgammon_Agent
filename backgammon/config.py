from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # TD learning hyperparameters
    # Best config from 29-trial autoresearch on H200 (exp #7, val_bpb=0.038)
    alpha: float = 0.01
    lambda_: float = 0.9

    # Network architecture
    hidden_size: int = 128
    n_hidden_layers: int = 3

    # Training schedule
    n_episodes: int = 50_000   # first run on 16-core Oracle VM (~7-9 hrs); increase once gnubg signal confirmed
    eval_every: int = 2_000
    checkpoint_every: int = 10_000

    # Paths
    checkpoint_dir: str = "data/checkpoints/"
    eval_dir: str = "data/evals/"

    # Parallelism — Oracle VM has 16 cores; reserve 2 for OS
    n_workers: int = 14        # CPU workers for parallel game generation
    batch_size: int = 64       # trajectories collected before each update step
    n_gpus: int = 0            # no GPU on free Oracle VM

    # Logging
    wandb_project: str = "backgammon-rl"
    run_name: Optional[str] = None

    @classmethod
    def from_json(cls, path: str) -> Config:
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {k: getattr(self, k) for k in self.__dataclass_fields__},
                f,
                indent=2,
            )
