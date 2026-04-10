from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # TD learning hyperparameters
    alpha: float = 0.1
    lambda_: float = 0.7

    # Network architecture
    hidden_size: int = 128
    n_hidden_layers: int = 2

    # Training schedule
    n_episodes: int = 500_000
    eval_every: int = 10_000
    checkpoint_every: int = 50_000

    # Paths
    checkpoint_dir: str = "data/checkpoints/"
    eval_dir: str = "data/evals/"

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
