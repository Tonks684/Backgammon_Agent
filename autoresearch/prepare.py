"""
prepare.py — frozen environment setup for autoresearch.

Run once before the hyperparameter search begins.
This file is NOT modified by the autoresearch agent.

For this project the backgammon package is already installed via
`pip install -e .` in the repo root, so this is a no-op health check.
"""

import subprocess
import sys


def check_import(module: str) -> None:
    try:
        __import__(module)
        print(f"  ok  {module}")
    except ImportError as e:
        print(f"  MISSING  {module}: {e}")
        sys.exit(1)


def main() -> None:
    print("prepare.py: checking environment...")

    required = [
        "torch",
        "numpy",
        "backgammon",
        "backgammon.agents.td_lambda",
        "backgammon.training.self_play",
        "backgammon.models.mlp",
        "backgammon.config",
    ]
    for mod in required:
        check_import(mod)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    if device == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")

    print("prepare.py: all checks passed.")


if __name__ == "__main__":
    main()
