"""
autoresearch/train.py — standalone experiment script (thin wrapper).

Calls trial.run_trial() with the TUNABLE PARAMETERS defined below.
Modify only the parameters under the TUNABLE PARAMETERS comment.

The agent reads:  grep "^val_bpb:" run.log
We output:        val_bpb: X.XXXXXX   (lower is better — we use 1 - win_rate)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Fixed — do not change
# ---------------------------------------------------------------------------
BUDGET_SECONDS = 300

# ---------------------------------------------------------------------------
# TUNABLE PARAMETERS — autoresearch agent will modify these
# ---------------------------------------------------------------------------
ALPHA           = 0.01    # TD learning rate
LAMBDA          = 0.7     # eligibility trace decay
HIDDEN_SIZE     = 128     # neurons per hidden layer
N_HIDDEN_LAYERS = 2       # number of hidden layers
BATCH_SIZE      = 32      # games collected before each weight update step
N_WORKERS       = 32      # parallel CPU workers (max = CPU core count)
# ---------------------------------------------------------------------------


def main() -> None:
    from autoresearch.trial import run_trial

    params = {
        "ALPHA":           ALPHA,
        "LAMBDA":          LAMBDA,
        "HIDDEN_SIZE":     HIDDEN_SIZE,
        "N_HIDDEN_LAYERS": N_HIDDEN_LAYERS,
        "BATCH_SIZE":      BATCH_SIZE,
    }

    metrics = run_trial(params, n_workers=N_WORKERS)
    if metrics is None:
        print("val_bpb: 1.000000")
        return

    print("---")
    print(f"val_bpb: {metrics['val_bpb']:.6f}")
    print(f"win_rate: {metrics['win_rate']:.6f}")
    print(f"episodes: {metrics['episodes']}")
    print(f"training_seconds: {metrics['training_seconds']:.1f}")
    print(f"total_seconds: {metrics['total_seconds']:.1f}")
    print(f"peak_vram_mb: {metrics['peak_vram_mb']:.1f}")
    print(f"hidden_size: {HIDDEN_SIZE}")
    print(f"n_hidden_layers: {N_HIDDEN_LAYERS}")
    print(f"alpha: {ALPHA}")
    print(f"lambda: {LAMBDA}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"n_workers: {N_WORKERS}")


if __name__ == "__main__":
    main()
