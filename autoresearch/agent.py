"""
autoresearch/agent.py — hyperparameter search loop.

Runs train.py repeatedly with different parameter combinations, logs results
to data/autoresearch_results.jsonl, and prints a leaderboard. Kill with Ctrl-C.

Strategy: random search over the parameter grid. Simple, embarrassingly
parallel-friendly, and surprisingly effective for <= 6 dimensions.

Usage (from repo root):
    python autoresearch/agent.py
    python autoresearch/agent.py --strategy grid   # exhaustive grid search
    python autoresearch/agent.py --max-experiments 50
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import re
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Parameter grid — edit these to expand / restrict the search space
# ---------------------------------------------------------------------------
GRID = {
    "ALPHA":          [0.001, 0.003, 0.005, 0.01, 0.02, 0.05],
    "LAMBDA":         [0.5, 0.6, 0.7, 0.8, 0.9],
    "HIDDEN_SIZE":    [64, 128, 256],
    "N_HIDDEN_LAYERS":[1, 2, 3],
    "BATCH_SIZE":     [16, 32, 64],
    "N_WORKERS":      [32],   # keep at CPU core count
}

TRAIN_SCRIPT = Path(__file__).parent / "train.py"
RESULTS_FILE = Path("data/autoresearch_results.jsonl")
LOG_DIR      = Path("data/autoresearch_logs")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_train_py(params: dict) -> None:
    """Overwrite the TUNABLE PARAMETERS block in train.py."""
    src = TRAIN_SCRIPT.read_text(encoding="utf-8")

    replacements = {
        "ALPHA":           f"ALPHA         = {params['ALPHA']}",
        "LAMBDA":          f"LAMBDA        = {params['LAMBDA']}",
        "HIDDEN_SIZE":     f"HIDDEN_SIZE   = {params['HIDDEN_SIZE']}",
        "N_HIDDEN_LAYERS": f"N_HIDDEN_LAYERS = {params['N_HIDDEN_LAYERS']}",
        "BATCH_SIZE":      f"BATCH_SIZE    = {params['BATCH_SIZE']}",
        "N_WORKERS":       f"N_WORKERS     = {params['N_WORKERS']}",
    }

    for key, new_line in replacements.items():
        src = re.sub(
            rf"^{key}\s*=.*$",
            new_line,
            src,
            flags=re.MULTILINE,
        )

    TRAIN_SCRIPT.write_text(src, encoding="utf-8")


def _parse_metrics(output: str) -> dict | None:
    """Extract val_bpb and other metrics from train.py stdout."""
    metrics: dict = {}
    for line in output.splitlines():
        m = re.match(r"^(\w+):\s+([\d.]+)$", line.strip())
        if m:
            key, val = m.group(1), m.group(2)
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val
    return metrics if "val_bpb" in metrics else None


def _run_experiment(params: dict, experiment_id: int) -> dict | None:
    """Patch train.py, run it, parse output. Returns metrics dict or None."""
    _patch_train_py(params)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"exp_{experiment_id:04d}.log"

    print(f"\n[Exp {experiment_id}] {params}")
    print(f"  log -> {log_path}")

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        capture_output=True,
        text=True,
    )
    wall = time.time() - t0

    combined = proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(combined, encoding="utf-8")

    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode}) — see {log_path}")
        return None

    metrics = _parse_metrics(proc.stdout)
    if metrics is None:
        print(f"  WARNING: val_bpb not found in output — see {log_path}")
        return None

    metrics["wall_seconds"] = round(wall, 1)
    metrics["experiment_id"] = experiment_id
    metrics["timestamp"] = datetime.datetime.utcnow().isoformat()
    metrics.update(params)
    return metrics


def _save_result(metrics: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")


def _print_leaderboard(results: list[dict]) -> None:
    sorted_r = sorted(results, key=lambda r: r["val_bpb"])
    print("\n--- Leaderboard (lower val_bpb = better) ---")
    print(f"{'Rank':>4}  {'val_bpb':>8}  {'win_rate':>8}  params")
    for rank, r in enumerate(sorted_r[:10], 1):
        p = (f"α={r['ALPHA']} λ={r['LAMBDA']} "
             f"h={r['HIDDEN_SIZE']}×{r['N_HIDDEN_LAYERS']} "
             f"bs={r['BATCH_SIZE']}")
        print(f"{rank:>4}  {r['val_bpb']:>8.4f}  {r.get('win_rate', 0):>8.4f}  {p}")
    print()


def _random_params() -> dict:
    return {k: random.choice(v) for k, v in GRID.items()}


def _grid_params() -> list[dict]:
    keys = list(GRID.keys())
    return [dict(zip(keys, combo)) for combo in product(*GRID.values())]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Autoresearch hyperparameter search")
    parser.add_argument("--strategy", choices=["random", "grid"], default="random")
    parser.add_argument("--max-experiments", type=int, default=None,
                        help="Stop after N experiments (default: run forever for random, "
                             "exhaust grid for grid)")
    args = parser.parse_args()

    results: list[dict] = []
    best_val_bpb = float("inf")
    experiment_id = 1

    if args.strategy == "grid":
        candidates = _grid_params()
        random.shuffle(candidates)   # avoid bias from ordering
        print(f"Grid search: {len(candidates)} combinations")
        iterator = iter(candidates)
    else:
        iterator = None
        print("Random search: running until interrupted (Ctrl-C to stop)")

    max_exp = args.max_experiments

    try:
        while True:
            if args.strategy == "grid":
                try:
                    params = next(iterator)
                except StopIteration:
                    print("Grid search complete.")
                    break
            else:
                params = _random_params()

            metrics = _run_experiment(params, experiment_id)

            if metrics is not None:
                results.append(metrics)
                _save_result(metrics)

                val_bpb = metrics["val_bpb"]
                marker = ""
                if val_bpb < best_val_bpb:
                    best_val_bpb = val_bpb
                    marker = "  <-- NEW BEST"
                print(f"  val_bpb={val_bpb:.4f}  win_rate={metrics.get('win_rate', 0):.4f}"
                      f"  episodes={int(metrics.get('episodes', 0)):,}{marker}")

                if len(results) % 5 == 0:
                    _print_leaderboard(results)

            experiment_id += 1

            if max_exp is not None and experiment_id > max_exp:
                print(f"Reached max experiments ({max_exp}).")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    if results:
        _print_leaderboard(results)
        print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
