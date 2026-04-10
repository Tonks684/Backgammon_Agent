"""
autoresearch/agent.py — parallel hyperparameter search loop.

Runs N_PARALLEL experiments simultaneously, each using a share of the CPU
workers and all sharing the GPU via time-multiplexing.

Parallelism breakdown (32-core CPU, single H200):
  N_PARALLEL=1 -> 1 experiment,  32 workers each  (~6 min per round)
  N_PARALLEL=4 -> 4 experiments,  8 workers each  (~6 min per round, 4x throughput)
  N_PARALLEL=8 -> 8 experiments,  4 workers each  (~6 min per round, 8x throughput)

Usage (from repo root):
  python autoresearch/agent.py                        # 4 parallel, random search
  python autoresearch/agent.py --parallel 8           # 8 parallel
  python autoresearch/agent.py --strategy grid        # exhaustive grid
  python autoresearch/agent.py --max-experiments 80   # stop after N total
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
GRID = {
    "ALPHA":           [0.001, 0.003, 0.005, 0.01, 0.02, 0.05],
    "LAMBDA":          [0.5, 0.6, 0.7, 0.8, 0.9],
    "HIDDEN_SIZE":     [64, 128, 256],
    "N_HIDDEN_LAYERS": [1, 2, 3],
    "BATCH_SIZE":      [16, 32, 64],
}

TOTAL_CPU_WORKERS = 32   # match Lightning.ai free plan (32 cores)
RESULTS_FILE      = Path("data/autoresearch_results.jsonl")
LOG_DIR           = Path("data/autoresearch_logs")

# ---------------------------------------------------------------------------
# Worker entry point — must be top-level for ProcessPoolExecutor (spawn)
# ---------------------------------------------------------------------------

def _trial_worker(job: dict) -> dict:
    """Runs in a subprocess. Calls trial.run_trial() and returns metrics."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from autoresearch.trial import run_trial

    params      = job["params"]
    n_workers   = job["n_workers"]
    exp_id      = job["experiment_id"]

    metrics = run_trial(params, n_workers=n_workers)

    if metrics is None:
        return {"experiment_id": exp_id, "failed": True, **params}

    metrics["experiment_id"] = exp_id
    metrics["timestamp"]     = datetime.datetime.utcnow().isoformat()
    metrics.update(params)
    return metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_result(metrics: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")


def _print_leaderboard(results: list[dict]) -> None:
    valid = [r for r in results if not r.get("failed") and "val_bpb" in r]
    if not valid:
        return
    sorted_r = sorted(valid, key=lambda r: r["val_bpb"])
    print("\n--- Leaderboard (lower val_bpb = better) ---")
    print(f"{'Rank':>4}  {'val_bpb':>8}  {'win_rate':>8}  {'episodes':>8}  params")
    for rank, r in enumerate(sorted_r[:10], 1):
        p = (f"α={r.get('ALPHA')} λ={r.get('LAMBDA')} "
             f"h={r.get('HIDDEN_SIZE')}×{r.get('N_HIDDEN_LAYERS')} "
             f"bs={r.get('BATCH_SIZE')}")
        print(f"{rank:>4}  {r['val_bpb']:>8.4f}  {r.get('win_rate',0):>8.4f}"
              f"  {int(r.get('episodes',0)):>8,}  {p}")
    print()


def _random_params() -> dict:
    return {k: random.choice(v) for k, v in GRID.items()}


def _grid_params() -> list[dict]:
    keys = list(GRID.keys())
    return [dict(zip(keys, combo)) for combo in product(*GRID.values())]


def _load_prior_results() -> tuple[list[dict], float, int]:
    """Load existing results for resume. Returns (results, best_val_bpb, next_id)."""
    results: list[dict] = []
    best = float("inf")
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    results.append(r)
                    if r.get("val_bpb", float("inf")) < best:
                        best = r["val_bpb"]
    next_id = max((r.get("experiment_id", 0) for r in results), default=0) + 1
    return results, best, next_id

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel autoresearch hyperparameter search")
    parser.add_argument("--strategy",        choices=["random", "grid"], default="random")
    parser.add_argument("--parallel", "-p",  type=int, default=4,
                        help="Number of experiments to run simultaneously (default: 4)")
    parser.add_argument("--max-experiments", type=int, default=None)
    args = parser.parse_args()

    n_parallel  = args.parallel
    n_workers   = max(1, TOTAL_CPU_WORKERS // n_parallel)

    results, best_val_bpb, experiment_id = _load_prior_results()
    if results:
        print(f"Resumed: {len(results)} prior results, best val_bpb={best_val_bpb:.4f}")

    print(f"Strategy : {args.strategy}")
    print(f"Parallel : {n_parallel} experiments × {n_workers} CPU workers each")
    print(f"GPU      : shared across all parallel experiments")

    if args.strategy == "grid":
        candidates = _grid_params()
        random.shuffle(candidates)
        print(f"Grid     : {len(candidates)} total combinations")
        candidate_iter = iter(candidates)
    else:
        candidate_iter = None
        print("Running until interrupted (Ctrl-C to stop)")

    max_exp    = args.max_experiments
    total_done = len(results)

    # Use 'spawn' context to avoid CUDA fork issues
    import multiprocessing
    mp_context = multiprocessing.get_context("spawn")

    try:
        with ProcessPoolExecutor(max_workers=n_parallel, mp_context=mp_context) as executor:
            while True:
                # Build a batch of n_parallel jobs
                jobs = []
                for _ in range(n_parallel):
                    if args.strategy == "grid":
                        try:
                            params = next(candidate_iter)
                        except StopIteration:
                            break
                    else:
                        params = _random_params()

                    jobs.append({
                        "params":        params,
                        "n_workers":     n_workers,
                        "experiment_id": experiment_id,
                    })
                    experiment_id += 1

                if not jobs:
                    print("Grid search complete.")
                    break

                print(f"\n=== Round: {n_parallel} experiments in parallel ===")
                for j in jobs:
                    print(f"  [Exp {j['experiment_id']}] {j['params']}")

                futures = {executor.submit(_trial_worker, j): j for j in jobs}
                round_results = []

                for future in as_completed(futures):
                    metrics = future.result()
                    job     = futures[future]
                    exp_id  = job["experiment_id"]

                    if metrics.get("failed"):
                        print(f"  [Exp {exp_id}] FAILED")
                        continue

                    results.append(metrics)
                    round_results.append(metrics)
                    _save_result(metrics)
                    total_done += 1

                    marker = ""
                    if metrics["val_bpb"] < best_val_bpb:
                        best_val_bpb = metrics["val_bpb"]
                        marker = "  <-- NEW BEST"

                    print(f"  [Exp {exp_id}] val_bpb={metrics['val_bpb']:.4f}"
                          f"  win_rate={metrics.get('win_rate',0):.4f}"
                          f"  episodes={int(metrics.get('episodes',0)):,}{marker}")

                _print_leaderboard(results)

                if max_exp is not None and total_done >= max_exp:
                    print(f"Reached max experiments ({max_exp}).")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    if results:
        _print_leaderboard(results)
        print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
