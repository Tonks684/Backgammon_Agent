"""
autoresearch/benchmark.py — parallelism dry run.

Runs short 30-second trials at N_PARALLEL = 1, 2, 4, 8, 16 to find the
optimal setting for your hardware before committing to overnight runs.

Measures:
  - Episodes/second  (training throughput — higher is better)
  - GPU memory used  (want headroom before OOM)
  - Effective speedup vs single experiment

Usage (from repo root inside Docker):
  python autoresearch/benchmark.py
  python autoresearch/benchmark.py --budget 60   # 60s trials for more accuracy
  python autoresearch/benchmark.py --max-parallel 8  # stop at 8 instead of 16
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TOTAL_CPU_WORKERS = 32
BENCH_PARAMS = {
    "ALPHA":           0.01,
    "LAMBDA":          0.7,
    "HIDDEN_SIZE":     128,
    "N_HIDDEN_LAYERS": 2,
    "BATCH_SIZE":      32,
}


# ---------------------------------------------------------------------------
# Workers — must be top-level for spawn
# ---------------------------------------------------------------------------

def _bench_worker_proc(job: dict, queue) -> None:
    """Wrapper so result goes into a Queue (non-daemon Process pattern)."""
    try:
        result = _bench_worker(job)
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {"episodes": 0, "elapsed": job["budget_seconds"], "ep_per_sec": 0,
                  "peak_vram_mb": 0, "n_workers": job["n_workers"],
                  "error": str(e)}
    queue.put(result)


def _bench_worker(job: dict) -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import time
    import torch
    import multiprocessing
    from backgammon.agents.td_lambda import TDLambdaAgent
    from backgammon.config import Config
    from backgammon.models.mlp import ValueNetwork
    from backgammon.training.self_play import play_batch

    params    = job["params"]
    n_workers = job["n_workers"]
    budget    = job["budget_seconds"]

    # CRITICAL: create Pool BEFORE initialising CUDA.
    # Forking after CUDA is initialised causes deadlocks on Linux.
    # Use spawn context so workers start clean regardless of parent state.
    ctx  = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=n_workers)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        config = Config(
            alpha=params["ALPHA"], lambda_=params["LAMBDA"],
            hidden_size=params["HIDDEN_SIZE"], n_hidden_layers=params["N_HIDDEN_LAYERS"],
            batch_size=params["BATCH_SIZE"], n_workers=n_workers,
        )
        network = ValueNetwork(
            hidden_size=params["HIDDEN_SIZE"],
            n_hidden_layers=params["N_HIDDEN_LAYERS"],
        )
        agent = TDLambdaAgent(network=network, config=config, device=device)

        t0      = time.time()
        episode = 0

        while time.time() - t0 < budget:
            batch = play_batch(agent, params["BATCH_SIZE"], n_workers, pool=pool)
            for trajectory, result in batch:
                agent.update(trajectory, result)
                episode += 1

    finally:
        pool.close()
        pool.join()

    elapsed   = time.time() - t0
    peak_vram = (torch.cuda.max_memory_allocated() / 1e6
                 if device.type == "cuda" else 0.0)

    return {
        "episodes":     episode,
        "elapsed":      round(elapsed, 1),
        "ep_per_sec":   round(episode / elapsed, 2) if elapsed > 0 else 0,
        "peak_vram_mb": round(peak_vram, 1),
        "n_workers":    n_workers,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget",      type=int, default=120,
                        help="Seconds per trial (default: 120)")
    parser.add_argument("--max-parallel", type=int, default=16,
                        help="Highest N_PARALLEL to test (default: 16)")
    args = parser.parse_args()

    budget     = args.budget
    max_par    = args.max_parallel

    # Parallelism levels to test
    levels = [n for n in [1, 2, 4, 8, 16] if n <= max_par]

    print("=" * 60)
    print(f"Autoresearch parallelism benchmark")
    print(f"  Trial budget : {budget}s each")
    print(f"  Levels       : {levels}")
    print(f"  Total time   : ~{len(levels) * (budget + 15) // 60 + 1} min")
    print("=" * 60)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e6
        print(f"  VRAM: {total_vram:.0f} MB total")
    print()

    # Use non-daemon Process objects so each experiment can spawn its own
    # inner Pool for game generation (daemon processes cannot have children).
    ctx     = multiprocessing.get_context("spawn")
    summary = []

    for n_parallel in levels:
        n_workers = max(1, TOTAL_CPU_WORKERS // n_parallel)
        print(f"--- N_PARALLEL={n_parallel}  ({n_workers} CPU workers each) ---")

        jobs = [
            {"params": BENCH_PARAMS, "n_workers": n_workers, "budget_seconds": budget}
            for _ in range(n_parallel)
        ]

        t_round  = time.time()
        queue    = ctx.Queue()
        procs    = [
            ctx.Process(target=_bench_worker_proc, args=(j, queue))
            for j in jobs
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        results = [queue.get() for _ in procs]

        round_time   = time.time() - t_round
        total_ep     = sum(r["episodes"]    for r in results)
        total_ep_s   = sum(r["ep_per_sec"]  for r in results)
        max_vram     = max(r["peak_vram_mb"] for r in results)
        avg_ep       = total_ep / n_parallel

        print(f"  Episodes per trial : {avg_ep:.0f}")
        print(f"  Total ep/s         : {total_ep_s:.1f}  (all {n_parallel} trials combined)")
        print(f"  Peak VRAM / trial  : {max_vram:.1f} MB")
        print(f"  Round wall time    : {round_time:.0f}s")
        print()

        summary.append({
            "n_parallel":    n_parallel,
            "n_workers":     n_workers,
            "total_ep_s":    total_ep_s,
            "avg_ep":        avg_ep,
            "max_vram_mb":   max_vram,
        })

    # Report
    best = max(summary, key=lambda s: s["total_ep_s"])
    baseline = summary[0]["total_ep_s"]

    print("=" * 60)
    print("RESULTS SUMMARY")
    print(f"{'N_PAR':>6}  {'workers':>7}  {'ep/s':>8}  {'speedup':>8}  {'VRAM/trial':>10}")
    for s in summary:
        speedup = s["total_ep_s"] / baseline if baseline > 0 else 1.0
        marker  = "  <-- BEST" if s["n_parallel"] == best["n_parallel"] else ""
        print(f"{s['n_parallel']:>6}  {s['n_workers']:>7}  "
              f"{s['total_ep_s']:>8.1f}  {speedup:>8.2f}x  "
              f"{s['max_vram_mb']:>10.1f} MB{marker}")

    print()
    print(f"Recommendation: --parallel {best['n_parallel']}")
    print(f"  {best['total_ep_s']:.1f} ep/s total  |  "
          f"{best['n_workers']} CPU workers per experiment  |  "
          f"{best['max_vram_mb']:.0f} MB VRAM per experiment")
    print("=" * 60)


if __name__ == "__main__":
    main()
