#!/bin/bash
# Launch autoresearch hyperparameter search inside the Docker container.
#
# Lightning.ai free plan constraints:
#   - 32 CPU cores  -> N_WORKERS capped at 32 in GRID
#   - Single GPU    -> 1 experiment runs at a time
#   - 4-hour Studio restarts -> run inside Docker (survives restarts if
#     container is kept alive via nohup / tmux)
#   - 50 GB persistent storage -> results written to mounted ./data volume
#
# Usage (from repo root, on Lightning.ai Studio):
#   bash autoresearch/run.sh                      # random search, runs forever
#   bash autoresearch/run.sh --strategy grid      # exhaustive grid search
#   bash autoresearch/run.sh --max-experiments 40 # stop after N experiments
#
# To run in the background and survive a terminal disconnect:
#   nohup bash autoresearch/run.sh > data/autoresearch_logs/nohup.out 2>&1 &
#   echo "PID: $!"

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Sanity check: environment
echo "=== autoresearch ==="
echo "  repo  : $REPO_ROOT"
echo "  python: $(python --version)"
python -c "import torch; print('  torch :', torch.__version__); print('  cuda  :', torch.cuda.is_available())"

# One-time environment check
python autoresearch/prepare.py

# Forward any extra args (--strategy, --max-experiments) to agent.py
python autoresearch/agent.py "$@"
