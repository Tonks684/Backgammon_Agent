#!/bin/bash
# Launch autoresearch as a Lightning.ai background job.
# Survives 4-hour Studio restarts via background execution.
#
# Usage (from repo root):
#   bash autoresearch/run.sh
#
# To run two parallel searches (uses both concurrent GPU slots):
#   bash autoresearch/run.sh &
#   PROGRAM=autoresearch/program_arch.md bash autoresearch/run.sh &

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROGRAM="${PROGRAM:-autoresearch/program.md}"
TRAIN_SCRIPT="autoresearch/train.py"
LOG_DIR="data/autoresearch_logs"
RESULTS_FILE="data/autoresearch_results.jsonl"

mkdir -p "$LOG_DIR"

echo "Starting autoresearch"
echo "  Program : $PROGRAM"
echo "  Script  : $TRAIN_SCRIPT"
echo "  Logs    : $LOG_DIR"
echo "  Results : $RESULTS_FILE"

cd "$REPO_ROOT"

# Install autoresearch if not already present
if ! command -v autoresearch &>/dev/null; then
    pip install autoresearch -q
fi

# Run autoresearch — iterates experiments until interrupted
autoresearch \
    --program "$PROGRAM" \
    --train "$TRAIN_SCRIPT" \
    --results "$RESULTS_FILE" \
    --log-dir "$LOG_DIR"
