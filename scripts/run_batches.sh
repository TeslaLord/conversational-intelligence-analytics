#!/usr/bin/env bash
# Runs the pipeline 100 times sequentially, each batch of 100 conversations.
# Stops early if a run fails (non-zero exit).
set -euo pipefail

RUNS="${1:-100}"
BATCH_SIZE="${2:-100}"

for i in $(seq 1 "$RUNS"); do
    echo "=== Run $i / $RUNS ==="
    python -m cc_insights.pipeline --batch-size "$BATCH_SIZE"
done

echo "All $RUNS runs completed."
