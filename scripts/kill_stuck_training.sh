#!/usr/bin/env bash
# Kill lingering imitation-training worker processes that match the provided pattern.
# Usage: ./kill_stuck_training.sh [pattern]
# Default pattern targets the Lightning imitation trainer command line.

set -euo pipefail

PATTERN=${1:-"training.offline.train_pl"}

mapfile -t PIDS < <(
    ps -eo pid=,command= \
    | grep -F "${PATTERN}" \
    | grep -v "grep" \
    | awk '{print $1}'
)

if ((${#PIDS[@]} == 0)); then
    echo "No processes found matching pattern: ${PATTERN}"
    exit 0
fi

echo "Found ${#PIDS[@]} process(es) matching pattern '${PATTERN}':"
ps -p "${PIDS[*]}" -o pid,etime,command

echo "Sending SIGTERM..."
for pid in "${PIDS[@]}"; do
    if kill -TERM "${pid}" 2>/dev/null; then
        echo "  Sent SIGTERM to PID ${pid}"
    else
        echo "  Failed to send SIGTERM to PID ${pid}" >&2
    fi
done

sleep 3

for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
        echo "PID ${pid} still alive; sending SIGKILL."
        kill -KILL "${pid}" 2>/dev/null || true
    fi
done

echo "Done."
