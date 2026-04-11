#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${1:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}"
TARGET_EXPERIMENTS="${2:-61}"
BATCH_SIZE="${BATCH_SIZE:-3}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"

cd "${ROOT_DIR}"

if [[ ! -f results_v2.tsv ]]; then
  echo "results_v2.tsv not found in ${ROOT_DIR}" >&2
  exit 1
fi

initial_count="$(tail -n +2 results_v2.tsv | wc -l | tr -d ' ')"
target_total=$((initial_count + TARGET_EXPERIMENTS))

echo "supervisor_start: $(date --iso-8601=seconds)"
echo "initial_count:    ${initial_count}"
echo "target_total:     ${target_total}"
echo "batch_size:       ${BATCH_SIZE}"

while true; do
  current_count="$(tail -n +2 results_v2.tsv | wc -l | tr -d ' ')"
  remaining=$((target_total - current_count))

  if (( remaining <= 0 )); then
    echo "supervisor_done:  $(date --iso-8601=seconds)"
    echo "final_count:      ${current_count}"
    break
  fi

  batch="${BATCH_SIZE}"
  if (( remaining < batch )); then
    batch="${remaining}"
  fi

  echo "batch_start:      $(date --iso-8601=seconds) remaining=${remaining} batch=${batch}"
  if env MAX_EXPERIMENTS="${batch}" bash run_autoresearch.sh "${TAG}"; then
    echo "batch_end:        $(date --iso-8601=seconds) status=ok"
  else
    echo "batch_end:        $(date --iso-8601=seconds) status=retry"
    sleep "${SLEEP_SECONDS}"
  fi
done
