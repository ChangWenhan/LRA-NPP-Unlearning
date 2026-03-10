#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASETS=("mnist" "cifar10" "cifar100" "imagenet" "mufac")
LOG_FILE="${SCRIPT_DIR}/batch_all_$(date +%Y%m%d_%H%M%S).log"

cd "${REPO_ROOT}"

echo "[batch] repo=${REPO_ROOT}" | tee -a "${LOG_FILE}"
echo "[batch] log=${LOG_FILE}" | tee -a "${LOG_FILE}"

for d in "${DATASETS[@]}"; do
  echo "[batch] >>> dataset=${d} start $(date '+%F %T')" | tee -a "${LOG_FILE}"
  python examples/batch_experiments/run_batch.py --dataset "${d}" 2>&1 | tee -a "${LOG_FILE}"
  echo "[batch] <<< dataset=${d} done  $(date '+%F %T')" | tee -a "${LOG_FILE}"
done

echo "[batch] all done $(date '+%F %T')" | tee -a "${LOG_FILE}"
