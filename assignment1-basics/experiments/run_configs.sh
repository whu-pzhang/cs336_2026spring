#!/usr/bin/env bash
# Run every JSON in experiments/configs sequentially.
#
# Usage (from assignment1-basics):
#   bash experiments/run_configs.sh
#
# Writes to experiments/artifacts/iters10k/<config_stem> so existing
# leaderboard artifacts are not overwritten. Skip a run if its metrics
# JSONL already exists. Override with DEVICE= and TOTAL_ITERS=.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-cuda:0}"
TOTAL_ITERS="${TOTAL_ITERS:-10000}"
OUT_ROOT="${OUT_ROOT:-experiments/artifacts/iters10k}"
PYTHON=(uv run python experiments/train.py)

shopt -s nullglob
configs=(experiments/configs/*.json)
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "no configs in experiments/configs" >&2
  exit 1
fi

for cfg in "${configs[@]}"; do
  name="$(basename "${cfg}" .json)"
  out_dir="${OUT_ROOT}/${name}"
  log_path="${out_dir}/ckpt.jsonl"
  if [[ -f "${log_path}" ]]; then
    echo "skip ${out_dir}: ${log_path} exists"
    continue
  fi
  mkdir -p "${out_dir}"
  echo "=== ${name}  total_iters=${TOTAL_ITERS}  ${out_dir} ==="
  "${PYTHON[@]}" \
    --config "${cfg}" \
    --optim.total_iters "${TOTAL_ITERS}" \
    --run.device "${DEVICE}" \
    --run.output_dir "${out_dir}"
done

echo "done: ${#configs[@]} configs, total_iters=${TOTAL_ITERS}"
