#!/usr/bin/env bash
# TinyStories LR / batch sweeps, architecture ablations, and the OpenWebText run.
#
# Usage (from assignment1-basics):
#   bash experiments/run_experiments.sh              # all stages
#   bash experiments/run_experiments.sh lr
#   bash experiments/run_experiments.sh batch
#   bash experiments/run_experiments.sh ablation
#   bash experiments/run_experiments.sh owt
#
# Skip a run if its metrics JSONL already exists. Override device with DEVICE=.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-cuda:0}"
PYTHON=(uv run python experiments/train.py)
TS_TRAIN="data/tokenized/tinystories_train.npy"
TS_VALID="data/tokenized/tinystories_valid.npy"
OWT_TRAIN="data/tokenized/owt_train.npy"
OWT_VALID="data/tokenized/owt_valid.npy"

run_train() {
  local out_dir="$1"
  shift
  local log_path="${out_dir}/ckpt.jsonl"
  if [[ -f "${log_path}" ]]; then
    echo "skip ${out_dir}: ${log_path} exists"
    return 0
  fi
  mkdir -p "${out_dir}"
  echo "=== ${out_dir} ==="
  "${PYTHON[@]}" \
    --device "${DEVICE}" \
    --checkpoint_path "${out_dir}/ckpt.pt" \
    --log_path "${log_path}" \
    "$@"
}

run_lr() {
  # Pairs: peak LR, min LR (0.1x). 1e-3 is a fresh copy of the baseline dir layout.
  local peaks=(1e-4 3e-4 1e-3 3e-3 1e-1)
  local mins=(1e-5 3e-5 1e-4 3e-4 1e-2)
  local i
  for i in "${!peaks[@]}"; do
    local lr="${peaks[$i]}"
    local lr_min="${mins[$i]}"
    run_train "experiments/artifacts/sweeps/lr_${lr}" \
      --train_path "${TS_TRAIN}" \
      --valid_path "${TS_VALID}" \
      --vocab_size 10000 \
      --learning_rate "${lr}" \
      --lr_min "${lr_min}"
  done
}

run_batch() {
  # Match the baseline token budget: 32 * 256 * 40000.
  # total_iters = 40000 * 32 / batch_size; warmup stays 1% of steps.
  local sizes=(16 64 128)
  local bs
  for bs in "${sizes[@]}"; do
    local iters=$((40000 * 32 / bs))
    local warmup=$((iters / 100))
    run_train "experiments/artifacts/sweeps/batch_${bs}" \
      --train_path "${TS_TRAIN}" \
      --valid_path "${TS_VALID}" \
      --vocab_size 10000 \
      --batch_size "${bs}" \
      --total_iters "${iters}" \
      --warmup_iters "${warmup}" \
      --learning_rate 1e-3 \
      --lr_min 1e-4
  done
}

run_ablation() {
  # Same TinyStories recipe as the baseline. silu d_ff is set inside TransformerLM.
  # no_rms first uses lr=1e-3; if it diverges, delete that ckpt.jsonl and rerun with a lower lr.
  local names=(nope silu post_norm no_rms)
  local name
  for name in "${names[@]}"; do
    run_train "experiments/artifacts/ablations/${name}" \
      --train_path "${TS_TRAIN}" \
      --valid_path "${TS_VALID}" \
      --vocab_size 10000 \
      --learning_rate 1e-3 \
      --lr_min 1e-4 \
      --ablation "${name}"
  done
}

run_owt() {
  run_train "experiments/artifacts/owt_lm" \
    --train_path "${OWT_TRAIN}" \
    --valid_path "${OWT_VALID}" \
    --vocab_size 32000 \
    --learning_rate 1e-3 \
    --lr_min 1e-4
}

stage="${1:-all}"
case "${stage}" in
  all)
    run_lr
    run_batch
    run_ablation
    run_owt
    ;;
  lr) run_lr ;;
  batch) run_batch ;;
  ablation) run_ablation ;;
  owt) run_owt ;;
  *)
    echo "usage: $0 [all|lr|batch|ablation|owt]" >&2
    exit 1
    ;;
esac

echo "done: ${stage}"
