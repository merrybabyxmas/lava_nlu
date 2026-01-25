#!/bin/bash
# run_parallel.sh - ViT Image Classification 병렬 실행 스크립트

ADAPTER=$1
GPUS=$2
PER_GPU=$3
TASKS=$4
SEEDS=$5
LAMBDA_VIB=${6:-0.1}
LAMBDA_STAB=${7:-0.1}
LAMBDA_LATENT=${8:-1.0}
ALPHA=${9:-16}
R=${10:-8}

GPU_ARR=(${GPUS//,/ })
i=0

# 스크립트 위치 기반 프로젝트 루트 자동 탐지
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " ViT Image Classification - $ADAPTER"
echo " GPUs: $GPUS | Per GPU: $PER_GPU"
echo " Tasks: $TASKS | Seeds: $SEEDS"
echo "============================================================"

for task in $TASKS; do
  for seed in $SEEDS; do
    gpu_id=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}

    EXTRA_ARGS="--alpha $ALPHA --r $R"
    if [[ "$ADAPTER" == "lava" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --lambda_vib $LAMBDA_VIB --lambda_stab $LAMBDA_STAB --lambda_latent_stability $LAMBDA_LATENT"
    fi

    echo "[START] Task: $task | Seed: $seed | Adapter: $ADAPTER (GPU: $gpu_id)"

    CUDA_VISIBLE_DEVICES=$gpu_id \
    python train_vit.py \
      --adapter $ADAPTER \
      --task $task \
      --seed $seed \
      $EXTRA_ARGS &

    i=$((i+1))

    if (( i % (${#GPU_ARR[@]} * PER_GPU) == 0 )); then
      wait
    fi
  done
done

wait
echo "============================================================"
echo " All experiments completed!"
echo "============================================================"
