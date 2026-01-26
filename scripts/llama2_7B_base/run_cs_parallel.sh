#!/bin/bash
# run_parallel_cs.sh - Commonsense Reasoning 병렬 실행 스크립트

ADAPTER=$1
GPUS=$2
PER_GPU=$3
TASKS=$4
SEEDS=$5
MODEL=${6:-"meta-llama/Llama-2-7b-hf"}
LAMBDA_VIB=${7:-1.0}
LAMBDA_STAB=${8:-0.1}
LAMBDA_LATENT=${9:-1.0}
ALPHA=${10:-8}
R=${11:-8}

GPU_ARR=(${GPUS//,/ })
i=0

# 스크립트 위치 기반 프로젝트 루트 자동 탐지
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " Commonsense Reasoning - $ADAPTER"
echo " Model: $MODEL"
echo " GPUs: $GPUS | Per GPU: $PER_GPU"
echo " Tasks: $TASKS | Seeds: $SEEDS"
echo "============================================================"

for task in $TASKS; do
  for seed in $SEEDS; do
    gpu_id=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}

    EXTRA_ARGS="--alpha $ALPHA --r $R --batch 1 --grad_accum 8"
    if [[ "$ADAPTER" == "lava" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --lambda_vib $LAMBDA_VIB --lambda_stab $LAMBDA_STAB --lambda_latent_stability $LAMBDA_LATENT"
    fi

    echo "[START] Task: $task | Seed: $seed | Adapter: $ADAPTER (GPU: $gpu_id)"

    CUDA_VISIBLE_DEVICES=$gpu_id \
    python train_CS.py \
      --adapter $ADAPTER \
      --task $task \
      --seed $seed \
      --model $MODEL \
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

chmod +x scripts/llama2_7B_base/run_cs_parallel.sh
