#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1

# ============================================================
# Commonsense Reasoning Ablation: LAVA Hyperparameter Sensitivity
# ============================================================
# Ablation Params:
#  - VIB: defined in ABLATION_GRID
#  - Latent Stability: defined in ABLATION_GRID
#
# Output:
#   outputs/commonsense_ablation_YYYYMMDD_HHMMSS/
#     ├── results.csv
#     ├── metadata.json
#     └── logs/
# ============================================================

# GPU 설정
GPUS="0,1,2,3"
PER_GPU_TASKS=1

# 실험 설정
SEEDS="42"
TASKS="arc_easy,arc_challenge,openbookqa"
PARAM="all"   # vib / latent_stab / all

# Model
MODEL="meta-llama/Llama-2-7b-hf"

# Training Parameters
LR=3e-4
BATCH_SIZE=16
EPOCHS=5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1

# LoRA Parameters
R=8
ALPHA=8

# LAVA Lambda (baseline 값, ablation 시 grid로 override됨)
LAMBDA_VIB=1.0
LAMBDA_LATENT_STAB=1.0

# Wandb
WANDB_PROJECT="Llama-CommonsenseReasoning-Ablation"

TEST_MODE=false

# 프로젝트 루트 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " Commonsense Reasoning Ablation"
echo " Model: $MODEL"
echo " GPUs: $GPUS | Per GPU Tasks: $PER_GPU_TASKS"
echo " Max Parallel Jobs: $(($(echo $GPUS | tr ',' '\n' | wc -l) * PER_GPU_TASKS))"
echo "============================================================"

CMD="python -u experiments/cs_ablation.py \
    --gpus $GPUS \
    --per_gpu_tasks $PER_GPU_TASKS \
    --seeds $SEEDS \
    --tasks $TASKS \
    --param $PARAM \
    --model $MODEL \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --r $R \
    --alpha $ALPHA \
    --lambda_vib $LAMBDA_VIB \
    --lambda_latent_stab $LAMBDA_LATENT_STAB \
    --wandb_project $WANDB_PROJECT \
    --git status"

if [ "$TEST_MODE" = true ]; then
    echo "[TEST MODE]"
    $CMD --test
else
    echo "[RUN MODE]"
    $CMD
fi

echo "결과는 outputs/ 폴더에 저장됩니다."
