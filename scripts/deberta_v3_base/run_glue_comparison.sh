#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1

# ============================================================
# GLUE Comparison: LAVA vs Other Methods (병렬 GPU 실행)
# ============================================================
# Methods: BitFit, LoRA, AdaLoRA, DoRA, PiSSA, LAVA
# Output: outputs/glue_comparison_YYYYMMDD_HHMMSS/
#         ├── results.csv
#         ├── metadata.json
#         └── logs/
# ============================================================

# GPU 설정 (병렬 실행)
GPUS="2,3"           # 사용할 GPU ID (예: "0,1,2,3")
PER_GPU_TASKS=3      # GPU당 동시 실행 작업 수 (RAM 메모리 절약을 위해 1로 설정)

# 실험 설정
# SEEDS="1,2,42"
SEEDS="1,2,42"

TASKS="rte,mrpc,cola,stsb,sst2,qnli,qqp,mnli"
# TASKS="mrpc"  # 빠른 테스트용
METHODS="bitfit,lora,adalora,dora,pissa,lava"
# METHODS="bitfit"

# Training Parameters
LR=1e-4
BATCH_SIZE=32
EPOCHS=15
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1

# LoRA Parameters
R=8
ALPHA=8
LORA_DROPOUT=0.1

LAMBDA_VIB=0.0
LAMBDA_LATENT_STAB=0.0

# Wandb 설정
WANDB_PROJECT="GLUE-all-comparison"

TEST_MODE=false

# 스크립트 위치 기반 프로젝트 루트 자동 탐지
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " GLUE Comparison 실험"
echo " GPUs: $GPUS | Per GPU Tasks: $PER_GPU_TASKS"
echo " 최대 동시 실행 작업 수: $(($(echo $GPUS | tr ',' '\n' | wc -l) * PER_GPU_TASKS))"
echo "============================================================"

if [ "$TEST_MODE" = true ]; then
    echo "[테스트 모드]"
    python -u experiments/glue_comparison.py \
        --gpus "$GPUS" \
        --per_gpu_tasks $PER_GPU_TASKS \
        --seeds "$SEEDS" \
        --tasks "$TASKS" \
        --methods "$METHODS" \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --warmup_ratio $WARMUP_RATIO \
        --r $R \
        --alpha $ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lambda_vib $LAMBDA_VIB \
        --lambda_latent_stab $LAMBDA_LATENT_STAB \
        --wandb_project "$WANDB_PROJECT" \
        --test
else
    echo "[실험 모드]"
    python -u experiments/glue_comparison.py \
        --gpus "$GPUS" \
        --per_gpu_tasks $PER_GPU_TASKS \
        --seeds "$SEEDS" \
        --tasks "$TASKS" \
        --methods "$METHODS" \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --warmup_ratio $WARMUP_RATIO \
        --r $R \
        --alpha $ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lambda_vib $LAMBDA_VIB \
        --lambda_latent_stab $LAMBDA_LATENT_STAB \
        --wandb_project "$WANDB_PROJECT"
fi

echo "결과는 outputs/ 폴더에 저장됩니다."
