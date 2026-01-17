#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ============================================================
# Image Classification Comparison: LAVA vs Other Methods (병렬 GPU 실행)
# ============================================================
# Datasets: DTD, EuroSAT, GTSRB, RESISC45, SUN397, SVHN
# Methods: BitFit, LoRA, AdaLoRA, DoRA, PiSSA, LAVA
# Output: outputs/img_comparison_YYYYMMDD_HHMMSS/
#         ├── results.csv
#         ├── metadata.json
#         └── logs/
# ============================================================

# GPU 설정 (병렬 실행)
GPUS="0,1,2,3"           # 사용할 GPU ID (예: "0,1,2,3")
PER_GPU_TASKS=1      # GPU당 동시 실행 작업 수

# 실험 설정
SEEDS="1,2,42"
TASKS="dtd,eurosat,gtsrb,resisc45,sun397,svhn"
# TASKS="dtd,eurosat"  # 빠른 테스트용
METHODS="bitfit,lora,adalora,dora,pissa,lava"

# Training Parameters
LR=1e-4
BATCH_SIZE=32
EPOCHS=30
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1

# LoRA Parameters
R=8
ALPHA=16
LORA_DROPOUT=0.1

# Wandb 설정
WANDB_PROJECT="IMG-Comparison"

TEST_MODE=false

cd /home/dongwoo39/PiSSA_NLU

echo "============================================================"
echo " Image Classification Comparison 실험"
echo " GPUs: $GPUS | Per GPU Tasks: $PER_GPU_TASKS"
echo " 최대 동시 실행 작업 수: $(($(echo $GPUS | tr ',' '\n' | wc -l) * PER_GPU_TASKS))"
echo "============================================================"

if [ "$TEST_MODE" = true ]; then
    echo "[테스트 모드]"
    python experiments/img_comparison.py \
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
        --wandb_project "$WANDB_PROJECT" \
        --test
else
    echo "[실험 모드]"
    python experiments/img_comparison.py \
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
        --wandb_project "$WANDB_PROJECT"
fi

echo "결과는 outputs/ 폴더에 저장됩니다."
