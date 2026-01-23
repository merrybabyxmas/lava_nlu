#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1

# ============================================================
# Image Classification Ablation: LAVA Hyperparameter Sensitivity (병렬 GPU 실행)
# ============================================================
# VIB: 0.1, 0.5, 1.0, 2.0
# Logit Stab: 0.01, 0.05, 0.1, 0.5
# Latent Stab: 0.1, 0.5, 1.0, 2.0
# Output: outputs/img_ablation_YYYYMMDD_HHMMSS/
#         ├── results.csv
#         ├── metadata.json
#         └── logs/
# ============================================================

# GPU 설정 (병렬 실행)
GPUS="0"           # 사용할 GPU ID (예: "0,1,2,3")
PER_GPU_TASKS=4      # GPU당 동시 실행 작업 수

# 실험 설정
SEEDS="2"
# TASKS="dtd,eurosat,gtsrb,resisc45,sun397,svhn"
TASKS="dtd"  # 빠른 테스트용 c
PARAM="all"  # vib / logit_stab / latent_stab / all

# Training Parameters
LR=1e-4
BATCH_SIZE=16
EPOCHS=30

# LoRA Parameters
R=8
ALPHA=8

# LAVA Lambda Parameters (기본값 - ablation에서는 param별로 그리드 탐색)
LAMBDA_VIB=0.0
LAMBDA_STAB=0.1
LAMBDA_LATENT_STAB=1.0

# Wandb 설정
WANDB_PROJECT="IMG-Ablation"

TEST_MODE=false

cd /home/dongwoo43/lava

echo "============================================================"
echo " Image Classification Ablation 실험"
echo " GPUs: $GPUS | Per GPU Tasks: $PER_GPU_TASKS"
echo " 최대 동시 실행 작업 수: $(($(echo $GPUS | tr ',' '\n' | wc -l) * PER_GPU_TASKS))"
echo "============================================================"

if [ "$TEST_MODE" = true ]; then
    echo "[테스트 모드]"
    python -u experiments/img_ablation.py \
        --gpus "$GPUS" \
        --per_gpu_tasks $PER_GPU_TASKS \
        --seeds "$SEEDS" \
        --tasks "$TASKS" \
        --param "$PARAM" \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --r $R \
        --alpha $ALPHA \
        --lambda_vib $LAMBDA_VIB \
        --lambda_stab $LAMBDA_STAB \
        --lambda_latent_stab $LAMBDA_LATENT_STAB \
        --wandb_project "$WANDB_PROJECT" \
        --test
else
    echo "[실험 모드]"
    python -u experiments/img_ablation.py \
        --gpus "$GPUS" \
        --per_gpu_tasks $PER_GPU_TASKS \
        --seeds "$SEEDS" \
        --tasks "$TASKS" \
        --param "$PARAM" \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --r $R \
        --alpha $ALPHA \
        --lambda_vib $LAMBDA_VIB \
        --lambda_stab $LAMBDA_STAB \
        --lambda_latent_stab $LAMBDA_LATENT_STAB \
        --wandb_project "$WANDB_PROJECT"
fi

echo "결과는 outputs/ 폴더에 저장됩니다."
