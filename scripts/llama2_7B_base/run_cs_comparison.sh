#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1

# ============================================================
# Commonsense Reasoning Comparison: LAVA vs Other Methods (병렬 GPU 실행)
# ============================================================
# Datasets: PIQA, SIQA, ARC-Easy, ARC-Challenge, OpenBookQA, HellaSwag, WinoGrande
# Methods: BitFit, LoRA, AdaLoRA, DoRA, PiSSA, LAVA
# Output: outputs/commonsense_comparison_YYYYMMDD_HHMMSS/
#         ├── results.csv
#         ├── metadata.json
#         └── logs/
# ============================================================

# GPU 설정 (병렬 실행)
GPUS="0,1,2,3"        # 사용할 GPU ID
PER_GPU_TASKS=1       # GPU당 동시 실행 작업 수

# 실험 설정
SEEDS="42"
# TASKS="piqa,siqa,arc_easy,arc_challenge,openbookqa,hellaswag,winogrande"
TASKS="arc_easy,arc_challenge,openbookqa"  # 작은 데이터셋만
# METHODS="bitfit,lora,adalora,dora,pissa,lava"
METHODS="lora,lava"  # 테스트용

# Model
MODEL="meta-llama/Llama-2-7b-hf"

# Training Parameters
LR=3e-4
BATCH_SIZE=4
EPOCHS=5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
GRAD_ACCUM=1

# LoRA Parameters
R=8
ALPHA=8
LORA_DROPOUT=0.1

# LAVA Parameters
LAMBDA_VIB=1.0
LAMBDA_LATENT_STAB=1.0

# Wandb 설정
WANDB_PROJECT="Llama2-CommonsenseReasoning-comparison"

TEST_MODE=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " Commonsense Reasoning Comparison 실험"
echo " Model: $MODEL"
echo " GPUs: $GPUS | Per GPU Tasks: $PER_GPU_TASKS"
echo " 최대 동시 실행 작업 수: $(($(echo $GPUS | tr ',' '\n' | wc -l) * PER_GPU_TASKS))"
echo "============================================================"

if [ "$TEST_MODE" = true ]; then
    echo "[테스트 모드]"
    python -u experiments/cs_comparison.py \
        --gpus "$GPUS" \
        --per_gpu_tasks $PER_GPU_TASKS \
        --seeds "$SEEDS" \
        --tasks "$TASKS" \
        --methods "$METHODS" \
        --model "$MODEL" \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --warmup_ratio $WARMUP_RATIO \
        --grad_accum $GRAD_ACCUM \
        --r $R \
        --alpha $ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lambda_vib $LAMBDA_VIB \
        --lambda_latent_stab $LAMBDA_LATENT_STAB \
        --wandb_project "$WANDB_PROJECT" \
        --test
else
    echo "[실험 모드]"
    python -u experiments/cs_comparison.py \
        --gpus "$GPUS" \
        --per_gpu_tasks $PER_GPU_TASKS \
        --seeds "$SEEDS" \
        --tasks "$TASKS" \
        --methods "$METHODS" \
        --model "$MODEL" \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --warmup_ratio $WARMUP_RATIO \
        --grad_accum $GRAD_ACCUM \
        --r $R \
        --alpha $ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lambda_vib $LAMBDA_VIB \
        --lambda_latent_stab $LAMBDA_LATENT_STAB \
        --wandb_project "$WANDB_PROJECT"
fi

echo "결과는 outputs/ 폴더에 저장됩니다."