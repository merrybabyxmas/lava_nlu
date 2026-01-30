#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1

# ============================================================
# LAVA Unified Visualization Engine
# ============================================================
# 3가지 시각화:
# 1. Cone vs. Tube (Uncertainty Stability)
# 2. 2D Decision Margin & Uncertainty
# 3. 2D Loss Landscape & Trajectory (GIF/PNG)
#
# 모든 methods가 하나의 그림에서 비교됩니다.
# ============================================================

# GPU 설정
GPU=0

# 실험 설정
SEED=42
TASK="eurosat"                                    # 데이터셋: dtd, eurosat, gtsrb, resisc45, sun397, svhn
METHODS="lava,lava_fullweight,lora"           # 비교할 메소드들 (하나의 그림에서 비교)

# 시각화 파라미터
NUM_SAMPLES=200                               # 시각화에 사용할 샘플 수
NUM_PASSES=50                                 # Stochastic forward pass 횟수
NUM_EPOCHS=20                                 # Loss landscape용 학습 에폭

# LoRA/LAVA 파라미터
R=8
ALPHA=8

# Training Parameters (Loss Landscape용)
BATCH_SIZE=32

# 스크립트 위치 기반 프로젝트 루트 자동 탐지
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " LAVA Unified Visualization Engine"
echo "============================================================"
echo " Task: $TASK"
echo " Methods: $METHODS"
echo " GPU: $GPU"
echo " Samples: $NUM_SAMPLES | Passes: $NUM_PASSES | Epochs: $NUM_EPOCHS"
echo "============================================================"

# 가상환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pissa

# 시각화 실행
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/img_visualize.py \
    --task "$TASK" \
    --methods "$METHODS" \
    --gpu 0 \
    --seed $SEED \
    --output_dir "./outputs" \
    --all \
    --num_samples $NUM_SAMPLES \
    --num_passes $NUM_PASSES \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --r $R \
    --alpha $ALPHA

echo "============================================================"
echo " Visualization Complete!"
echo " Results saved to: outputs/visualization_${TASK}_*/"
echo "============================================================"
