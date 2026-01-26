#!/bin/bash
# run_pissa_cs.sh - PiSSA Commonsense Reasoning 실험

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

GPUS="0,1,2,3"
PER_GPU=1
SEEDS="42"
TASKS="arc_easy"
MODEL="meta-llama/Llama-2-7b-hf"

# LAVA Parameters
R=8
ALPHA=8
LAMBDA_VIB=1.0
LAMBDA_LATENT=1.0

# 스크립트 위치 기반 경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_cs_parallel.sh" \
    pissa \
    "$GPUS" \
    $PER_GPU \
    "$TASKS" \
    "$SEEDS" \
    "$MODEL" \
    $LAMBDA_VIB \
    $LAMBDA_LATENT \
    $ALPHA \
    $R
EOF

chmod +x scripts/llama2_7B_base/run_cs_pissa.sh