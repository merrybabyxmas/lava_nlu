export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

#!/bin/bash
GPUS="0,1,2,3"
PER_GPU=4
SEEDS="19 29 59 69"
TASKS="sst2 cola qnli rte stsb mnli qqp"
# TASKS="mrpc"

# TASKS="mrpc sst2 cola qnli rte stsb mnli qqp"
# TASKS="mrpc"


# AdaLoRA는 LAVA 계열이 아니므로 LAMBDA 값은 기본값(0.1 등)으로 자리만 채워주거나 생략 가능합니다.
# 하지만 run_parallel.sh의 인자 순서를 지키기 위해 명시하는 것이 좋습니다.
LAMBDA_VIB=0.1
LAMBDA_STAB=1.0
ALPHA=8

# 실행 (GPUS 변수를 반드시 따옴표 "$GPUS"로 감싸주세요)
bash /home/dongwoo39/PiSSA/scripts/deberta_v3_base/run_parallel.sh \
    pissa \
    "$GPUS" \
    "$PER_GPU" \
    "$TASKS" \
    "$SEEDS" \
    "$LAMBDA_VIB" \
    "$LAMBDA_STAB" \
    "$ALPHA"