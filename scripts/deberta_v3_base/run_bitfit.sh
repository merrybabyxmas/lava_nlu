#!/bin/bash
# NCCL 관련 환경 변수 설정 (P2P 이슈 방지)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 실험 설정
GPUS="0,1,2,3"
PER_GPU=4
SEEDS="19 29 59 69"
# STSB를 포함한 모든 GLUE 태스크
TASKS="mrpc sst2 cola qnli rte stsb mnli qqp"

# 병렬 실행 스크립트 호출
# 'bitfit'을 첫 번째 인자로 전달하여 Python 코드 내에서 apply_bitfit()이 작동하게 함
bash /home/dongwoo39/PiSSA/scripts/deberta_v3_base/run_parallel.sh bitfit $GPUS $PER_GPU "$TASKS" "$SEEDS"