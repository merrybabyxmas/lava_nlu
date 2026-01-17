#!/bin/bash
GPUS="0,1"
PER_GPU=4
SEEDS="59 69"
TASKS="mrpc sst2 cola qnli rte stsb mnli qqp"

# AdaLoRA 실행
bash run_parallel.sh adalora $GPUS $PER_GPU "$TASKS" "$SEEDS"

