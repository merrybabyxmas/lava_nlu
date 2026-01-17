export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


GPUS="0,1"
PER_GPU=2
SEEDS="19 29"
# TASKS="cola stsb sst2 qnli mnli qqp"
TASKS="qqp"



LAMBDA_VIB=0.1
LAMBDA_STAB=1.0
ALPHA=8


# TASKS="mrpc sst2 cola qnli rte stsb mnli qqp"
# TASKS="mrpc"
bash /home/dongwoo39/PiSSA/scripts/deberta_v3_base/run_parallel.sh lora $GPUS $PER_GPU "$TASKS" "$SEEDS" "$LAMBDA_VIB" "$LAMBDA_STAB" "$ALPHA"
