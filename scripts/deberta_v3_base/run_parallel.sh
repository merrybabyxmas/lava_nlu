#!/bin/bash
# run_parallel.sh

ADAPTER=$1
GPUS=$2
PER_GPU=$3
TASKS=$4
SEEDS=$5
LAMBDA_VIBS=$6    # 예: "0.0 0.1"
LAMBDA_STABS=$7   # 예: "0.0 1.0" (Logit Stability)
LATENT_STABS=$8   # 예: "0.0 1.0 2.0" (Latent Stability)
ALPHA=$9

GPU_ARR=(${GPUS//,/ })
i=0

# 하이퍼파라미터 조합을 위한 5중 루프
for task in $TASKS; do
  for seed in $SEEDS; do
    for vib in $LAMBDA_VIBS; do
      for l_stab in $LAMBDA_STABS; do
        for lat_stab in $LATENT_STABS; do
          
          # 현재 조합에 할당할 GPU 결정
          gpu_id=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}

          # LAVA 전용 인자 및 신규 Latent Stability 인자 구성
          EXTRA_ARGS="--alpha $ALPHA"
          if [[ "$ADAPTER" == *"lava"* ]]; then
              EXTRA_ARGS="$EXTRA_ARGS --lambda_vib $vib --lambda_stab $l_stab --lambda_latent_stab $lat_stab"
          fi

          echo "[실험 시작] Task: $task | Seed: $seed | VIB: $vib | LogitStab: $l_stab | LatentStab: $lat_stab (GPU: $gpu_id)"

          # 백그라운드에서 실험 실행
          CUDA_VISIBLE_DEVICES=$gpu_id \
          python train_nlu.py \
            --adapter $ADAPTER \
            --task $task \
            --seed $seed \
            $EXTRA_ARGS & 

          i=$((i+1))

          # 지정된 병렬 처리 수(PER_GPU * GPU 개수)만큼 차면 대기
          if (( i % (${#GPU_ARR[@]} * PER_GPU) == 0 )); then 
            wait 
          fi
          
        done
      done
    done
  done
done

# 남은 프로세스 완료 대기
wait
echo "모든 그리드 서치 실험이 완료되었습니다."