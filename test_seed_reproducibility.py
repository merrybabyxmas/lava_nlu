#!/usr/bin/env python
"""
Seed Reproducibility Test
=========================
같은 시드로 2번 실행하여 결과가 동일한지 확인
"""

import subprocess
import json
import os
import sys

def run_experiment(seed=42, epochs=1, task="gtsrb"):
    """단일 실험 실행"""
    cmd = [
        "python", "train_vit.py",
        "--adapter", "lava",
        "--task", task,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--batch", "32",
        "--r", "8",
        "--alpha", "8",
        "--lambda_vib", "1.0",
        "--lambda_stab", "0.1",
        "--lambda_latent_stability", "1.0",
        "--no_wandb",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    print(f"Running: {' '.join(cmd)}")
    print("-" * 40)

    # 실시간 출력을 위해 Popen 사용
    process = subprocess.Popen(
        cmd,
        cwd="/home/dongwoo43/lava",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 실시간으로 출력 표시
    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)

    process.wait()
    print("-" * 40)

    if process.returncode != 0:
        print(f"Error: Process exited with code {process.returncode}")
        return None

    # 결과 파일 읽기
    result_file = f"/home/dongwoo43/lava/results/img_result_{task}_s{seed}_vib1.0_stab0.1_lat1.0.json"
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            return data.get("best_accuracy")
    return None


def main():
    print("=" * 60)
    print(" Seed Reproducibility Test")
    print("=" * 60)

    seed = 42
    task = "gtsrb"  # 이미 다운로드된 데이터 사용
    epochs = 1  # 빠른 테스트를 위해 1 epoch만

    # 첫 번째 실행
    print("\n[Run 1]")
    acc1 = run_experiment(seed=seed, epochs=epochs, task=task)
    print(f"Accuracy 1: {acc1}")

    # 두 번째 실행
    print("\n[Run 2]")
    acc2 = run_experiment(seed=seed, epochs=epochs, task=task)
    print(f"Accuracy 2: {acc2}")

    # 결과 비교
    print("\n" + "=" * 60)
    if acc1 is not None and acc2 is not None:
        if abs(acc1 - acc2) < 1e-6:
            print(f"✓ SUCCESS: Results are identical! ({acc1:.6f} == {acc2:.6f})")
        else:
            print(f"✗ FAIL: Results differ! ({acc1:.6f} != {acc2:.6f})")
            print(f"  Difference: {abs(acc1 - acc2):.6f}")
    else:
        print("✗ FAIL: Could not get results from one or both runs")
    print("=" * 60)


if __name__ == "__main__":
    main()
