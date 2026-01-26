#!/usr/bin/env python
"""
Commonsense Reasoning Comparison Experiment
============================================
Llama-2-7B에서 LAVA와 다른 메소드(BitFit, LoRA, AdaLoRA, DoRA, PiSSA) 비교 실험
병렬 GPU 실행 지원
"""

import os
import sys
import subprocess
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.base_runner import (
    BaseExperimentRunner,
    TrainingConfig,
    LoRAConfig,
    LAVAConfig,
    COMMONSENSE_TASKS,
    COMMONSENSE_CSV_COLUMNS,
    COMPARISON_METHODS
)


class CommonsenseComparisonRunner(BaseExperimentRunner):
    """Commonsense Reasoning 태스크에서 메소드 비교 실험 (병렬 GPU 실행 지원)"""

    def __init__(self, seeds=None, gpus="0", per_gpu_tasks=1, test_mode=False,
                 tasks=None, methods=None, output_dir=None,
                 training_config=None, lora_config=None, lava_config=None,
                 use_wandb=True, wandb_project=None, model="meta-llama/Llama-2-7b-hf"):
        super().__init__(
            experiment_name="commonsense_comparison",
            seeds=seeds,
            gpus=gpus,
            per_gpu_tasks=per_gpu_tasks,
            test_mode=test_mode,
            output_dir=output_dir,
            training_config=training_config,
            lora_config=lora_config,
            lava_config=lava_config,
            use_wandb=use_wandb,
            wandb_project=wandb_project or "Llama2-CommonsenseReasoning",
        )
        self._tasks = tasks if tasks else COMMONSENSE_TASKS
        self._methods = methods if methods else COMPARISON_METHODS
        self._model = model

    @property
    def csv_columns(self):
        return COMMONSENSE_CSV_COLUMNS

    @property
    def tasks(self):
        return self._tasks

    def run_single_experiment(self, method: str, task: str, seed: int,
                               gpu_id: str = None) -> dict:
        """단일 실험 실행 (GPU ID 지정 가능)

        Returns:
            dict: {"score": float, "oom": bool, "method": str, "task": str, "seed": int}
        """
        cmd = [
            "python", "train_CS.py",
            "--adapter", method,
            "--task", task,
            "--seed", str(seed),
            "--model", self._model,
        ] + self.build_training_args(method)

        job_name = f"{method}_{task}_s{seed}"
        result = {"method": method, "task": task, "seed": seed, "score": 0.0, "oom": False}

        if self.test_mode:
            result["score"] = self.get_dummy_result()
            time.sleep(0.5)
            self.update_progress(job_name)
            return result

        use_gpu = gpu_id if gpu_id else self.gpus
        ret_code, stdout, stderr = self.run_subprocess_with_gpu(cmd, use_gpu, job_name)

        # OOM 감지 (SIGKILL = -9)
        if ret_code == -9:
            result["oom"] = True
            self.log(f"[OOM] {job_name} - GPU {use_gpu}에서 OOM 발생", "WARN")
            return result

        if ret_code != 0:
            return result

        # 결과 파일 경로
        result_file = self.result_dir / f"commonsense_result_{method}_{task}_r{self.lora_config.r}_s{seed}.json"

        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                result["score"] = data.get("best_accuracy", 0.0)
                self.update_progress(f"{job_name} = {result['score']:.4f}")
        return result

    def _job_executor(self, gpu_id: str, method: str, task: str, seed: int) -> dict:
        """병렬 작업 실행기 (OOM 플래그 포함)"""
        return self.run_single_experiment(method, task, seed, gpu_id)

    def get_params_percentage(self, method: str) -> str:
        """메소드별 파라미터 비율 (Llama-2-7B 기준, r=8)"""
        params_map = {
            "bitfit": "0.08",
            "lora": "0.33",
            "adalora": "0.33",
            "dora": "0.34",
            "pissa": "0.33",
            "lava": "0.33"
        }
        return params_map.get(method, "-")

    def run_all_experiments(self):
        """모든 비교 실험 병렬 실행"""
        self.save_metadata({
            "methods": self._methods,
            "model": self._model
        })
        self.init_csv()

        # 모든 실험 작업 생성
        jobs = []
        for method in self._methods:
            for seed in self.seeds:
                for task in self._tasks:
                    jobs.append({
                        "method": method,
                        "task": task,
                        "seed": seed
                    })

        self.log(f"총 {len(jobs)}개 실험 준비 완료")
        self.log(f"Model: {self._model}")
        self.log(f"Methods: {self._methods}")
        self.log(f"Tasks: {self._tasks}")
        self.log(f"Seeds: {self.seeds}")

        # 병렬 실행
        results = self.execute_parallel_jobs(jobs, self._job_executor)

        # 결과 집계 (method -> task -> [scores])
        method_results = defaultdict(lambda: defaultdict(list))
        for res in results:
            if res and not res.get("oom", False) and res.get("score", 0) > 0:
                method_results[res["method"]][res["task"]].append(res["score"])

        # CSV에 메소드별 결과 기록
        for method in self._methods:
            task_results = method_results[method]

            row = {
                "method": method.upper(),
                "params(%)": self.get_params_percentage(method),
            }

            all_means = []
            for task in self._tasks:
                if task in task_results and task_results[task]:
                    scores = task_results[task]
                    mean = sum(scores) / len(scores)
                    std = self.calculate_std(scores)
                    row[task] = self.format_result(mean, std)
                    all_means.append(mean)
                else:
                    row[task] = ""

            if all_means:
                row["avg"] = f"{sum(all_means)/len(all_means)*100:.2f}"

            self.append_result(row)

        self.log(f"")
        self.log(f"{'='*60}")
        self.log(f" Commonsense Reasoning Comparison 완료!")
        self.log(f" 결과: {self.csv_path}")
        self.log(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Commonsense Reasoning Comparison (병렬 GPU 지원)")
    parser.add_argument("--seeds", type=str, default="1,2,42")
    parser.add_argument("--gpus", type=str, default="0",
                        help="사용할 GPU ID (예: '0,1,2,3')")
    parser.add_argument("--per_gpu_tasks", type=int, default=1,
                        help="GPU당 동시 실행 작업 수")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tasks", type=str, default=None,
                        help="실행할 태스크 (예: 'piqa,siqa')")
    parser.add_argument("--methods", type=str, default=None,
                        help="실행할 메소드 (예: 'lora,lava')")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")

    # wandb 설정
    parser.add_argument("--no_wandb", action="store_true", help="wandb 비활성화")
    parser.add_argument("--wandb_project", type=str, default="Llama2-CommonsenseReasoning")

    # Training Config
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)

    # LoRA Config
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lambda_vib", type=float, default=1.0)
    parser.add_argument("--lambda_latent_stab", type=float, default=1.0)

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = args.tasks.split(",") if args.tasks else None
    methods = args.methods.split(",") if args.methods else None
    use_wandb = not args.no_wandb

    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )

    lora_config = LoRAConfig(
        r=args.r,
        alpha=args.alpha,
        dropout=args.lora_dropout,
    )

    lava_config = LAVAConfig(
        lambda_vib=args.lambda_vib,
        lambda_latent_stability=args.lambda_latent_stab,
    )

    runner = CommonsenseComparisonRunner(
        seeds=seeds,
        gpus=args.gpus,
        per_gpu_tasks=args.per_gpu_tasks,
        test_mode=args.test,
        tasks=tasks,
        methods=methods,
        output_dir=args.output_dir,
        training_config=training_config,
        lora_config=lora_config,
        lava_config=lava_config,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        model=args.model,
    )

    runner.run_all_experiments()


if __name__ == "__main__":
    main()