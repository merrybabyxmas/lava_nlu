#!/usr/bin/env python
"""
GLUE Ablation Experiment
========================
LAVA 하이퍼파라미터 민감도 분석 (VIB, Logit Stab, Latent Stab)
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
    GLUE_TASKS,
    GLUE_ABLATION_CSV_COLUMNS,
    ABLATION_GRID
)


class GLUEAblationRunner(BaseExperimentRunner):
    """GLUE 태스크에서 LAVA ablation 실험 (병렬 GPU 실행 지원)"""

    def __init__(self, seeds=None, gpus="0", per_gpu_tasks=1, test_mode=False,
                 tasks=None, output_dir=None,
                 training_config=None, lora_config=None, lava_config=None,
                 use_wandb=True, wandb_project=None):
        super().__init__(
            experiment_name="glue_ablation",
            seeds=seeds,
            gpus=gpus,
            per_gpu_tasks=per_gpu_tasks,
            test_mode=test_mode,
            output_dir=output_dir,
            training_config=training_config,
            lora_config=lora_config,
            lava_config=lava_config,
            use_wandb=use_wandb,
            wandb_project=wandb_project or "GLUE-Ablation",
        )
        self._tasks = tasks if tasks else GLUE_TASKS

    @property
    def csv_columns(self):
        return GLUE_ABLATION_CSV_COLUMNS

    @property
    def tasks(self):
        return self._tasks

    def run_single_experiment(self, task: str, seed: int,
                              vib: float, logit_stab: float, latent_stab: float,
                              gpu_id: str = None) -> float:
        """단일 실험 실행 (GPU ID 지정 가능)"""
        tc = self.training_config
        lc = self.lora_config

        cmd = [
            "python", "train_nlu.py",
            "--adapter", "lava",
            "--task", task,
            "--seed", str(seed),
            "--learning_rate", str(tc.learning_rate),
            "--batch", str(tc.batch_size),
            "--epochs", str(tc.epochs),
            "--weight_decay", str(tc.weight_decay),
            "--warmup_ratio", str(tc.warmup_ratio),
            "--r", str(lc.r),
            "--alpha", str(lc.alpha),
            "--lambda_vib", str(vib),
            "--lambda_stab", str(logit_stab),
            "--lambda_latent_stability", str(latent_stab),
        ]

        job_name = f"lava_{task}_s{seed}_vib{vib}_stab{logit_stab}_lat{latent_stab}"

        if self.test_mode:
            dummy = self.get_dummy_result()
            time.sleep(0.5)
            self.update_progress(job_name)
            return dummy

        use_gpu = gpu_id if gpu_id else self.gpus
        ret_code, stdout, stderr = self.run_subprocess_with_gpu(cmd, use_gpu, job_name)

        if ret_code != 0:
            return 0.0

        result_file = self.result_dir / f"result_{task}_s{seed}_vib{vib}_stab{logit_stab}_lat{latent_stab}.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                score = data.get("best_metric", 0.0)
                self.update_progress(f"{job_name} = {score:.4f}")
                return score
        return 0.0

    def _job_executor(self, gpu_id: str, task: str, seed: int,
                      vib: float, logit_stab: float, latent_stab: float) -> dict:
        """병렬 작업 실행기"""
        score = self.run_single_experiment(task, seed, vib, logit_stab, latent_stab, gpu_id)
        return {
            "task": task, "seed": seed, "score": score,
            "vib": vib, "logit_stab": logit_stab, "latent_stab": latent_stab
        }

    def run_ablation_for_param(self, param_type: str):
        """특정 파라미터에 대한 ablation 실험 (병렬 실행)"""
        grid = ABLATION_GRID[param_type]
        values = grid["values"]
        fixed = grid["fixed"]

        self.log(f"")
        self.log(f"{'='*60}")
        self.log(f" {param_type.upper()} Ablation 시작")
        self.log(f" 테스트 값: {values}")
        self.log(f" 고정값: {fixed}")
        self.log(f"{'='*60}")

        # 모든 작업 생성
        jobs = []
        for val in values:
            if param_type == "vib":
                vib, logit_stab, latent_stab = val, fixed["logit_stab"], fixed["latent_stab"]
            elif param_type == "logit_stab":
                vib, logit_stab, latent_stab = fixed["vib"], val, fixed["latent_stab"]
            else:
                vib, logit_stab, latent_stab = fixed["vib"], fixed["logit_stab"], val

            for seed in self.seeds:
                for task in self._tasks:
                    jobs.append({
                        "task": task,
                        "seed": seed,
                        "vib": vib,
                        "logit_stab": logit_stab,
                        "latent_stab": latent_stab
                    })

        # 병렬 실행
        results = self.execute_parallel_jobs(jobs, self._job_executor)

        # 결과 집계 (vib, logit_stab, latent_stab, seed -> task -> score)
        config_results = defaultdict(lambda: defaultdict(dict))
        for res in results:
            if res:
                key = (res["vib"], res["logit_stab"], res["latent_stab"], res["seed"])
                config_results[key][res["task"]] = res["score"]

        # CSV에 기록
        for val in values:
            if param_type == "vib":
                vib, logit_stab, latent_stab = val, fixed["logit_stab"], fixed["latent_stab"]
            elif param_type == "logit_stab":
                vib, logit_stab, latent_stab = fixed["vib"], val, fixed["latent_stab"]
            else:
                vib, logit_stab, latent_stab = fixed["vib"], fixed["logit_stab"], val

            for seed in self.seeds:
                key = (vib, logit_stab, latent_stab, seed)
                task_results = config_results[key]

                avg = self.calculate_average(task_results)

                row = {
                    "seed": seed,
                    "vib": vib,
                    "logit stab / latent stab": f"{logit_stab} / {latent_stab}",
                    "avg": f"{avg*100:.2f}"
                }

                for task in GLUE_TASKS:
                    row[task] = f"{task_results.get(task, 0.0)*100:.2f}" if task in task_results else ""

                self.append_result(row)

    def run_all_experiments(self, param_types=None):
        """모든 ablation 실험 실행"""
        if param_types is None:
            param_types = ["vib", "logit_stab", "latent_stab"]

        self.save_metadata({"ablation_params": param_types})
        self.init_csv()

        for param_type in param_types:
            if param_type in ABLATION_GRID:
                self.run_ablation_for_param(param_type)

        self.log(f"")
        self.log(f"{'='*60}")
        self.log(f" GLUE Ablation 완료!")
        self.log(f" 결과: {self.csv_path}")
        self.log(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="GLUE Ablation Experiment (병렬 GPU 지원)")
    parser.add_argument("--seeds", type=str, default="1,2,42")
    parser.add_argument("--gpus", type=str, default="0",
                        help="사용할 GPU ID (예: '0,1,2,3')")
    parser.add_argument("--per_gpu_tasks", type=int, default=1,
                        help="GPU당 동시 실행 작업 수")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--param", type=str, choices=["vib", "logit_stab", "latent_stab", "all"],
                        default="all")
    parser.add_argument("--output_dir", type=str, default=None)

    # wandb 설정
    parser.add_argument("--no_wandb", action="store_true", help="wandb 비활성화")
    parser.add_argument("--wandb_project", type=str, default="GLUE-Ablation")

    # Training Config
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)

    # LoRA Config
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = args.tasks.split(",") if args.tasks else None
    use_wandb = not args.no_wandb

    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    lora_config = LoRAConfig(r=args.r, alpha=args.alpha)

    runner = GLUEAblationRunner(
        seeds=seeds,
        gpus=args.gpus,
        per_gpu_tasks=args.per_gpu_tasks,
        test_mode=args.test,
        tasks=tasks,
        output_dir=args.output_dir,
        training_config=training_config,
        lora_config=lora_config,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
    )

    if args.param == "all":
        runner.run_all_experiments()
    else:
        runner.save_metadata({"ablation_params": [args.param]})
        runner.init_csv()
        runner.run_ablation_for_param(args.param)


if __name__ == "__main__":
    main()
