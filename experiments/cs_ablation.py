#!/usr/bin/env python
"""
Commonsense Reasoning Ablation Experiment
==========================================
Llama-2-7B에서 LAVA 하이퍼파라미터 민감도 분석
(Image Ablation Runner 구조와 통일)
"""

import sys
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
    COMMONSENSE_TASKS,
    COMMONSENSE_ABLATION_CSV_COLUMNS,
    ABLATION_GRID
)


class CommonsenseAblationRunner(BaseExperimentRunner):
    """Commonsense Reasoning LAVA Ablation (병렬 GPU 지원)"""

    def __init__(self, seeds=None, gpus="0", per_gpu_tasks=1, test_mode=False,
                 tasks=None, output_dir=None,
                 training_config=None, lora_config=None,
                 use_wandb=True, wandb_project=None,
                 model="meta-llama/Llama-2-7b-hf"):
        super().__init__(
            experiment_name="commonsense_ablation",
            seeds=seeds,
            gpus=gpus,
            per_gpu_tasks=per_gpu_tasks,
            test_mode=test_mode,
            output_dir=output_dir,
            training_config=training_config,
            lora_config=lora_config,
            use_wandb=use_wandb,
            wandb_project=wandb_project or "Llama2-Ablation",
        )
        self._tasks = tasks if tasks else COMMONSENSE_TASKS
        self._model = model

    @property
    def csv_columns(self):
        return COMMONSENSE_ABLATION_CSV_COLUMNS

    @property
    def tasks(self):
        return self._tasks

    def run_single_experiment(self, task, seed, vib, latent_stab, gpu_id=None):
        tc = self.training_config
        lc = self.lora_config

        cmd = [
            "python", "train_CS.py",
            "--adapter", "lava",
            "--task", task,
            "--seed", str(seed),
            "--model", self._model,
            "--learning_rate", str(tc.learning_rate),
            "--batch", str(tc.batch_size),
            "--epochs", str(tc.epochs),
            "--weight_decay", str(tc.weight_decay),
            "--warmup_ratio", str(tc.warmup_ratio),
            "--r", str(lc.r),
            "--alpha", str(lc.alpha),
            "--lambda_vib", str(vib),
            "--lambda_latent_stability", str(latent_stab),
            "--wandb_project", self.wandb_project,
        ]

        if not self.use_wandb:
            cmd.append("--no_wandb")

        job_name = f"lava_{task}_s{seed}_vib{vib}_lat{latent_stab}"

        if self.test_mode:
            dummy = self.get_dummy_result()
            time.sleep(0.3)
            self.update_progress(job_name)
            return dummy

        use_gpu = gpu_id if gpu_id else self.gpus
        ret_code, _, _ = self.run_subprocess_with_gpu(cmd, use_gpu, job_name)

        if ret_code != 0:
            return 0.0

        result_file = (
            self.result_dir /
            f"commonsense_result_{task}_s{seed}_vib{vib}_lat{latent_stab}.json"
        )

        if result_file.exists():
            with open(result_file) as f:
                score = json.load(f).get("best_accuracy", 0.0)
                self.update_progress(f"{job_name} = {score:.4f}")
                return score

        return 0.0

    def _job_executor(self, gpu_id, task, seed, vib, latent_stab):
        return {
            "task": task,
            "seed": seed,
            "vib": vib,
            "latent_stab": latent_stab,
            "score": self.run_single_experiment(
                task, seed, vib, latent_stab, gpu_id
            )
        }

    def run_ablation_for_param(self, param_type):
        grid = ABLATION_GRID[param_type]
        values, fixed = grid["values"], grid["fixed"]

        self.log(f"\n{'='*60}")
        self.log(f" {param_type.upper()} Ablation")
        self.log(f" values={values}, fixed={fixed}")
        self.log(f"{'='*60}")

        jobs = []
        for val in values:
            vib, latent_stab = (
                (val, fixed["latent_stab"])
                if param_type == "vib"
                else (fixed["vib"], val)
            )

            for seed in self.seeds:
                for task in self._tasks:
                    jobs.append({
                        "task": task,
                        "seed": seed,
                        "vib": vib,
                        "latent_stab": latent_stab
                    })

        results = self.execute_parallel_jobs(jobs, self._job_executor)

        grouped = defaultdict(lambda: defaultdict(dict))
        for r in results:
            key = (r["vib"], r["latent_stab"], r["seed"])
            grouped[key][r["task"]] = r["score"]

        for val in values:
            vib, latent_stab = (
                (val, fixed["latent_stab"])
                if param_type == "vib"
                else (fixed["vib"], val)
            )

            for seed in self.seeds:
                key = (vib, latent_stab, seed)
                task_scores = grouped[key]
                avg = self.calculate_average(task_scores)

                row = {
                    "seed": seed,
                    "vib": vib,
                    "latent_stab": latent_stab,
                    "avg": f"{avg*100:.2f}"
                }

                for t in COMMONSENSE_TASKS:
                    row[t] = f"{task_scores.get(t, 0.0)*100:.2f}"

                self.append_result(row)

    def run_all_experiments(self):
        self.save_metadata({
            "ablation_params": ["vib", "latent_stab"],
            "model": self._model
        })
        self.init_csv()

        for p in ["vib", "latent_stab"]:
            self.run_ablation_for_param(p)

        self.log(f"\n{'='*60}")
        self.log(" Commonsense Ablation 완료")
        self.log(f" 결과 CSV: {self.csv_path}")
        self.log(f"{'='*60}")
