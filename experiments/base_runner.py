#!/usr/bin/env python
"""
Base Runner for Experiments
===========================
CSV 관리, Timestamp 폴더, Metadata 저장, 병렬 GPU 실행을 위한 공통 유틸리티
"""

import os
import csv
import json
import subprocess
import random
import threading
import queue
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# Hyperparameter Configuration
# ============================================================
@dataclass
class TrainingConfig:
    """General Training Parameters"""
    learning_rate: float = 5e-4
    batch_size: int = 32
    epochs: int = 30
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler: str = "linear"
    max_grad_norm: float = 1.0


@dataclass
class LoRAConfig:
    """LoRA Parameters"""
    r: int = 16
    alpha: int = 16
    dropout: float = 0.1


@dataclass
class LAVAConfig:
    """LAVA Specific Parameters"""
    lambda_vib: float = 1.0
    lambda_stab: float = 0.1
    lambda_latent_stability: float = 1.0
    latent_dim: int = 16
    noise_scale: float = 1.0
    kl_annealing: bool = False


# 기본 설정값
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_LORA_CONFIG = LoRAConfig()
DEFAULT_LAVA_CONFIG = LAVAConfig()


# ============================================================
# GPU Pool for Parallel Execution
# ============================================================
class GPUPool:
    """
    GPU 풀 관리자 - 각 GPU당 per_gpu_tasks만큼의 동시 작업 허용
    예: gpus="0,1,2,3", per_gpu_tasks=3 이면 총 12개 작업 동시 실행
    """

    def __init__(self, gpus: str, per_gpu_tasks: int = 1):
        self.gpu_list = [g.strip() for g in gpus.split(",")]
        self.per_gpu_tasks = per_gpu_tasks

        # 각 GPU별 슬롯 생성 (GPU ID를 여러 번 넣어서 per_gpu_tasks 만큼 사용 가능하게)
        self.gpu_queue = queue.Queue()
        for gpu_id in self.gpu_list:
            for _ in range(per_gpu_tasks):
                self.gpu_queue.put(gpu_id)

        self.total_slots = len(self.gpu_list) * per_gpu_tasks
        self.lock = threading.Lock()

    def acquire(self) -> str:
        """GPU 슬롯 획득 (blocking)"""
        return self.gpu_queue.get()

    def release(self, gpu_id: str):
        """GPU 슬롯 반환"""
        self.gpu_queue.put(gpu_id)

    def get_max_workers(self) -> int:
        """최대 동시 워커 수"""
        return self.total_slots


class BaseExperimentRunner(ABC):
    """실험 실행을 위한 기본 클래스 (병렬 GPU 실행 지원)"""

    def __init__(
        self,
        experiment_name: str,
        seeds: List[int] = None,
        gpus: str = "0",
        per_gpu_tasks: int = 1,
        test_mode: bool = False,
        output_dir: str = None,
        training_config: TrainingConfig = None,
        lora_config: LoRAConfig = None,
        lava_config: LAVAConfig = None,
        use_wandb: bool = True,
        wandb_project: str = None,
    ):
        self.experiment_name = experiment_name
        self.seeds = seeds if seeds else [1, 2, 42]
        self.gpus = gpus
        self.per_gpu_tasks = per_gpu_tasks
        self.test_mode = test_mode
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        # GPU Pool 초기화
        self.gpu_pool = GPUPool(gpus, per_gpu_tasks)

        # 설정
        self.training_config = training_config or DEFAULT_TRAINING_CONFIG
        self.lora_config = lora_config or DEFAULT_LORA_CONFIG
        self.lava_config = lava_config or DEFAULT_LAVA_CONFIG

        # Timestamp 기반 출력 폴더 생성
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output = output_dir or str(Path(__file__).parent.parent / "outputs")
        self.output_dir = Path(base_output) / f"{experiment_name}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 결과 임시 저장용
        self.result_dir = Path(__file__).parent.parent / "results"
        self.result_dir.mkdir(exist_ok=True)

        # CSV 경로
        self.csv_path = self.output_dir / "results.csv"

        # 로그 파일 경로
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # Thread-safe CSV 쓰기를 위한 락
        self.csv_lock = threading.Lock()

        # 진행 상황 추적
        self.completed_count = 0
        self.total_count = 0
        self.progress_lock = threading.Lock()

    @property
    @abstractmethod
    def csv_columns(self) -> List[str]:
        """CSV 컬럼 정의"""
        pass

    @property
    @abstractmethod
    def tasks(self) -> List[str]:
        """태스크 목록"""
        pass

    def save_metadata(self, extra_info: dict = None):
        """실험 메타데이터를 JSON으로 저장"""
        gpu_list = [g.strip() for g in self.gpus.split(",")]
        metadata = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "seeds": self.seeds,
            "gpus": self.gpus,
            "num_gpus": len(gpu_list),
            "per_gpu_tasks": self.per_gpu_tasks,
            "max_parallel_jobs": len(gpu_list) * self.per_gpu_tasks,
            "test_mode": self.test_mode,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "tasks": self.tasks,
            "training_config": asdict(self.training_config),
            "lora_config": asdict(self.lora_config),
            "lava_config": asdict(self.lava_config),
        }
        if extra_info:
            metadata.update(extra_info)

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.log(f"[*] Metadata 저장: {metadata_path}")

    def init_csv(self):
        """CSV 파일 초기화 (헤더만 작성)"""
        with self.csv_lock:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()
        self.log(f"[*] CSV 초기화: {self.csv_path}")

    def append_result(self, row_data: dict):
        """CSV에 결과 행 추가 (thread-safe)"""
        with self.csv_lock:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writerow(row_data)
        method_info = row_data.get('method', row_data.get('vib', 'N/A'))
        self.log(f"[+] CSV 업데이트: {method_info}")

    def get_dummy_result(self) -> float:
        """테스트 모드용 더미 결과"""
        return round(random.uniform(0.6, 0.95), 4)

    def log(self, message: str, level: str = "INFO"):
        """Thread-safe 로깅 (터미널 + 파일)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        print(formatted_msg)

        # 로그 파일에도 저장
        log_file = self.log_dir / "experiment.log"
        with self.csv_lock:  # 파일 쓰기 동기화
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(formatted_msg + "\n")

    def update_progress(self, job_info: str = ""):
        """진행 상황 업데이트"""
        with self.progress_lock:
            self.completed_count += 1
            progress = f"[{self.completed_count}/{self.total_count}]"
        self.log(f"{progress} 완료: {job_info}")

    def build_training_args(self, method: str) -> List[str]:
        """학습 인자 구성"""
        tc = self.training_config
        lc = self.lora_config

        args = [
            "--learning_rate", str(tc.learning_rate),
            "--batch", str(tc.batch_size),
            "--epochs", str(tc.epochs),
            "--weight_decay", str(tc.weight_decay),
            "--warmup_ratio", str(tc.warmup_ratio),
            "--lr_scheduler", tc.lr_scheduler,
            "--max_grad_norm", str(tc.max_grad_norm),
            "--r", str(lc.r),
            "--alpha", str(lc.alpha),
            "--lora_dropout", str(lc.dropout),
        ]

        # LAVA 전용 파라미터
        if method == "lava":
            lv = self.lava_config
            args.extend([
                "--lambda_vib", str(lv.lambda_vib),
                "--lambda_stab", str(lv.lambda_stab),
                "--lambda_latent_stability", str(lv.lambda_latent_stability),
                "--latent_dim", str(lv.latent_dim),
                "--noise_scale", str(lv.noise_scale),
            ])
            if lv.kl_annealing:
                args.append("--kl_annealing")

        return args

    def run_subprocess_with_gpu(self, cmd: List[str], gpu_id: str,
                                 job_name: str = "") -> Tuple[int, str, str]:
        """
        GPU를 할당받아 subprocess 실행
        Returns: (return_code, stdout, stderr)
        """
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_IB_DISABLE"] = "1"

        # 로그 파일 설정
        log_file = self.log_dir / f"{job_name.replace(' ', '_')}.log"

        self.log(f"[GPU {gpu_id}] 시작: {job_name}")
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent.parent),
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 2시간 타임아웃
            )

            elapsed = time.time() - start_time

            # 로그 파일에 출력 저장
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Command ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== GPU: {gpu_id} ===\n")
                f.write(f"=== Elapsed: {elapsed:.1f}s ===\n\n")
                f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
                f.write(f"=== STDERR ===\n{result.stderr}\n")

            if result.returncode == 0:
                self.log(f"[GPU {gpu_id}] 완료: {job_name} ({elapsed:.1f}s)")
            else:
                self.log(f"[GPU {gpu_id}] 실패: {job_name} (code={result.returncode})", "ERROR")

            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            self.log(f"[GPU {gpu_id}] 타임아웃: {job_name}", "ERROR")
            return -1, "", "Timeout"
        except Exception as e:
            self.log(f"[GPU {gpu_id}] 에러: {job_name} - {e}", "ERROR")
            return -1, "", str(e)

    def execute_parallel_jobs(self, jobs: List[Dict[str, Any]],
                               job_executor: callable) -> List[Any]:
        """
        여러 작업을 병렬로 실행

        Args:
            jobs: 각 작업의 파라미터 딕셔너리 리스트
            job_executor: 각 작업을 실행하는 함수 (gpu_id, **job_params) -> result

        Returns:
            각 작업의 결과 리스트
        """
        self.total_count = len(jobs)
        self.completed_count = 0
        results = [None] * len(jobs)

        self.log(f"{'='*60}")
        self.log(f" 병렬 실행 시작: {len(jobs)}개 작업")
        self.log(f" GPUs: {self.gpus} | Per GPU Tasks: {self.per_gpu_tasks}")
        self.log(f" 최대 동시 실행: {self.gpu_pool.get_max_workers()}개")
        self.log(f"{'='*60}")

        def worker(idx: int, job_params: dict):
            gpu_id = self.gpu_pool.acquire()
            try:
                result = job_executor(gpu_id, **job_params)
                return idx, result
            finally:
                self.gpu_pool.release(gpu_id)

        with ThreadPoolExecutor(max_workers=self.gpu_pool.get_max_workers()) as executor:
            futures = {
                executor.submit(worker, idx, job): idx
                for idx, job in enumerate(jobs)
            }

            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = futures[future]
                    self.log(f"작업 {idx} 실패: {e}", "ERROR")
                    results[idx] = None

        self.log(f"{'='*60}")
        self.log(f" 병렬 실행 완료: {self.completed_count}/{self.total_count} 성공")
        self.log(f"{'='*60}")

        return results

    @abstractmethod
    def run_single_experiment(self, **kwargs) -> float:
        """단일 실험 실행"""
        pass

    @abstractmethod
    def run_all_experiments(self):
        """모든 실험 실행"""
        pass

    def calculate_average(self, results: Dict[str, float]) -> float:
        """평균 계산"""
        valid_scores = [s for s in results.values() if s > 0]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    def calculate_std(self, results: List[float]) -> float:
        """표준편차 계산"""
        if len(results) < 2:
            return 0.0
        mean = sum(results) / len(results)
        variance = sum((x - mean) ** 2 for x in results) / (len(results) - 1)
        return variance ** 0.5

    def format_result(self, mean: float, std: float = None) -> str:
        """결과 포맷팅 (mean ± std)"""
        if std is not None and std > 0:
            return f"{mean*100:.2f}±{std*100:.2f}"
        return f"{mean*100:.2f}"


# ============================================================
# GLUE 태스크 설정
# ============================================================
GLUE_TASKS = ["rte", "mrpc", "cola", "stsb", "sst2", "qnli", "qqp", "mnli"]

GLUE_CSV_COLUMNS = [
    "method", "params(%)",
    "rte", "mrpc", "cola", "stsb", "sst2", "qnli", "qqp", "mnli", "avg"
]

GLUE_ABLATION_CSV_COLUMNS = [
    "seed", "vib", "logit stab / latent stab",
    "rte", "mrpc", "cola", "stsb", "sst2", "qnli", "qqp", "mnli", "avg"
]


# ============================================================
# Image Classification 태스크 설정
# ============================================================
IMG_TASKS = ["dtd", "eurosat", "gtsrb", "resisc45", "sun397", "svhn"]

IMG_CSV_COLUMNS = [
    "method", "params(%)",
    "dtd", "eurosat", "gtsrb", "resisc45", "sun397", "svhn", "avg"
]

IMG_ABLATION_CSV_COLUMNS = [
    "seed", "vib", "logit stab / latent stab",
    "dtd", "eurosat", "gtsrb", "resisc45", "sun397", "svhn", "avg"
]


# ============================================================
# 비교 메소드 목록
# ============================================================
COMPARISON_METHODS = ["bitfit", "lora", "adalora", "dora", "pissa", "lava"]

# Ablation 그리드
ABLATION_GRID = {
    "vib": {
        "values": [0.1, 0.5, 1.0, 2.0],
        "fixed": {"logit_stab": 0.1, "latent_stab": 1.0}
    },
    "logit_stab": {
        "values": [0.01, 0.05, 0.1, 0.5],
        "fixed": {"vib": 1.0, "latent_stab": 1.0}
    },
    "latent_stab": {
        "values": [0.1, 0.5, 1.0, 2.0],
        "fixed": {"vib": 1.0, "logit_stab": 0.1}
    }
}
