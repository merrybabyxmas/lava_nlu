#!/usr/bin/env python
"""
Unified Visualization Engine for LAVA, LoRA, LAVA-Fullweight (v2)
=================================================================
핵심 수정:
1. Stochastic Inference 활성화 (model.train() 강제 유지)
2. Monte Carlo Sampling (30-50회 Forward)
3. 3개의 개별 GIF 생성

출력 파일:
- 1_uncertainty_stability.gif: Cone vs Tube 애니메이션
- 2_decision_boundary.gif: 결정 경계 안정화 애니메이션
- 3_loss_landscape.gif: 최적화 궤적 애니메이션
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import ssl
import copy

# SSL 인증서 검증 우회
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision import datasets as tv_datasets
from datasets import Dataset
from peft import get_peft_model, LoraConfig
from peft.tuners.lava.config import LavaConfig
from peft.tuners.lava_fullweight.config import LavaFullWeightConfig
from trainer import setup_seed, register_lava

# LAVA 등록
register_lava()

# ============================================================
# 데이터셋 설정
# ============================================================
IMG_TASK_META = {
    "dtd": dict(source="torchvision", tv_class=tv_datasets.DTD, num_labels=47),
    "eurosat": dict(source="torchvision", tv_class=tv_datasets.EuroSAT, num_labels=10),
    "gtsrb": dict(source="torchvision", tv_class=tv_datasets.GTSRB, num_labels=43),
    "resisc45": dict(source="huggingface", dataset_name="timm/resisc45", num_labels=45, split_train="train", split_val="test"),
    "sun397": dict(source="huggingface", dataset_name="tanganke/sun397", num_labels=397, split_train="train", split_val="test"),
    "svhn": dict(source="torchvision", tv_class=tv_datasets.SVHN, num_labels=10),
}

# 시각화용 색상 정의
METHOD_COLORS = {
    "lava": "#2ecc71",           # 초록색 (LAVA - bias only)
    "lora": "#3498db",           # 파랑색 (LoRA)
    "lava_fullweight": "#e74c3c",# 빨강색 (LAVA Fullweight)
}

METHOD_LABELS = {
    "lava": "LAVA (Bias-only)",
    "lora": "LoRA (Deterministic)",
    "lava_fullweight": "LAVA-Fullweight",
}

METHOD_MARKERS = {
    "lava": "o",
    "lora": "s",
    "lava_fullweight": "^",
}


def load_torchvision_dataset(task: str, meta: dict, data_root: str = "./data", seed: int = 42):
    """Torchvision 데이터셋 로드"""
    tv_class = meta["tv_class"]

    if task == "dtd":
        train_ds = tv_class(root=data_root, split="train", download=True)
        val_ds = tv_class(root=data_root, split="test", download=True)
    elif task == "gtsrb":
        train_ds = tv_class(root=data_root, split="train", download=True)
        val_ds = tv_class(root=data_root, split="test", download=True)
    elif task == "svhn":
        train_ds = tv_class(root=data_root, split="train", download=True)
        val_ds = tv_class(root=data_root, split="test", download=True)
    elif task in ["eurosat", "sun397"]:
        full_ds = tv_class(root=data_root, download=True)
        total_len = len(full_ds)
        train_len = int(0.8 * total_len)
        val_len = total_len - train_len
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [train_len, val_len],
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    def convert_to_hf_dataset(tv_dataset):
        images, labels = [], []
        if hasattr(tv_dataset, 'dataset'):
            for idx in tv_dataset.indices:
                img, label = tv_dataset.dataset[idx]
                images.append(img)
                labels.append(label)
        else:
            for i in range(len(tv_dataset)):
                img, label = tv_dataset[i]
                images.append(img)
                labels.append(label)
        return Dataset.from_dict({"image": images, "label": labels})

    return {"train": convert_to_hf_dataset(train_ds), "test": convert_to_hf_dataset(val_ds)}


def load_dataset_for_task(task: str, seed: int = 42):
    """태스크에 맞는 데이터셋 로드"""
    meta = IMG_TASK_META[task]
    if meta.get("source") == "torchvision":
        return load_torchvision_dataset(task, meta, seed=seed), meta
    else:
        if "subset" in meta:
            raw = load_dataset(meta["dataset_name"], meta["subset"])
        else:
            raw = load_dataset(meta["dataset_name"])
        return {meta["split_train"]: raw[meta["split_train"]],
                meta["split_val"]: raw[meta["split_val"]]}, meta


def build_model(adapter_type: str, num_labels: int, r: int = 8, alpha: int = 8):
    """어댑터 타입에 따른 모델 생성"""
    model_name = "google/vit-base-patch16-224"
    base = ViTForImageClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    target_modules = ["query", "key", "value", "dense"]
    at = adapter_type.lower()

    if at == "lora":
        config = LoraConfig(r=r, lora_alpha=alpha, target_modules=target_modules)
    elif at == "lava":
        config = LavaConfig(r=r, alpha=alpha, target_modules=target_modules)
    elif at == "lava_fullweight":
        config = LavaFullWeightConfig(r=r, alpha=alpha, target_modules=target_modules)
    else:
        raise ValueError(f"Unknown adapter: {adapter_type}")

    model = get_peft_model(base, config)
    return model


def force_stochastic_mode(model, enable: bool = True):
    """
    LAVA 레이어의 stochastic 모드 강제 활성화/비활성화
    핵심: model.eval()에서도 노이즈 샘플링이 동작하도록 함
    """
    for module in model.modules():
        # LAVA 어댑터의 training 플래그 강제 설정
        if hasattr(module, '_last_mu') and hasattr(module, '_last_logvar'):
            module.training = enable
        # LoRA는 영향 없음 (결정론적)
    return model


def reset_lava_generators(model, seed: int = None):
    """
    모델 내 모든 LAVA 어댑터의 generator와 global seed를 변경하여
    새로운 노이즈 시퀀스를 강제합니다.

    핵심 전략:
    1. _global_seed를 새 값으로 변경 (클래스 레벨)
    2. generator를 CPU로 리셋하여 device mismatch 강제 발생
    3. 다음 forward에서 LAVA가 자동으로 CUDA generator를 재생성하며 새 _global_seed 사용
    """
    target_seed = seed if seed is not None else torch.randint(0, 2**31, (1,)).item()

    for module in model.modules():
        if hasattr(module, '_rng_generator') and hasattr(module, '_global_seed'):
            # Step 1: 클래스 레벨 global seed 변경 (핵심!)
            module.__class__._global_seed = target_seed

            # Step 2: generator를 CPU로 리셋 → device mismatch 강제
            # 다음 forward에서 _sample_noise가 호출되면:
            # if self._rng_generator.device != mu.device:
            #     self._rng_generator = torch.Generator(device=mu.device)
            #     self._rng_generator.manual_seed(LavaAdapter._global_seed)  ← 새 시드 사용!
            module._rng_generator = torch.Generator()  # CPU generator


def monte_carlo_forward(model, inputs, num_passes: int = 50, temperature: float = 1.0):
    """
    Monte Carlo Sampling으로 Stochastic Forward Pass 수행

    핵심 전략:
    1. model.train() 모드로 LAVA의 샘플링 활성화
    2. 각 pass마다 _global_seed를 다른 값으로 변경
    3. generator를 CPU로 리셋 → device mismatch 유발
    4. forward시 LAVA가 새 _global_seed로 CUDA generator 재생성

    Args:
        model: PEFT 모델
        inputs: 입력 딕셔너리
        num_passes: MC 샘플링 횟수
        temperature: Logit scaling (1.0 = no scaling)

    Returns:
        logits_mean: 평균 로짓 [batch, num_classes]
        logits_var: 로짓 분산 [batch, num_classes]
        probs_var: 확률 분산 [batch, num_classes]
    """
    # 중요: train 모드로 설정하여 LAVA의 샘플링 활성화
    model.train()
    force_stochastic_mode(model, enable=True)

    logits_list = []

    for pass_idx in range(num_passes):
        # 핵심: 각 pass마다 다른 시드 + CPU generator로 device mismatch 유발
        # LAVA의 _sample_noise가 호출되면 새 _global_seed로 CUDA generator 재생성
        reset_lava_generators(model, seed=pass_idx * 1000 + 42)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits / temperature
            logits_list.append(logits.cpu())

    # [num_passes, batch, num_classes]
    logits_stack = torch.stack(logits_list, dim=0)

    # 통계 계산
    logits_mean = logits_stack.mean(dim=0)
    logits_var = logits_stack.var(dim=0)

    # 확률 분산 계산
    probs_stack = F.softmax(logits_stack, dim=-1)
    probs_var = probs_stack.var(dim=0)

    return logits_mean, logits_var, probs_var


def verify_stochastic_sampling(model, inputs, num_tests: int = 5):
    """
    LAVA의 stochastic 샘플링이 정상 작동하는지 검증
    각 pass마다 다른 출력이 나와야 함
    """
    print("\n[DEBUG] Verifying stochastic sampling...")
    model.train()
    force_stochastic_mode(model, enable=True)

    outputs_list = []
    for i in range(num_tests):
        reset_lava_generators(model, seed=i * 12345)
        with torch.no_grad():
            out = model(**inputs)
            outputs_list.append(out.logits.cpu())

    # 출력 간 차이 확인
    stacked = torch.stack(outputs_list, dim=0)
    variance = stacked.var(dim=0).mean().item()

    print(f"[DEBUG] Output variance across {num_tests} passes: {variance:.8f}")
    if variance < 1e-10:
        print("[WARNING] Variance is near zero! Stochastic sampling may not be working.")
        print("[DEBUG] First 3 outputs (should differ):")
        for i in range(min(3, num_tests)):
            print(f"  Pass {i}: {outputs_list[i][0, :5].numpy()}")
    else:
        print("[OK] Stochastic sampling is working correctly!")
        print(f"[DEBUG] Sample outputs (first 5 classes):")
        for i in range(min(3, num_tests)):
            print(f"  Pass {i}: {outputs_list[i][0, :5].numpy()}")

    return variance > 1e-10


# ============================================================
# 시각화 1: Cone vs. Tube (Uncertainty Stability) - GIF
# ============================================================
class ConeVsTubeVisualizer:
    """입력 노름에 따른 출력 분산 변화 시각화 (애니메이션)"""

    def __init__(self, methods: list, task: str, device: torch.device,
                 num_samples: int = 200, num_passes: int = 50):
        self.methods = methods
        self.task = task
        self.device = device
        self.num_samples = num_samples
        self.num_passes = num_passes

    def run(self, processor, raw_dataset, num_labels: int, output_dir: Path,
            train_epochs: int = 3, batch_size: int = 16):
        """
        Cone vs Tube 시각화 실행

        중요: W_o가 0으로 초기화되어 있어, untrained 모델은 variance=0.
        따라서 먼저 모델을 훈련시킨 후 uncertainty를 측정합니다.
        """
        print("\n" + "="*60)
        print(" Visualization 1: Cone vs. Tube (Uncertainty Stability)")
        print(" [Stochastic Mode: ENABLED for LAVA variants]")
        print(" [Models will be trained briefly for meaningful variance]")
        print("="*60)

        train_key = "train" if "train" in raw_dataset else list(raw_dataset.keys())[0]
        test_key = "test" if "test" in raw_dataset else list(raw_dataset.keys())[-1]
        train_ds = raw_dataset[train_key]
        test_ds = raw_dataset[test_key]

        # 학습 데이터 준비
        subset_size = min(500, len(train_ds))
        train_subset = train_ds.select(range(subset_size))

        def preprocess(examples):
            images = [img.convert("RGB") for img in examples["image"]]
            inputs = processor(images, return_tensors="pt")
            return {"pixel_values": inputs["pixel_values"], "labels": examples["label"]}

        train_subset = train_subset.map(preprocess, batched=True, remove_columns=train_subset.column_names)
        train_subset.set_format("torch")
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        sample_indices = list(range(min(self.num_samples, len(test_ds))))

        results = {}

        for method in self.methods:
            print(f"\n[{method.upper()}] Training for {train_epochs} epochs before uncertainty measurement...")
            model = build_model(method, num_labels)
            model = model.to(self.device)

            # 모델 훈련 (W_o가 0이 아니게 만들기 위해)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
            )

            model.train()
            for epoch in range(train_epochs):
                epoch_loss = 0
                num_batches = 0
                for batch in train_loader:
                    pixel_values = batch["pixel_values"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    optimizer.zero_grad()
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1
                    if num_batches >= 30:  # 빠른 훈련을 위해 제한
                        break

                print(f"    Epoch {epoch+1}/{train_epochs}: Loss = {epoch_loss/num_batches:.4f}")

            # 훈련 후 stochastic 검증
            print(f"[{method.upper()}] Now measuring uncertainty with {self.num_passes} MC passes...")
            first_sample = test_ds[0]
            first_img = first_sample["image"].convert("RGB")
            first_inputs = processor(first_img, return_tensors="pt")
            first_inputs = {k: v.to(self.device) for k, v in first_inputs.items()}
            verify_stochastic_sampling(model, first_inputs, num_tests=5)

            input_norms = []
            logit_variances = []
            prob_variances = []

            for idx in tqdm(sample_indices, desc=f"  {method}"):
                sample = test_ds[idx]
                img = sample["image"].convert("RGB")

                inputs = processor(img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 입력 노름 계산
                input_norm = inputs["pixel_values"].norm().item()
                input_norms.append(input_norm)

                # Monte Carlo Forward (핵심!)
                _, logit_var, prob_var = monte_carlo_forward(
                    model, inputs, self.num_passes, temperature=1.0
                )

                # 전체 클래스에 대한 평균 분산
                logit_variances.append(logit_var.mean().item())
                prob_variances.append(prob_var.mean().item())

            results[method] = {
                "input_norms": np.array(input_norms),
                "logit_variances": np.array(logit_variances),
                "prob_variances": np.array(prob_variances)
            }

            # 통계 출력
            print(f"    Logit Var - Mean: {np.mean(logit_variances):.6f}, "
                  f"Std: {np.std(logit_variances):.6f}, "
                  f"Max: {np.max(logit_variances):.6f}")

            del model
            torch.cuda.empty_cache()

        # 시각화 (정적 + 애니메이션)
        self._plot_static(results, output_dir)
        self._plot_animation(results, output_dir)

        return results

    def _plot_static(self, results: dict, output_dir: Path):
        """정적 Cone vs Tube 플롯"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 왼쪽: Logit Variance
        ax1 = axes[0]
        for method in self.methods:
            data = results[method]
            color = METHOD_COLORS.get(method, "#333333")
            label = METHOD_LABELS.get(method, method.upper())
            marker = METHOD_MARKERS.get(method, "o")

            ax1.scatter(data["input_norms"], data["logit_variances"],
                       c=color, alpha=0.6, s=40, label=label, marker=marker)

            # 트렌드 라인
            z = np.polyfit(data["input_norms"], data["logit_variances"], 2)
            p = np.poly1d(z)
            x_line = np.linspace(data["input_norms"].min(), data["input_norms"].max(), 100)
            ax1.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--')

        ax1.set_xlabel("Input Norm ||x||", fontsize=12)
        ax1.set_ylabel("Logit Variance (MC Sampling)", fontsize=12)
        ax1.set_title("Logit Space Uncertainty", fontsize=13)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 오른쪽: Probability Variance
        ax2 = axes[1]
        for method in self.methods:
            data = results[method]
            color = METHOD_COLORS.get(method, "#333333")
            label = METHOD_LABELS.get(method, method.upper())
            marker = METHOD_MARKERS.get(method, "o")

            ax2.scatter(data["input_norms"], data["prob_variances"],
                       c=color, alpha=0.6, s=40, label=label, marker=marker)

            z = np.polyfit(data["input_norms"], data["prob_variances"], 2)
            p = np.poly1d(z)
            x_line = np.linspace(data["input_norms"].min(), data["input_norms"].max(), 100)
            ax2.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--')

        ax2.set_xlabel("Input Norm ||x||", fontsize=12)
        ax2.set_ylabel("Probability Variance (MC Sampling)", fontsize=12)
        ax2.set_title("Probability Space Uncertainty", fontsize=13)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"Cone vs. Tube: Uncertainty Stability ({self.task.upper()})\n"
                    f"LAVA=Tube (flat), Fullweight=Cone (slope), LoRA=Zero (deterministic)",
                    fontsize=14, y=1.02)
        plt.tight_layout()

        save_path = output_dir / f"1_uncertainty_stability_{self.task}.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

    def _plot_animation(self, results: dict, output_dir: Path):
        """애니메이션 GIF 생성 - 점이 하나씩 추가되는 형태"""
        fig, ax = plt.subplots(figsize=(10, 7))

        # 축 범위 설정
        all_norms = np.concatenate([r["input_norms"] for r in results.values()])
        all_vars = np.concatenate([r["logit_variances"] for r in results.values()])

        x_min, x_max = all_norms.min() - 0.5, all_norms.max() + 0.5
        y_min, y_max = -0.001, all_vars.max() * 1.2 + 0.001

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Input Norm ||x||", fontsize=12)
        ax.set_ylabel("Logit Variance (MC Sampling)", fontsize=12)
        ax.grid(True, alpha=0.3)

        scatters = {}
        trend_lines = {}

        for method in self.methods:
            color = METHOD_COLORS.get(method, "#333333")
            label = METHOD_LABELS.get(method, method.upper())
            marker = METHOD_MARKERS.get(method, "o")

            scatters[method] = ax.scatter([], [], c=color, s=50, label=label,
                                          marker=marker, alpha=0.7)
            trend_lines[method], = ax.plot([], [], color=color, linewidth=2.5,
                                           linestyle='--', alpha=0.8)

        ax.legend(loc='upper left')
        title = ax.set_title("", fontsize=13)

        num_frames = min(len(results[self.methods[0]]["input_norms"]), 100)

        def init():
            for method in self.methods:
                scatters[method].set_offsets(np.empty((0, 2)))
                trend_lines[method].set_data([], [])
            title.set_text("")
            return list(scatters.values()) + list(trend_lines.values()) + [title]

        def animate(frame):
            n = frame + 1
            for method in self.methods:
                data = results[method]
                norms = data["input_norms"][:n]
                vars_ = data["logit_variances"][:n]

                offsets = np.column_stack([norms, vars_])
                scatters[method].set_offsets(offsets)

                # 트렌드 라인 업데이트
                if n >= 10:
                    z = np.polyfit(norms, vars_, 2)
                    p = np.poly1d(z)
                    x_line = np.linspace(norms.min(), norms.max(), 50)
                    trend_lines[method].set_data(x_line, p(x_line))

            title.set_text(f"Cone vs. Tube ({self.task.upper()}) - Sample {n}/{num_frames}")
            return list(scatters.values()) + list(trend_lines.values()) + [title]

        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=num_frames, interval=100, blit=True
        )

        save_path = output_dir / f"1_uncertainty_stability_{self.task}.gif"
        anim.save(str(save_path), writer='pillow', fps=10)
        plt.close()
        print(f"  Saved: {save_path}")


# ============================================================
# 시각화 2: 2D Decision Margin & Uncertainty (에폭별 GIF)
# ============================================================
class DecisionMarginVisualizer:
    """결정 경계 주변의 불확실성 '안전 지대' 시각화"""

    def __init__(self, methods: list, task: str, device: torch.device,
                 num_samples: int = 300, num_passes: int = 30, num_epochs: int = 5):
        self.methods = methods
        self.task = task
        self.device = device
        self.num_samples = num_samples
        self.num_passes = num_passes
        self.num_epochs = num_epochs

    def extract_features_and_uncertainty(self, model, processor, dataset, num_samples: int):
        """특징 추출 및 불확실성 계산 (Stochastic 모드)"""
        # 중요: train 모드 유지
        model.train()
        force_stochastic_mode(model, enable=True)

        features = []
        labels = []
        uncertainties = []

        sample_indices = list(range(min(num_samples, len(dataset))))

        # Hook으로 중간 특징 추출
        hook_output = []
        def hook_fn(module, input, output):
            if hasattr(output, 'last_hidden_state'):
                hook_output.append(output.last_hidden_state[:, 0, :].detach().cpu())
            elif isinstance(output, tuple):
                hook_output.append(output[0][:, 0, :].detach().cpu())
            else:
                hook_output.append(output[:, 0, :].detach().cpu())

        handle = model.base_model.model.vit.encoder.register_forward_hook(hook_fn)

        with torch.no_grad():
            for idx in tqdm(sample_indices, desc="  Extracting"):
                hook_output.clear()
                sample = dataset[idx]
                img = sample["image"].convert("RGB")
                inputs = processor(img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 특징 추출 (1회)
                _ = model(**inputs)
                if hook_output:
                    features.append(hook_output[0].squeeze(0))

                # MC Forward로 불확실성 계산
                _, logit_var, _ = monte_carlo_forward(
                    model, inputs, self.num_passes
                )
                uncertainties.append(logit_var.mean().item())
                labels.append(sample["label"])

        handle.remove()

        return (torch.stack(features).numpy(),
                np.array(labels),
                np.array(uncertainties))

    def run(self, processor, raw_dataset, num_labels: int, output_dir: Path):
        """Decision Margin 시각화 - 학습 진행에 따른 변화"""
        print("\n" + "="*60)
        print(" Visualization 2: 2D Decision Margin & Uncertainty")
        print(" [Tracking uncertainty stabilization over epochs]")
        print("="*60)

        train_key = "train" if "train" in raw_dataset else list(raw_dataset.keys())[0]
        test_key = "test" if "test" in raw_dataset else list(raw_dataset.keys())[-1]
        train_ds = raw_dataset[train_key]
        test_ds = raw_dataset[test_key]

        # 데이터 전처리
        def preprocess(examples):
            images = [img.convert("RGB") for img in examples["image"]]
            inputs = processor(images, return_tensors="pt")
            return {"pixel_values": inputs["pixel_values"], "labels": examples["label"]}

        subset_size = min(300, len(train_ds))
        train_subset = train_ds.select(range(subset_size))
        train_subset = train_subset.map(preprocess, batched=True, remove_columns=train_subset.column_names)
        train_subset.set_format("torch")
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=16, shuffle=True)

        # 각 메소드별로 에폭마다 불확실성 기록
        epoch_data = {method: [] for method in self.methods}

        for method in self.methods:
            print(f"\n[{method.upper()}] Training and collecting uncertainty...")
            model = build_model(method, num_labels).to(self.device)

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
            )

            # 초기 상태 (epoch 0)
            features, labels, unc = self.extract_features_and_uncertainty(
                model, processor, test_ds, self.num_samples
            )
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            epoch_data[method].append({
                "features_2d": features_2d.copy(),
                "labels": labels.copy(),
                "uncertainties": unc.copy(),
                "epoch": 0
            })
            print(f"    Epoch 0: Mean Unc = {unc.mean():.6f}, Std = {unc.std():.6f}")

            # 학습 및 에폭별 기록
            for epoch in range(1, self.num_epochs + 1):
                model.train()
                epoch_loss = 0
                num_batches = 0

                for batch in train_loader:
                    pixel_values = batch["pixel_values"].to(self.device)
                    labels_batch = batch["labels"].to(self.device)

                    optimizer.zero_grad()
                    outputs = model(pixel_values=pixel_values, labels=labels_batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1
                    if num_batches >= 20:
                        break

                # 에폭 후 불확실성 측정
                features, labels, unc = self.extract_features_and_uncertainty(
                    model, processor, test_ds, self.num_samples
                )
                features_2d = pca.transform(features)
                epoch_data[method].append({
                    "features_2d": features_2d.copy(),
                    "labels": labels.copy(),
                    "uncertainties": unc.copy(),
                    "epoch": epoch
                })
                print(f"    Epoch {epoch}: Loss = {epoch_loss/num_batches:.4f}, "
                      f"Mean Unc = {unc.mean():.6f}")

            del model
            torch.cuda.empty_cache()

        # 시각화
        self._plot_static(epoch_data, output_dir)
        self._plot_animation(epoch_data, output_dir)

        return epoch_data

    def _plot_static(self, epoch_data: dict, output_dir: Path):
        """마지막 에폭의 정적 플롯"""
        n_methods = len(self.methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
        if n_methods == 1:
            axes = [axes]

        for ax, method in zip(axes, self.methods):
            final = epoch_data[method][-1]
            features_2d = final["features_2d"]
            labels = final["labels"]
            unc = final["uncertainties"]

            color = METHOD_COLORS.get(method, "#333333")
            label_name = METHOD_LABELS.get(method, method.upper())

            # 불확실성을 점 크기와 투명도로 표현
            unc_normalized = (unc - unc.min()) / (unc.max() - unc.min() + 1e-10)
            sizes = 30 + 150 * unc_normalized
            alphas = 0.4 + 0.5 * unc_normalized

            # 레이블별 색상
            unique_labels = np.unique(labels)
            n_colors = min(len(unique_labels), 10)
            cmap = plt.cm.get_cmap('tab10', n_colors)
            label_colors = [cmap(l % n_colors) for l in labels]

            for i in range(len(features_2d)):
                ax.scatter(features_2d[i, 0], features_2d[i, 1],
                          c=[label_colors[i]], s=sizes[i], alpha=alphas[i],
                          edgecolors=color, linewidths=0.5)

            ax.set_xlabel("PCA Component 1", fontsize=10)
            ax.set_ylabel("PCA Component 2", fontsize=10)
            ax.set_title(f"{label_name}\n(size/alpha ∝ uncertainty)", fontsize=11)
            ax.grid(True, alpha=0.3)

            # 통계 표시
            ax.text(0.02, 0.98,
                    f"Mean Unc: {unc.mean():.5f}\nStd Unc: {unc.std():.5f}\n"
                    f"Max Unc: {unc.max():.5f}",
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle(f"Decision Margin & Uncertainty ({self.task.upper()}) - Final Epoch",
                    fontsize=14, y=1.02)
        plt.tight_layout()

        save_path = output_dir / f"2_decision_boundary_{self.task}.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

    def _plot_animation(self, epoch_data: dict, output_dir: Path):
        """에폭 진행에 따른 불확실성 변화 애니메이션"""
        n_methods = len(self.methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
        if n_methods == 1:
            axes = [axes]

        # 축 범위 고정
        for ax, method in zip(axes, self.methods):
            all_features = np.vstack([e["features_2d"] for e in epoch_data[method]])
            ax.set_xlim(all_features[:, 0].min() - 1, all_features[:, 0].max() + 1)
            ax.set_ylim(all_features[:, 1].min() - 1, all_features[:, 1].max() + 1)
            ax.set_xlabel("PCA Component 1", fontsize=10)
            ax.set_ylabel("PCA Component 2", fontsize=10)
            ax.grid(True, alpha=0.3)

        scatters = {method: None for method in self.methods}
        texts = {method: None for method in self.methods}

        num_epochs = len(epoch_data[self.methods[0]])

        def init():
            for ax, method in zip(axes, self.methods):
                ax.clear()
                ax.grid(True, alpha=0.3)
            return []

        def animate(frame):
            artists = []
            for ax, method in zip(axes, self.methods):
                ax.clear()

                data = epoch_data[method][frame]
                features_2d = data["features_2d"]
                labels = data["labels"]
                unc = data["uncertainties"]
                epoch = data["epoch"]

                color = METHOD_COLORS.get(method, "#333333")
                label_name = METHOD_LABELS.get(method, method.upper())

                unc_normalized = (unc - unc.min()) / (unc.max() - unc.min() + 1e-10)
                sizes = 30 + 150 * unc_normalized

                unique_labels = np.unique(labels)
                n_colors = min(len(unique_labels), 10)
                cmap = plt.cm.get_cmap('tab10', n_colors)
                label_colors = [cmap(l % n_colors) for l in labels]

                sc = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                               c=label_colors, s=sizes, alpha=0.6,
                               edgecolors=color, linewidths=0.5)
                artists.append(sc)

                ax.set_title(f"{label_name}\nEpoch {epoch}", fontsize=11)
                ax.grid(True, alpha=0.3)

                txt = ax.text(0.02, 0.98,
                             f"Mean: {unc.mean():.5f}\nStd: {unc.std():.5f}",
                             transform=ax.transAxes, fontsize=8, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                artists.append(txt)

            return artists

        anim = animation.FuncAnimation(
            fig, animate, frames=num_epochs, interval=800, blit=False
        )

        save_path = output_dir / f"2_decision_boundary_{self.task}.gif"
        anim.save(str(save_path), writer='pillow', fps=1.5)
        plt.close()
        print(f"  Saved: {save_path}")


# ============================================================
# 시각화 3: 2D Loss Landscape & Trajectory (GIF)
# ============================================================
class LossLandscapeVisualizer:
    """파라미터 공간에서의 최적화 궤적 시각화"""

    def __init__(self, methods: list, task: str, device: torch.device,
                 num_epochs: int = 10):
        self.methods = methods
        self.task = task
        self.device = device
        self.num_epochs = num_epochs

    def collect_trajectory(self, model, train_loader, num_epochs: int, eval_batch):
        """학습 중 출력 공간에서의 궤적 수집"""
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
        )

        trajectory = []
        losses = []

        def get_output_snapshot():
            """현재 모델의 출력 스냅샷"""
            model.eval()
            with torch.no_grad():
                outputs = model(pixel_values=eval_batch["pixel_values"])
                logits = outputs.logits.cpu().flatten()
            model.train()
            return logits

        # 초기 출력 저장
        trajectory.append(get_output_snapshot())
        losses.append(float('inf'))

        model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if num_batches >= 20:
                    break

            trajectory.append(get_output_snapshot())
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        return torch.stack(trajectory).numpy(), losses

    def run(self, processor, raw_dataset, num_labels: int, output_dir: Path, batch_size: int = 16):
        """Loss Landscape 시각화 실행"""
        print("\n" + "="*60)
        print(" Visualization 3: 2D Loss Landscape & Trajectory")
        print("="*60)

        train_key = "train" if "train" in raw_dataset else list(raw_dataset.keys())[0]
        train_ds = raw_dataset[train_key]

        subset_size = min(500, len(train_ds))
        train_subset = train_ds.select(range(subset_size))

        def preprocess(examples):
            images = [img.convert("RGB") for img in examples["image"]]
            inputs = processor(images, return_tensors="pt")
            return {"pixel_values": inputs["pixel_values"], "labels": examples["label"]}

        train_subset = train_subset.map(preprocess, batched=True, remove_columns=train_subset.column_names)
        train_subset.set_format("torch")
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        # 고정된 평가 배치
        eval_batch = None
        for batch in train_loader:
            eval_batch = {
                "pixel_values": batch["pixel_values"][:8].to(self.device),
                "labels": batch["labels"][:8].to(self.device)
            }
            break

        all_trajectories = {}
        all_losses = {}

        for method in self.methods:
            print(f"\n[{method.upper()}] Collecting trajectory...")
            model = build_model(method, num_labels).to(self.device)

            trajectory, losses = self.collect_trajectory(
                model, train_loader, self.num_epochs, eval_batch
            )
            all_trajectories[method] = trajectory
            all_losses[method] = losses

            del model
            torch.cuda.empty_cache()

        # PCA 투영
        all_outputs = np.vstack(list(all_trajectories.values()))
        pca = PCA(n_components=2)
        pca.fit(all_outputs)

        trajectories_2d = {}
        for method, traj in all_trajectories.items():
            trajectories_2d[method] = pca.transform(traj)

        # 시각화
        self._plot_static(trajectories_2d, all_losses, output_dir)
        self._plot_animation(trajectories_2d, all_losses, output_dir)

        return {"trajectories_2d": trajectories_2d, "losses": all_losses}

    def _plot_static(self, trajectories_2d: dict, losses: dict, output_dir: Path):
        """정적 궤적 플롯"""
        fig, ax = plt.subplots(figsize=(10, 8))

        for method in self.methods:
            traj = trajectories_2d[method]
            method_losses = losses[method]
            color = METHOD_COLORS.get(method, "#333333")
            label = METHOD_LABELS.get(method, method.upper())
            marker = METHOD_MARKERS.get(method, "o")

            # 궤적 라인
            ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2.5,
                   label=label, alpha=0.8)

            # 시작점
            ax.scatter(traj[0, 0], traj[0, 1], c=color, s=200, marker='o',
                      edgecolors='white', linewidths=2, zorder=5)

            # 끝점
            ax.scatter(traj[-1, 0], traj[-1, 1], c=color, s=250, marker='*',
                      edgecolors='white', linewidths=1, zorder=5)

            # 중간 포인트 (Loss로 크기 조절)
            for i in range(1, len(traj)-1):
                if i < len(method_losses):
                    size = max(20, 100 - method_losses[i] * 10)
                else:
                    size = 30
                ax.scatter(traj[i, 0], traj[i, 1], c=color, s=size,
                          marker=marker, alpha=0.6)

        ax.set_xlabel("PCA Component 1 (Output Space)", fontsize=11)
        ax.set_ylabel("PCA Component 2 (Output Space)", fontsize=11)
        ax.set_title(f"Optimization Trajectory ({self.task.upper()})\n○: Start, ★: End",
                    fontsize=13)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = output_dir / f"3_loss_landscape_{self.task}.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

    def _plot_animation(self, trajectories_2d: dict, losses: dict, output_dir: Path):
        """궤적 애니메이션"""
        fig, ax = plt.subplots(figsize=(10, 8))

        all_points = np.vstack(list(trajectories_2d.values()))
        x_min, x_max = all_points[:, 0].min() - 1, all_points[:, 0].max() + 1
        y_min, y_max = all_points[:, 1].min() - 1, all_points[:, 1].max() + 1

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("PCA Component 1", fontsize=11)
        ax.set_ylabel("PCA Component 2", fontsize=11)
        ax.grid(True, alpha=0.3)

        lines = {}
        points = {}
        current_markers = {}

        for method in self.methods:
            color = METHOD_COLORS.get(method, "#333333")
            label = METHOD_LABELS.get(method, method.upper())
            marker = METHOD_MARKERS.get(method, "o")

            lines[method], = ax.plot([], [], '-', color=color, linewidth=2.5,
                                    label=label, alpha=0.7)
            points[method], = ax.plot([], [], marker, color=color, markersize=6, alpha=0.5)
            current_markers[method], = ax.plot([], [], marker, color=color, markersize=15,
                                               markeredgecolor='white', markeredgewidth=2)

        ax.legend(loc='upper right')
        title = ax.set_title("", fontsize=13)

        # Loss 텍스트
        loss_texts = {}
        for i, method in enumerate(self.methods):
            color = METHOD_COLORS.get(method, "#333333")
            loss_texts[method] = ax.text(0.02, 0.98 - i*0.08, "",
                                        transform=ax.transAxes, fontsize=9,
                                        color=color, verticalalignment='top')

        max_frames = max(len(traj) for traj in trajectories_2d.values())

        def init():
            for method in self.methods:
                lines[method].set_data([], [])
                points[method].set_data([], [])
                current_markers[method].set_data([], [])
                loss_texts[method].set_text("")
            title.set_text("")
            return (list(lines.values()) + list(points.values()) +
                    list(current_markers.values()) + list(loss_texts.values()) + [title])

        def animate(frame):
            for method in self.methods:
                traj = trajectories_2d[method]
                method_losses = losses[method]
                n = min(frame + 1, len(traj))

                lines[method].set_data(traj[:n, 0], traj[:n, 1])
                points[method].set_data(traj[:n, 0], traj[:n, 1])
                current_markers[method].set_data([traj[n-1, 0]], [traj[n-1, 1]])

                if n-1 < len(method_losses) and method_losses[n-1] != float('inf'):
                    loss_texts[method].set_text(f"{METHOD_LABELS[method]}: Loss={method_losses[n-1]:.3f}")
                else:
                    loss_texts[method].set_text(f"{METHOD_LABELS[method]}: Initial")

            title.set_text(f"Optimization Trajectory ({self.task.upper()}) - Epoch {frame}")
            return (list(lines.values()) + list(points.values()) +
                    list(current_markers.values()) + list(loss_texts.values()) + [title])

        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=max_frames, interval=600, blit=True
        )

        save_path = output_dir / f"3_loss_landscape_{self.task}.gif"
        anim.save(str(save_path), writer='pillow', fps=2)
        plt.close()
        print(f"  Saved: {save_path}")


# ============================================================
# 메인 실행
# ============================================================
def main(args):
    setup_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 절대 경로로 변환하여 디렉토리 생성
    base_dir = Path(args.output_dir).resolve()
    output_dir = base_dir / f"visualization_{args.task}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Output directory: {output_dir}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f" LAVA Visualization Engine v2")
    print(f" [Stochastic Mode: ENABLED]")
    print(f"{'='*60}")
    print(f" Task: {args.task}")
    print(f" Methods: {args.methods}")
    print(f" Device: {device}")
    print(f" MC Passes: {args.num_passes}")
    print(f" Output: {output_dir}")
    print(f"{'='*60}")

    methods = [m.strip() for m in args.methods.split(",")]

    print("\n[*] Loading dataset...")
    raw_dataset, meta = load_dataset_for_task(args.task, args.seed)
    num_labels = meta["num_labels"]

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    results = {}

    # 1. Cone vs. Tube
    if args.vis1:
        vis1 = ConeVsTubeVisualizer(
            methods=methods, task=args.task, device=device,
            num_samples=args.num_samples, num_passes=args.num_passes
        )
        results["cone_vs_tube"] = vis1.run(
            processor, raw_dataset, num_labels, output_dir,
            train_epochs=3, batch_size=args.batch_size
        )

    # 2. Decision Margin
    if args.vis2:
        vis2 = DecisionMarginVisualizer(
            methods=methods, task=args.task, device=device,
            num_samples=min(args.num_samples, 200), num_passes=args.num_passes,
            num_epochs=args.num_epochs
        )
        results["decision_margin"] = vis2.run(processor, raw_dataset, num_labels, output_dir)

    # 3. Loss Landscape
    if args.vis3:
        vis3 = LossLandscapeVisualizer(
            methods=methods, task=args.task, device=device,
            num_epochs=args.num_epochs
        )
        results["loss_landscape"] = vis3.run(processor, raw_dataset, num_labels, output_dir, args.batch_size)

    # 메타데이터 저장
    metadata = {
        "task": args.task,
        "methods": methods,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "num_passes": args.num_passes,
        "num_epochs": args.num_epochs,
        "timestamp": timestamp,
        "stochastic_mode": "ENABLED",
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f" Visualization Complete!")
    print(f" Generated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"   - {f.name}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAVA Unified Visualization Engine v2")

    parser.add_argument("--task", type=str, required=True,
                       choices=list(IMG_TASK_META.keys()))
    parser.add_argument("--methods", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")

    parser.add_argument("--vis1", action="store_true", default=True)
    parser.add_argument("--vis2", action="store_true", default=True)
    parser.add_argument("--vis3", action="store_true", default=True)
    parser.add_argument("--all", action="store_true")

    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_passes", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)

    args = parser.parse_args()

    if args.all:
        args.vis1 = True
        args.vis2 = True
        args.vis3 = True

    main(args)
