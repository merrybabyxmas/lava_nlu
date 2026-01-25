import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from torch.utils.data import DataLoader
from transformers import Trainer
from typing import Dict, List, Optional


class LavaBaseTrainer(Trainer):
    """
    LAVA Trainer - Hyperparameter Version
    ====================================
    - Fixed Hyperparameters: lambda_vib, lambda_latent_stability
    - Single-pass (N batch) training
    - Cached LAVA layers & Vectorized loss computation
    - Logit Stability Removed
    """

    def __init__(self, *args, lambda_vib=1.0, lambda_latent_stability=1.0, dataloader_seed=42, **kwargs):
        super().__init__(*args, **kwargs)

        # [1] 하이퍼파라미터 설정 (고정값)
        self.lambda_vib = lambda_vib
        self.lambda_latent_stability = lambda_latent_stability
        self.dataloader_seed = dataloader_seed

        # loss 추적용 딕셔너리 초기화
        self.loss_track = {}

        # [2] LAVA 레이어 캐싱
        self.lava_layers: List[nn.Module] = []
        self._cache_lava_layers()

    def _cache_lava_layers(self):
        self.lava_layers = []
        for m in self.model.modules():
            # LavaAdapter의 고유 속성 존재 여부 확인
            if hasattr(m, "_last_mu") and hasattr(m, "_last_logvar"):
                self.lava_layers.append(m)

    def _ensure_lava_layers_cached(self):
        if len(self.lava_layers) == 0:
            self._cache_lava_layers()

    def _get_worker_init_fn(self):
        base_seed = self.dataloader_seed
        def worker_init_fn(worker_id):
            worker_seed = (base_seed + worker_id) % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        return worker_init_fn

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        g = torch.Generator()
        g.manual_seed(self.dataloader_seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
            worker_init_fn=self._get_worker_init_fn(),
            generator=g,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        g = torch.Generator()
        g.manual_seed(self.dataloader_seed)
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
            worker_init_fn=self._get_worker_init_fn(),
            generator=g,
        )

    def _get_task_specific_inputs(self, inputs):
        return inputs

    def compute_task_loss(self, logits, labels):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

    def compute_loss(self, model, inputs, return_outputs=False):
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs)

        self._ensure_lava_layers_cached()
        inputs = self._get_task_specific_inputs(inputs)
        labels = inputs["labels"]

        # [3] Single-pass Forward (N 배치)
        outputs = model(**inputs)
        logits = outputs.logits
        task_loss = self.compute_task_loss(logits, labels)

        # [4] Vectorized Loss 수집
        mus, logvars, latent_stabs = [], [], []

        for layer in self.lava_layers:
            if layer._last_mu is not None:
                mus.append(layer._last_mu)
                logvars.append(layer._last_logvar)
                layer._last_mu = None # 메모리 비우기
                layer._last_logvar = None

            if layer._latent_stability is not None:
                latent_stabs.append(layer._latent_stability)
                layer._latent_stability = None

        # VIB Loss (KL) 계산
        if mus:
            all_mus = torch.cat(mus, dim=0)
            all_logvars = torch.cat(logvars, dim=0)
            vib_loss = -0.5 * torch.mean(1 + all_logvars - all_mus.pow(2) - all_logvars.exp())
        else:
            vib_loss = torch.tensor(0.0, device=task_loss.device)

        # Latent Stability Loss 계산
        if latent_stabs:
            latent_loss = torch.stack(latent_stabs).mean()
        else:
            latent_loss = torch.tensor(0.0, device=task_loss.device)

        # [5] 하이퍼파라미터 가중치 적용 (Fixed)
        loss = task_loss + (self.lambda_vib * vib_loss) + (self.lambda_latent_stability * latent_loss)

        # 로깅 데이터 업데이트
        self.loss_track.update({
            "task_loss": task_loss.item(),
            "vib_raw": vib_loss.item(),
            "latent_stab_raw": latent_loss.item(),
            "lambda_vib_val": self.lambda_vib,
            "lambda_latent_val": self.lambda_latent_stability,
        })

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        logs.update({f"train/{k}": v for k, v in self.loss_track.items()})
        super().log(logs)