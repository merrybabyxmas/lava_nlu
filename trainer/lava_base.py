import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, List, Optional


class LavaBaseTrainer(Trainer):
    """
    LAVA Trainer - Hyperparameter Version
    ====================================
    - Fixed Hyperparameters: lambda_vib, lambda_latent_stability
    - Single-pass (N batch) training
    - Cached LAVA layers & Vectorized loss computation
    - Uses standard Trainer's DataLoader for fair comparison (same shuffling as other adapters)
    """

    def __init__(self, *args, lambda_vib=1.0, lambda_latent_stability=1.0, **kwargs):
        # NOTE: dataloader_seed parameter removed - now uses TrainingArguments.seed
        # This ensures identical data ordering between LavaTrainer and standard Trainer
        kwargs.pop('dataloader_seed', None)  # Remove if passed for backward compatibility
        super().__init__(*args, **kwargs)

        # [1] 하이퍼파라미터 설정 (고정값)
        self.lambda_vib = lambda_vib
        self.lambda_latent_stability = lambda_latent_stability

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

    # NOTE: Removed custom get_train_dataloader() and get_eval_dataloader()
    # Now uses standard Trainer's DataLoader implementation for fair comparison
    # The seed from TrainingArguments is used for reproducible shuffling

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
            # Flatten each mu/logvar to 2D [N, rank] before concat
            # (handles different sequence lengths across layers)
            flattened_mus = [mu.view(-1, mu.size(-1)) for mu in mus]
            flattened_logvars = [lv.view(-1, lv.size(-1)) for lv in logvars]
            all_mus = torch.cat(flattened_mus, dim=0)
            all_logvars = torch.cat(flattened_logvars, dim=0)
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