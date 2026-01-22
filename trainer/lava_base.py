import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import seed_worker
from typing import Dict, Union, Any, Optional

class LavaBaseTrainer(Trainer):
    def __init__(self, *args, lambda_vib=1.0, lambda_latent_stability=1.0, lambda_stab=0.1,
                 dataloader_seed: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_vib = lambda_vib
        self.lambda_latent_stability = lambda_latent_stability
        self.lambda_stab = lambda_stab  # 이제 0.1 등의 가중치가 정상 작동합니다.

        self.loss_track = {"task_loss": 0, "vib": 0, "logit_stab": 0, "latent_stab": 0}

        # DataLoader 재현성을 위한 seed 저장
        self.dataloader_seed = dataloader_seed if dataloader_seed is not None else self.args.seed

    def _get_worker_init_fn(self):
        """각 worker에 대해 재현 가능한 seed를 설정하는 함수 반환"""
        base_seed = self.dataloader_seed
        def worker_init_fn(worker_id):
            worker_seed = (base_seed + worker_id) % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        return worker_init_fn

    def get_train_dataloader(self) -> DataLoader:
        """Train DataLoader에 worker_init_fn과 generator 추가"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # Reproducibility를 위한 generator
        g = torch.Generator()
        g.manual_seed(self.dataloader_seed)

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": True,
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": self._get_worker_init_fn(),
            "generator": g,
        }

        return DataLoader(train_dataset, **dataloader_params)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Eval DataLoader에 worker_init_fn과 generator 추가"""
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        # Reproducibility를 위한 generator (eval은 shuffle 없으므로 선택적)
        g = torch.Generator()
        g.manual_seed(self.dataloader_seed)

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
            "drop_last": False,
            "worker_init_fn": self._get_worker_init_fn(),
            "generator": g,
        }

        return DataLoader(eval_dataset, **dataloader_params)

    def _get_task_specific_inputs(self, inputs):
        return inputs

    def compute_task_loss(self, logits, labels):
        """기본 분류 Task Loss (CrossEntropy)"""
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

    def compute_loss(self, model, inputs, return_outputs=False):
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs)

        # 1. 태스크별 입력 정제
        inputs = self._get_task_specific_inputs(inputs)
        labels = inputs["labels"]
        
        # 2. Logit Stability를 위해 배치를 2N으로 복제 (Concat)
        # 동일한 데이터가 모델의 stochastic한 LAVA 레이어를 두 번 통과하게 됩니다.
        concat_inputs = {
            k: torch.cat([v, v], dim=0) for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }
        
        # 3. Forward Pass (2N Batch)
        outputs = model(**concat_inputs)
        logits = outputs.logits
        
        # 4. 출력을 다시 절반으로 나눔 (N, N)
        logits1, logits2 = logits.chunk(2, dim=0)
        
        # 5. Task Loss 계산 (2N 전체에 대해 계산)
        full_labels = torch.cat([labels, labels], dim=0)
        task_loss = self.compute_task_loss(logits, full_labels)

        # 6. Logit Stability Loss 계산 (Symmetric KL Divergence / JSD)
        # 같은 데이터에 대해 샘플링 노이즈가 섞였을 때 결과가 얼마나 일관된지 측정
        p = F.log_softmax(logits1, dim=-1)
        q = F.softmax(logits2, dim=-1)
        p_rev = F.log_softmax(logits2, dim=-1)
        q_rev = F.softmax(logits1, dim=-1)
        
        # 두 분포 사이의 거리를 계산 (Stability)
        stab_loss = (F.kl_div(p, q, reduction='batchmean') + F.kl_div(p_rev, q_rev, reduction='batchmean')) / 2

        # 7. VIB & Latent Stability 수집
        kl_divs, latent_stabs = [], []
        for m in model.modules():
            # VIB (KL Divergence)
            if hasattr(m, "_last_mu") and m._last_mu is not None:
                # 2N 배치이므로 첫 절반(N)의 통계량만 사용하거나 평균을 냅니다.
                mu, logvar = m._last_mu.chunk(2, dim=0)[0], m._last_logvar.chunk(2, dim=0)[0]
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
                kl_divs.append(kl)
                m._last_mu, m._last_logvar = None, None
            
            # Latent Stability (내부 Antithetic Sampling 결과)
            if hasattr(m, "_latent_stability") and m._latent_stability is not None:
                latent_stabs.append(m._latent_stability)
                m._latent_stability = None

        vib_loss = torch.stack(kl_divs).mean() if kl_divs else torch.tensor(0.0).to(task_loss.device)
        latent_loss = torch.stack(latent_stabs).mean() if latent_stabs else torch.tensor(0.0).to(task_loss.device)

        # 8. 최종 합산 (Task + LogitStab + VIB + LatentStab)
        loss = task_loss + \
               (self.lambda_stab * stab_loss) + \
               (self.lambda_vib * vib_loss) + \
               (self.lambda_latent_stability * latent_loss)
        
        # 로깅 데이터 업데이트
        self.loss_track.update({
            "task_loss": task_loss.item(), 
            "vib": vib_loss.item(),
            "logit_stab": stab_loss.item(), 
            "latent_stab": latent_loss.item()
        })

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        logs["train/task_loss"] = self.loss_track["task_loss"]
        logs["train/vib_raw"] = self.loss_track["vib"]
        logs["train/logit_stab_raw"] = self.loss_track["logit_stab"]
        logs["train/latent_stab_raw"] = self.loss_track["latent_stab"]
        super().log(logs)