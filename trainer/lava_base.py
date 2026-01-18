import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Union, Any

class LavaBaseTrainer(Trainer):
    def __init__(self, *args, lambda_vib=1.0, lambda_latent_stability=1.0, lambda_stab=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_vib = lambda_vib
        self.lambda_latent_stability = lambda_latent_stability
        self.lambda_stab = lambda_stab
        
        # 상세 Loss 추적용
        self.loss_track = {
            "ce_loss": 0, "raw_vib": 0, "raw_stab": 0, "raw_latent": 0
        }

    def _get_task_specific_inputs(self, inputs):
        """하위 클래스에서 데이터 특성에 맞게 오버라이드"""
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs)

        # 1. 입력 준비 (2N 배치 생성)
        inputs = self._get_task_specific_inputs(inputs)
        labels = inputs["labels"]
        concat_inputs = {k: torch.cat([v, v], dim=0) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # 2. Forward Pass
        outputs = model(**concat_inputs)
        logits = outputs.logits
        logits1, logits2 = logits.chunk(2, dim=0)
        
        # 3. Task Loss (기본적으로 CrossEntropy 사용, 필요시 서브클래스에서 변경)
        full_labels = torch.cat([labels, labels], dim=0)
        task_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), full_labels.view(-1))

        # 4. Logit Stability Loss (JSD)
        p = F.log_softmax(logits1, dim=-1)
        q = F.softmax(logits2, dim=-1)
        p_rev = F.log_softmax(logits2, dim=-1)
        q_rev = F.softmax(logits1, dim=-1)
        stab_loss = (F.kl_div(p, q, reduction='batchmean') + F.kl_div(p_rev, q_rev, reduction='batchmean')) / 2

        # 5. LAVA 내부 Loss 수집 (VIB & Latent Stability)
        kl_divs, latent_stabs = [], []
        for m in model.modules():
            if hasattr(m, "_last_mu") and m._last_mu is not None:
                mu, logvar = m._last_mu.chunk(2)[0], m._last_logvar.chunk(2)[0]
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
                kl_divs.append(kl)
                m._last_mu, m._last_logvar = None, None
            if hasattr(m, "_latent_stability") and m._latent_stability is not None:
                latent_stabs.append(m._latent_stability)
                m._latent_stability = None

        vib_loss = torch.stack(kl_divs).mean() if kl_divs else torch.tensor(0.0).to(task_loss.device)
        latent_loss = torch.stack(latent_stabs).mean() if latent_stabs else torch.tensor(0.0).to(task_loss.device)

        # 6. 합산
        loss = task_loss + (self.lambda_stab * stab_loss) + (self.lambda_vib * vib_loss) + (self.lambda_latent_stability * latent_loss)
        
        # 로깅용 업데이트
        self.loss_track.update({
            "ce_loss": task_loss.item(), "raw_vib": vib_loss.item(),
            "raw_stab": stab_loss.item(), "raw_latent": latent_loss.item()
        })

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """WandB 등에 상세 Loss 기록"""
        logs["train/ce_loss"] = self.loss_track["ce_loss"]
        logs["train/vib_raw"] = self.loss_track["raw_vib"]
        logs["train/stab_raw"] = self.loss_track["raw_stab"]
        logs["train/latent_raw"] = self.loss_track["raw_latent"]
        super().log(logs)