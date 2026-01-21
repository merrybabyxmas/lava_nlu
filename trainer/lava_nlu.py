from .lava_base import LavaBaseTrainer
import torch
import torch.nn as nn

class LavaNLUTrainer(LavaBaseTrainer):
    def compute_task_loss(self, logits, labels):
        # Regression 체크
        if labels.dtype in [torch.float32, torch.float64]:
            loss_fct = nn.MSELoss()
            # Regression일 경우 logits의 마지막 차원이 1이어야 함
            return loss_fct(logits.view(-1), labels.view(-1))
        else:
            # Classification
            return super().compute_task_loss(logits, labels)