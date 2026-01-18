from .lava_base import LavaBaseTrainer
import torch.nn as nn

class LavaNLUTrainer(LavaBaseTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Regression(STSB 등)인 경우 Task Loss만 MSE로 교체하기 위해 로직 확장 가능
        # 여기서는 기존 로직을 최대한 활용하되, 라벨 타입에 따른 처리를 추가합니다.
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs)
            
        labels = inputs["labels"]
        if labels.dtype in [torch.float32, torch.float64]:
            # Regression용 로직은 base를 호출하지 않고 직접 구현하거나 base를 수정하여 대응
            # (편의상 여기서는 base의 CE를 오버라이드하는 방식 제안)
            return super().compute_loss(model, inputs, return_outputs) 
        return super().compute_loss(model, inputs, return_outputs)