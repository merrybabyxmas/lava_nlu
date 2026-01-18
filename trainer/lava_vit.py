from .lava_base import LavaBaseTrainer
import torch

class LavaViTTrainer(LavaBaseTrainer):
    def _get_task_specific_inputs(self, inputs):
        # ViT에 불필요한 키(텍스트용 등)가 섞여 들어오는 것을 방지
        vit_keys = {"pixel_values", "labels"}
        return {k: v for k, v in inputs.items() if k in vit_keys}