from .lava_base import LavaBaseTrainer

class LavaViTTrainer(LavaBaseTrainer):
    def _get_task_specific_inputs(self, inputs):
        # ViT 학습에 필요한 키만 남김
        vit_keys = {"pixel_values", "labels"}
        return {k: v for k, v in inputs.items() if k in vit_keys}