"""
Trainer Utilities
=================
train_vit.py와 train_nlu.py에서 공통으로 사용되는 유틸리티 함수들
"""

import random
import numpy as np
import torch
import wandb
from transformers import TrainerCallback

import peft.utils.save_and_load
import peft.mapping
from peft.utils.peft_types import PeftType
from peft.tuners.lava.config import LavaConfig
from peft.tuners.lava.model import LavaModel


def setup_seed(seed: int):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # LavaAdapter의 global seed 설정
    try:
        from peft.tuners.lava.layer import LavaAdapter
        LavaAdapter.set_global_seed(seed)
    except ImportError:
        pass  # LAVA가 등록되지 않은 경우 무시


def reset_lava_generators(model, seed: int = None):
    """모델 내 모든 LavaAdapter의 generator를 리셋 (에폭 시작 시 호출 가능)"""
    try:
        from peft.tuners.lava.layer import LavaAdapter
        for module in model.modules():
            if isinstance(module, LavaAdapter):
                module.reset_generator(seed)
    except ImportError:
        pass

def register_lava():
    """LAVA를 PEFT 매핑에 등록"""
    if not hasattr(PeftType, "LAVA"):
        PeftType.LAVA = "LAVA"

    for lava_key in ["LAVA", PeftType.LAVA]:
        peft.mapping.PEFT_TYPE_TO_CONFIG_MAPPING[lava_key] = LavaConfig
        peft.mapping.PEFT_TYPE_TO_TUNER_MAPPING[lava_key] = LavaModel
        peft.utils.save_and_load.PEFT_TYPE_TO_PREFIX_MAPPING[lava_key] = "adapter_model"
        peft.mapping.PEFT_TYPE_TO_PREFIX_MAPPING[lava_key] = "adapter_model"


class BestMetricCallback(TrainerCallback):
    """
    학습 중 최고 메트릭을 추적하는 콜백

    Args:
        main_metric: 추적할 메트릭 이름 (예: "accuracy", "f1", "pearson")
                    None이면 "accuracy"를 사용
    """
    def __init__(self, main_metric: str = None):
        if main_metric:
            self.main_metric = f"eval_{main_metric}"
        else:
            self.main_metric = "eval_accuracy"
        self.best_score = -float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        if metrics:
            # 메트릭 키 확인 (eval_accuracy 또는 main_metric)
            current = metrics.get(self.main_metric)
            if current is None:
                current = metrics.get("eval_accuracy", metrics.get("accuracy"))

            if current is not None:
                is_best = current > self.best_score
                if is_best:
                    self.best_score = current
                print(f"[EVAL] Epoch {epoch}: {self.main_metric} = {current:.4f} | Best = {self.best_score:.4f}" + (" *" if is_best else ""))
                wandb.log({"eval/best_main": self.best_score}, step=state.global_step)
            else:
                loss = metrics.get("eval_loss", 0)
                print(f"[EVAL] Epoch {epoch}: Loss = {loss:.4f}")

    def on_train_begin(self, args, state, control, **kwargs):
        print("[TRAIN] Training started...")

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        print(f"[TRAIN] Epoch {epoch + 1}/{int(args.num_train_epochs)} starting...")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"[TRAIN] Training completed. Best score: {self.best_score:.4f}")


def print_trainable_parameters(model):
    """모델의 학습 가능한 파라미터 수와 비율 출력"""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    percentage = 100 * trainable_params / all_params
    print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {percentage:.4f}")
    return trainable_params, all_params, percentage
