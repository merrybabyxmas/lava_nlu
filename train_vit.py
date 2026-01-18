#!/usr/bin/env python
"""
ViT Image Classification Training
==================================
ViT-B/16을 사용한 이미지 분류 학습 (DTD, EuroSAT, GTSRB, RESISC45, SUN397, SVHN)
"""

import argparse
import torch
import random
import numpy as np
import os
import json
import tempfile
import shutil
import ssl

# SSL 인증서 검증 우회 (데이터셋 다운로드용)
ssl._create_default_https_context = ssl._create_unverified_context
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from torchvision import transforms
from torchvision import datasets as tv_datasets
from datasets import Dataset
import wandb

from peft import get_peft_model, LoraConfig, AdaLoraConfig
from peft.tuners.lava.config import LavaConfig
import peft.utils.save_and_load
import peft.mapping
from peft.utils.peft_types import PeftType
from peft.tuners.lava.model import LavaModel
from trainer import LavaViTTrainer
# LAVA 등록
if not hasattr(PeftType, "LAVA"):
    PeftType.LAVA = "LAVA"
for lava_key in ["LAVA", PeftType.LAVA]:
    peft.mapping.PEFT_TYPE_TO_CONFIG_MAPPING[lava_key] = LavaConfig
    peft.mapping.PEFT_TYPE_TO_TUNER_MAPPING[lava_key] = LavaModel
    peft.utils.save_and_load.PEFT_TYPE_TO_PREFIX_MAPPING[lava_key] = "adapter_model"
    peft.mapping.PEFT_TYPE_TO_PREFIX_MAPPING[lava_key] = "adapter_model"


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 데이터셋 설정
# ============================================================
IMG_TASK_META = {
    "dtd": dict(
        source="torchvision",
        tv_class=tv_datasets.DTD,
        num_labels=47,
    ),
    "eurosat": dict(
        source="torchvision",
        tv_class=tv_datasets.EuroSAT,
        num_labels=10,
    ),
    "gtsrb": dict(
        source="torchvision",
        tv_class=tv_datasets.GTSRB,
        num_labels=43,
    ),
    "resisc45": dict(
        source="huggingface",
        dataset_name="timm/resisc45",
        num_labels=45,
        split_train="train",
        split_val="test"
    ),
    "sun397": dict(
        source="huggingface",
        dataset_name="tanganke/sun397",
        num_labels=397,
        split_train="train",
        split_val="test"
    ),
    "svhn": dict(
        source="torchvision",
        tv_class=tv_datasets.SVHN,
        num_labels=10,
    ),
}

# 태스크별 하이퍼파라미터
IMG_TASK_CONFIG = {
    "dtd": dict(epochs=50, batch=32, lr=1e-4),
    "eurosat": dict(epochs=20, batch=32, lr=1e-4),
    "gtsrb": dict(epochs=20, batch=32, lr=1e-4),
    "resisc45": dict(epochs=20, batch=32, lr=1e-4),
    "sun397": dict(epochs=30, batch=32, lr=1e-4),
    "svhn": dict(epochs=10, batch=32, lr=1e-4),
}


# ============================================================
# Torchvision Dataset Loader
# ============================================================
def load_torchvision_dataset(task: str, meta: dict, data_root: str = "./data"):
    """
    Torchvision 데이터셋을 HuggingFace Dataset 형식으로 변환
    """
    tv_class = meta["tv_class"]

    # DTD는 split 파라미터 사용
    if task == "dtd":
        train_ds = tv_class(root=data_root, split="train", download=True)
        val_ds = tv_class(root=data_root, split="test", download=True)
    # GTSRB는 split 파라미터 사용
    elif task == "gtsrb":
        train_ds = tv_class(root=data_root, split="train", download=True)
        val_ds = tv_class(root=data_root, split="test", download=True)
    # SVHN은 split 파라미터 사용
    elif task == "svhn":
        train_ds = tv_class(root=data_root, split="train", download=True)
        val_ds = tv_class(root=data_root, split="test", download=True)
    # EuroSAT, SUN397는 train 파라미터 사용
    elif task in ["eurosat", "sun397"]:
        # 전체 데이터셋 로드 후 분할
        full_ds = tv_class(root=data_root, download=True)
        # 80/20 split
        total_len = len(full_ds)
        train_len = int(0.8 * total_len)
        val_len = total_len - train_len
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [train_len, val_len],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    def convert_to_hf_dataset(tv_dataset):
        """Torchvision dataset을 HuggingFace Dataset으로 변환"""
        images = []
        labels = []

        # random_split 결과인 경우 Subset 처리
        if hasattr(tv_dataset, 'dataset'):
            # Subset인 경우
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

    train_hf = convert_to_hf_dataset(train_ds)
    val_hf = convert_to_hf_dataset(val_ds)

    return {"train": train_hf, "test": val_hf}


class BestMetricCallback(TrainerCallback):
    def __init__(self):
        self.best_accuracy = 0.0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        if metrics:
            # eval_accuracy 또는 accuracy 키 확인
            current = metrics.get("eval_accuracy", metrics.get("accuracy"))
            if current is not None:
                is_best = current > self.best_accuracy
                if is_best:
                    self.best_accuracy = current
                print(f"[EVAL] Epoch {epoch}: Accuracy = {current:.4f} | Best = {self.best_accuracy:.4f}" + (" *" if is_best else ""))
                wandb.log({"eval/best_accuracy": self.best_accuracy}, step=state.global_step)
            else:
                # accuracy가 없으면 loss만 출력
                loss = metrics.get("eval_loss", 0)
                print(f"[EVAL] Epoch {epoch}: Loss = {loss:.4f}")

    def on_train_begin(self, args, state, control, **kwargs):
        print("[TRAIN] Training started...")

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        print(f"[TRAIN] Epoch {epoch + 1}/{int(args.num_train_epochs)} starting...")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"[TRAIN] Training completed. Best accuracy: {self.best_accuracy:.4f}")


# ============================================================
# LAVA Trainer (Stability Loss 포함)
# ============================================================
class StabilityViTTrainer(Trainer):
    def __init__(self, *args, lambda_vib=0.1, lambda_latent_stability=0.1, lambda_stab=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_vib = lambda_vib
        self.lambda_latent_stability = lambda_latent_stability
        self.lambda_stab = lambda_stab

    def compute_loss(self, model, inputs, return_outputs=False):
        import torch.nn.functional as F

        if not model.training:
            return super().compute_loss(model, inputs, return_outputs)

        labels = inputs["labels"]
        # ViT는 pixel_values와 labels만 필요 (input_ids 등 텍스트 관련 키 제외)
        vit_keys = {"pixel_values", "labels"}
        concat_inputs = {k: torch.cat([v, v], dim=0) for k, v in inputs.items()
                        if isinstance(v, torch.Tensor) and k in vit_keys}

        outputs = model(**concat_inputs)
        logits = outputs.logits

        logits1, logits2 = logits.chunk(2, dim=0)
        full_labels = torch.cat([labels, labels], dim=0)

        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), full_labels.view(-1))

        p = F.log_softmax(logits1, dim=-1)
        q = F.softmax(logits2, dim=-1)
        p_rev = F.log_softmax(logits2, dim=-1)
        q_rev = F.softmax(logits1, dim=-1)
        const_loss = (F.kl_div(p, q, reduction='batchmean') +
                      F.kl_div(p_rev, q_rev, reduction='batchmean')) / 2

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

        vib_loss = torch.stack(kl_divs).mean() if kl_divs else torch.tensor(0.0).to(ce_loss.device)
        latent_stab_loss = torch.stack(latent_stabs).mean() if latent_stabs else torch.tensor(0.0).to(ce_loss.device)

        loss = ce_loss + self.lambda_stab * const_loss + self.lambda_vib * vib_loss + self.lambda_latent_stability * latent_stab_loss

        return (loss, outputs) if return_outputs else loss


def build_adapter(adapter_type, r=8, alpha=8, total_step=None):
    at = adapter_type.lower()
    target_modules = ["query", "key", "value", "dense"]

    if at in ["lora", "dora", "pissa"]:
        kwargs = dict(r=r, lora_alpha=alpha, target_modules=target_modules)
        if at == "pissa":
            kwargs["init_lora_weights"] = "pissa"
        if at == "dora":
            kwargs["use_dora"] = True
        return LoraConfig(**kwargs)

    if at == "adalora":
        # AdaLoRA requires total_step for rank scheduling
        return AdaLoraConfig(
            init_r=r,
            target_r=r // 2,  # final rank
            lora_alpha=alpha,
            target_modules=target_modules,
            total_step=total_step if total_step else 1000,
        )

    if at == "lava":
        return LavaConfig(r=r, alpha=alpha, target_modules=target_modules)

    if at == "bitfit":
        return "bitfit"

    raise ValueError(f"Unknown adapter: {adapter_type}")


def main(args):
    task = args.task
    adapter_type = args.adapter.lower()

    meta = IMG_TASK_META[task]
    cfg = IMG_TASK_CONFIG.get(task, {"epochs": 20, "batch": 32, "lr": 1e-4})

    num_labels = meta["num_labels"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    batch = args.batch if args.batch else cfg["batch"]
    lr = args.learning_rate if args.learning_rate else cfg["lr"]

    # 데이터셋 로드
    if meta.get("source") == "torchvision":
        raw = load_torchvision_dataset(task, meta)
        split_train, split_val = "train", "test"
    else:
        # HuggingFace datasets
        if "subset" in meta:
            raw = load_dataset(meta["dataset_name"], meta["subset"])
        else:
            raw = load_dataset(meta["dataset_name"])
        split_train = meta["split_train"]
        split_val = meta["split_val"]

    # 모델 및 프로세서 로드
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    base = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)

    # 이미지 전처리
    def preprocess(examples):
        images = examples["image"]
        images = [img.convert("RGB") for img in images]
        inputs = processor(images, return_tensors="pt")
        # ViT에 필요한 pixel_values와 labels만 반환
        return {
            "pixel_values": inputs["pixel_values"],
            "labels": examples["label"]
        }

    # 데이터 전처리 (keep_in_memory=False로 디스크 캐시 사용하여 RAM 절약)
    train_ds = raw[split_train].map(
        preprocess,
        batched=True,
        remove_columns=raw[split_train].column_names,
        keep_in_memory=False,  # 디스크에 캐시하여 메모리 사용 줄이기
        load_from_cache_file=True,  # 캐시 활용
        batch_size=100,  # 작은 배치로 메모리 사용 줄이기
    )
    val_ds = raw[split_val].map(
        preprocess,
        batched=True,
        remove_columns=raw[split_val].column_names,
        keep_in_memory=False,
        load_from_cache_file=True,
        batch_size=100,
    )
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # AdaLoRA를 위한 total_step 계산
    total_step = (len(train_ds) // batch) * epochs

    # Adapter 적용
    if adapter_type == "bitfit":
        model = base
        for name, param in model.named_parameters():
            if "bias" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif adapter_type.lower() == "pissa":
        # PiSSA precompute 로직: SVD 계산 결과를 캐시하여 재사용
        peft_cfg = build_adapter(adapter_type, r=args.r, alpha=args.alpha, total_step=total_step)

        cache_dir = ".precomputed"
        os.makedirs(cache_dir, exist_ok=True)

        # ViT 모델명과 Rank를 조합해 고유 파일명 생성
        model_name_safe = "vit-base-patch16-224"
        cache_path = os.path.join(cache_dir, f"{model_name_safe}_r{args.r}.pt")

        if os.path.exists(cache_path):
            print(f"[*] Found precomputed PiSSA weights at {cache_path}. Loading...")
            # 캐시가 있으면 SVD 연산을 건너뜀
            peft_cfg.init_lora_weights = False
            model = get_peft_model(base, peft_cfg)

            # 저장된 PiSSA 가중치 로드
            checkpoint = torch.load(cache_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            print(f"[*] PiSSA initialization loaded from cache.")
        else:
            print(f"[*] No precomputed weights found. Computing PiSSA SVD (this may take a while)...")
            peft_cfg.init_lora_weights = "pissa"
            model = get_peft_model(base, peft_cfg)

            # 초기화된 가중치 저장 (lora_A, lora_B 및 수정된 base_layer)
            to_save = {}
            for name, param in model.named_parameters():
                if "lora_" in name or any(tm in name for tm in peft_cfg.target_modules):
                    if param.requires_grad or "base_layer" in name:
                        to_save[name] = param.cpu().detach()

            # base_layer 가중치도 저장 (PiSSA에서 수정됨)
            for name, module in model.named_modules():
                if hasattr(module, 'base_layer') and hasattr(module.base_layer, 'weight'):
                    to_save[f"{name}.base_layer.weight"] = module.base_layer.weight.cpu().detach()

            torch.save(to_save, cache_path)
            print(f"[*] PiSSA SVD computation finished and saved to {cache_path}")
    else:
        peft_cfg = build_adapter(adapter_type, r=args.r, alpha=args.alpha, total_step=total_step)
        model = get_peft_model(base, peft_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print("=" * 60)
    print(f"[CONFIG] Task: {task} | Adapter: {adapter_type}")
    print(f"[CONFIG] Seed: {args.seed} | Epochs: {epochs} | Batch: {batch} | LR: {lr}")
    print(f"[CONFIG] Rank: {args.r} | Alpha: {args.alpha}")
    if adapter_type == "lava":
        print(f"[CONFIG] Lambda VIB: {args.lambda_vib} | Stab: {args.lambda_stab} | Latent: {args.lambda_latent_stability}")
    print(f"[MODEL] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    print(f"[DATA] Train: {len(train_ds)} | Val: {len(val_ds)}")
    print("=" * 60)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.argmax(-1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    run_name = f"{adapter_type}_{task}_r{args.r}_s{args.seed}"
    wandb.init(project="ViT-ImageClassification", name=run_name, config=vars(args))

    tmp_dir = tempfile.mkdtemp()

    training_args = TrainingArguments(
        output_dir=tmp_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        report_to="wandb",
        seed=args.seed,
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=False,
        log_level="info",
        label_names=["labels"],  # compute_metrics 호출을 위해 명시적으로 설정
        dataloader_num_workers=4,  # 메모리 사용 줄이기 (multiprocessing 비활성화)
        dataloader_pin_memory=False,  # RAM 메모리 사용 줄이기
    )

    callback = BestMetricCallback()

    if adapter_type == "lava":
        trainer = LavaViTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        lambda_vib=args.lambda_vib,
        lambda_stab=args.lambda_stab,
        lambda_latent_stability=args.lambda_latent_stab  
    )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[callback],
        )

    trainer.train()

    best_acc = None
    for log in trainer.state.log_history:
        if "eval_accuracy" in log:
            val = log["eval_accuracy"]
            best_acc = val if best_acc is None else max(best_acc, val)

    if best_acc is not None:
        wandb.run.summary["best_accuracy"] = best_acc

    wandb.finish()

    # 결과 저장
    result_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(result_dir, exist_ok=True)

    result_file = os.path.join(
        result_dir,
        f"img_result_{adapter_type}_{task}_r{args.r}_s{args.seed}.json"
    )

    with open(result_file, "w") as f:
        json.dump({
            "task": task,
            "seed": args.seed,
            "adapter": adapter_type,
            "best_accuracy": best_acc if best_acc else 0.0,
            "lambda_vib": args.lambda_vib,
            "lambda_stab": args.lambda_stab,
            "lambda_latent_stability": args.lambda_latent_stability,
        }, f, indent=2)

    print("=" * 60)
    if best_acc is not None:
        print(f"[RESULT] Task: {task} | Adapter: {adapter_type}")
        print(f"[RESULT] Best Accuracy: {best_acc:.4f}")
    else:
        print(f"[RESULT] No valid metric")
    print(f"[RESULT] Saved to: {result_file}")
    print("=" * 60)

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Task & Model
    parser.add_argument("--task", type=str, required=True, choices=list(IMG_TASK_META.keys()))
    parser.add_argument("--adapter", type=str, required=True)

    # General Training Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler", type=str, default="linear",
                        choices=["linear", "cosine", "constant"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA Parameters
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # LAVA Specific Parameters
    parser.add_argument("--lambda_vib", type=float, default=1.0, help="VIB loss weight")
    parser.add_argument("--lambda_stab", type=float, default=0.1, help="Logit stability weight")
    parser.add_argument("--lambda_latent_stability", type=float, default=1.0, help="Latent stability weight")
    parser.add_argument("--latent_dim", type=int, default=16, help="LAVA latent dimension")
    parser.add_argument("--kl_annealing", action="store_true", help="Enable KL annealing")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="LAVA noise scale")

    args = parser.parse_args()
    setup_seed(args.seed)
    main(args)
