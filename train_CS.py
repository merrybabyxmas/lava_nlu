# train_commonsense.py
import os
import sys
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import math
import wandb

import json
import csv
from datetime import datetime

import tempfile
from peft import get_peft_model, LoraConfig, AdaLoraConfig
from peft.tuners.lava.config import LavaConfig

from trainer import (
    LavaBaseTrainer,
    setup_seed,
    register_lava,
    BestMetricCallback,
    print_trainable_parameters,
)

# LAVA 등록
register_lava()


# ============================================================
# 데이터셋 설정
# ============================================================
COMMONSENSE_META = {
    "piqa": {
        "dataset_name": "ybisk/piqa",
        "dataset_config": None,
        "num_labels": 2,
        "label_col": "label",
        "question_col": "goal",
        "choices_cols": ["sol1", "sol2"],
        "main": "accuracy",
        "eval_key": "validation"
    },
    "siqa": {
        "dataset_name": "social_i_qa",
        "dataset_config": None,
        "num_labels": 3,
        "label_col": "label",
        "context_col": "context",
        "question_col": "question",
        "choices_cols": ["answerA", "answerB", "answerC"],
        "main": "accuracy",
        "eval_key": "validation"
    },
    "arc_easy": {
        "dataset_name": "allenai/ai2_arc",
        "dataset_config": "ARC-Easy",
        "num_labels": 4,
        "label_col": "answerKey",
        "question_col": "question",
        "choices_col": "choices",
        "main": "accuracy",
        "eval_key": "validation"
    },
    "arc_challenge": {
        "dataset_name": "allenai/ai2_arc",
        "dataset_config": "ARC-Challenge",
        "num_labels": 4,
        "label_col": "answerKey",
        "question_col": "question",
        "choices_col": "choices",
        "main": "accuracy",
        "eval_key": "validation"
    },
    "openbookqa": {
        "dataset_name": "allenai/openbookqa",
        "dataset_config": "main",
        "num_labels": 4,
        "label_col": "answerKey",
        "question_col": "question_stem",
        "choices_col": "choices",
        "main": "accuracy",
        "eval_key": "validation"
    },
    "hellaswag": {
        "dataset_name": "Rowan/hellaswag",
        "dataset_config": None,
        "num_labels": 4,
        "label_col": "label",
        "context_col": "ctx",
        "choices_col": "endings",
        "main": "accuracy",
        "eval_key": "validation"
    },
    "winogrande": {
        "dataset_name": "allenai/winogrande",
        "dataset_config": "winogrande_xl",
        "num_labels": 2,
        "label_col": "answer",
        "question_col": "sentence",
        "choices_cols": ["option1", "option2"],
        "main": "accuracy",
        "eval_key": "validation"
    }
}

# 태스크별 하이퍼파라미터
LORA_COMMONSENSE_CONFIG = {
    "piqa": {"epochs": 3, "lr": 3e-4, "batch": 16, "alpha": 8},
    "siqa": {"epochs": 3, "lr": 3e-4, "batch": 16, "alpha": 8},
    "arc_easy": {"epochs": 5, "lr": 3e-4, "batch": 16, "alpha": 8},
    "arc_challenge": {"epochs": 5, "lr": 3e-4, "batch": 16, "alpha": 8},
    "openbookqa": {"epochs": 5, "lr": 3e-4, "batch": 16, "alpha": 8},
    "hellaswag": {"epochs": 3, "lr": 3e-4, "batch": 16, "alpha": 8},
    "winogrande": {"epochs": 3, "lr": 3e-4, "batch": 16, "alpha": 8}
}

LAVA_COMMONSENSE_CONFIG = LORA_COMMONSENSE_CONFIG
PISSA_COMMONSENSE_CONFIG = LORA_COMMONSENSE_CONFIG
DORA_COMMONSENSE_CONFIG = LORA_COMMONSENSE_CONFIG
ADALORA_COMMONSENSE_CONFIG = LORA_COMMONSENSE_CONFIG
BITFIT_COMMONSENSE_CONFIG = LORA_COMMONSENSE_CONFIG

def get_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        import numpy as np
        import random
        worker_seed = (seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn


# ==========================================================
# Adapter builder
# ==========================================================
def build_adapter(adapter_type, r, alpha, model_name=None,total_step=None):
    at = adapter_type.lower()
    
    # Llama-3용 target modules
    if "llama" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # DeBERTa용 target modules
    else:
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]

    # 1. LoRA 계열
    if at in ["lora", "dora", "pissa"]:
        kwargs = dict(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            task_type="SEQ_CLS",
        )
        if at == "pissa":
            kwargs["init_lora_weights"] = "pissa"
        if at == "dora":
            kwargs["use_dora"] = True
        return LoraConfig(**kwargs)

    # 2. AdaLoRA
    if at == "adalora":
        if total_step is None or total_step <= 0:
            raise ValueError("AdaLoRA requires total_step > 0")

        return AdaLoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            task_type="SEQ_CLS",
            total_step=total_step,
        )

    # 3. LAVA
    if at in ["lava", "lava_init"]:
        return LavaConfig(
            r=r,
            alpha=alpha,
            target_modules=target_modules,
            task_type="SEQ_CLS",
        )

    # 4. BitFit
    if at == "bitfit":
        return "bitfit"

    raise ValueError(f"Unknown adapter type: {adapter_type}")


# train_commonsense.py의 preprocess_commonsense 완전 재작성

def preprocess_commonsense(examples, meta, tokenizer, task, max_length=256):
    """
    Commonsense multiple-choice preprocessing (task-safe version)
    - task별 label 스키마 엄격 적용
    - 정답 없는 샘플 drop
    - clamping 제거
    """
    batch_size = len(examples[list(examples.keys())[0]])
    processed = {"input_ids": [], "attention_mask": [], "labels": []}

    label_col = meta["label_col"]
    num_labels = meta["num_labels"]

    # -------------------------
    # task별 label 파서
    # -------------------------
    def parse_label(label_raw):
        # PIQA: {0,1}, test/val = -1
        if task == "piqa":
            if label_raw == -1:
                return None
            return int(label_raw)

        # SIQA: {1,2,3} → {0,1,2}
        if task == "siqa":
            if label_raw == -1:
                return None
            return int(label_raw) - 1

        # ARC (easy / challenge): A–D
        if task.startswith("arc"):
            label_str = str(label_raw).strip().upper()
            if label_str not in ["A", "B", "C", "D"]:
                return None
            return ord(label_str) - ord("A")

        # WinoGrande: "1","2" → {0,1}
        if task == "winogrande":
            if label_raw not in ["1", "2"]:
                return None
            return int(label_raw) - 1

        # HellaSwag
        if task == "hellaswag":
            if label_raw == "" or label_raw == -1:
                return None
            return int(label_raw)
        
        # OpenBookQA: A–D
        if task == "openbookqa":
            label_str = str(label_raw).strip().upper()
            if label_str not in ["A", "B", "C", "D"]:
                return None
            return ord(label_str) - ord("A")

        raise ValueError(f"Unknown task: {task}")

    # -------------------------
    # main loop
    # -------------------------
    for i in range(batch_size):
        # ----- Question / Context -----
        if "question_col" in meta:
            question = examples[meta["question_col"]][i]
        elif "context_col" in meta:
            question = examples[meta["context_col"]][i]
        else:
            question = ""

        # Context + Question (SIQA)
        if "context_col" in meta and "question_col" in meta:
            context = examples[meta["context_col"]][i]
            question_text = examples[meta["question_col"]][i]
            question = f"{context} {question_text}"

        # ----- Choices -----
        if "choices_cols" in meta:
            choices = [examples[col][i] for col in meta["choices_cols"]]
        elif "choices_col" in meta:
            choices_dict = examples[meta["choices_col"]][i]
            if isinstance(choices_dict, dict) and "text" in choices_dict:
                choices = choices_dict["text"]
            elif isinstance(choices_dict, list):
                choices = choices_dict
            else:
                choices = []
        else:
            choices = []

        # ----- Label parsing (STRICT) -----
        label_raw = examples[label_col][i]
        label = parse_label(label_raw)

        # 정답 없는 샘플은 버림
        if label is None:
            continue

        # 최종 범위 체크 (안 맞으면 바로 에러)
        if not (0 <= label < num_labels):
            raise RuntimeError(
                f"[FATAL] Label out of range: {label} (raw={label_raw}, task={task})"
            )

        # ----- Tokenization -----
        if choices:
            text = question + " " + " ".join(str(c) for c in choices[:num_labels])
        else:
            text = question

        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        processed["input_ids"].append(encoded["input_ids"])
        processed["attention_mask"].append(encoded["attention_mask"])
        processed["labels"].append(label)

    return processed



# ==========================================================
# MAIN
# ==========================================================
def main(args):
    task = args.task
    adapter_type = args.adapter
    
    meta = COMMONSENSE_META[task]
    num_labels = meta["num_labels"]
    main_metric = meta["main"]
    eval_key = meta["eval_key"]
    dataset_name = meta["dataset_name"]
    dataset_config = meta.get("dataset_config", None)
    
    # 데이터셋 로드
    if dataset_config:
        raw = load_dataset(dataset_name, dataset_config)
    else:
        raw = load_dataset(dataset_name)
    # =========================
    # Remove invalid labels (-1)
    # =========================
    label_col = meta["label_col"]
    
    # Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Llama의 경우 pad_token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[*] Set pad_token to eos_token: {tokenizer.eos_token}")

    # 전처리
    def preprocess(batch):
        return preprocess_commonsense(batch, meta, tokenizer, task=args.task, max_length=args.max_length)
    
    from datasets import DatasetDict

    raw = DatasetDict({
        k: v for k, v in raw.items()
        if k != "test"
    })

    encoded = raw.map(
        preprocess,
        batched=True,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=raw["train"].column_names,
    )
    
    encoded.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # 모델 로드
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16,
    )
    
    # Llama의 경우 config 수정
    if tokenizer.pad_token_id is not None:
        base.config.pad_token_id = tokenizer.pad_token_id

    # Config 매핑
    at = adapter_type.lower()
    config_map = {
        "pissa": PISSA_COMMONSENSE_CONFIG,
        "dora": DORA_COMMONSENSE_CONFIG,
        "lora": LORA_COMMONSENSE_CONFIG,
        "lava": LAVA_COMMONSENSE_CONFIG,
        "adalora": ADALORA_COMMONSENSE_CONFIG,
        "bitfit": BITFIT_COMMONSENSE_CONFIG
    }
    
    task_configs = config_map.get(at, LORA_COMMONSENSE_CONFIG)
    cfg = task_configs.get(task, {"epochs": 5, "lr": 3e-4, "batch": 16, "alpha": 8})

    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    lr = args.learning_rate if args.learning_rate is not None else cfg.get("lr", 3e-4)
    batch = args.batch if args.batch is not None else cfg.get("batch", 16)

    if args.alpha is not None:
        final_alpha = args.alpha
    elif at in ["lava", "lava_init"]:
        final_alpha = 4 * math.sqrt(args.r / 8)
        print(f"[*] LAVA Optimal Alpha calculated: {final_alpha:.2f} (r={args.r})")
    else:
        final_alpha = cfg.get("alpha", args.r)

    # Adapter 적용
    peft_cfg = build_adapter(adapter_type, r=args.r, alpha=final_alpha, model_name=args.model)
    
    if at == "pissa":
        cache_dir = ".precomputed"
        os.makedirs(cache_dir, exist_ok=True)
        
        model_name_safe = args.model.replace("/", "_")
        cache_path = os.path.join(cache_dir, f"{model_name_safe}_r{args.r}.pt")

        if os.path.exists(cache_path):
            print(f"[*] Found precomputed PiSSA weights at {cache_path}. Loading...")
            peft_cfg.init_lora_weights = False
            model = get_peft_model(base, peft_cfg)
            checkpoint = torch.load(cache_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            print(f"[*] PiSSA initialization skipped and weights loaded from cache.")
        else:
            print(f"[*] No precomputed weights found. Computing PiSSA SVD (this may take a while)...")
            peft_cfg.init_lora_weights = "pissa"
            model = get_peft_model(base, peft_cfg)
            
            to_save = {}
            for name, param in model.named_parameters():
                if "lora_" in name or "pissa" in name or any(tm in name for tm in peft_cfg.target_modules):
                    if param.requires_grad or "base_layer" in name:
                        to_save[name] = param.cpu().detach()
            
            torch.save(to_save, cache_path)
            print(f"[*] PiSSA SVD computation finished and saved to {cache_path}")

    elif adapter_type.lower() == "bitfit":
        model = base
        for name, param in model.named_parameters():
            if "bias" in name or "classifier" in name or "pooler" in name or "score" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("[*] BitFit adapter applied.")

    else:
        model = get_peft_model(base, peft_cfg)
        print(f"[*] {adapter_type.upper()} adapter applied.")
    
    # ===== 모든 파라미터를 BF16으로 통일 =====
    model = model.to(torch.bfloat16)
    print(f"[*] Model converted to bfloat16")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / all_params
    print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {trainable_percentage:.4f}")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.argmax(-1)
        acc = (preds == labels).mean().item()
        return {"accuracy": acc, "main": acc}
    
    run_name = (
        f"{adapter_type}_{task}_"
        f"r{args.r}_a{final_alpha:.1f}_"
        f"vb{args.lambda_vib}_st{args.lambda_stab}_"
        f"ls{args.lambda_latent_stability}_"
        f"s{args.seed}"
    )
    
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args)
    )
    wandb.run.summary["trainable_params"] = trainable_params
    wandb.run.summary["all_params"] = all_params
    wandb.run.summary["trainable_percentage"] = trainable_percentage

    best_callback = BestMetricCallback(main_metric)

    tmp_dir = tempfile.mkdtemp()

    args_out = TrainingArguments(
        output_dir=tmp_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        report_to="wandb",
        seed=args.seed,
        logging_steps=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
    )

    if at in ["lava", "lava_init"]:
        trainer = LavaBaseTrainer(
            model=model,
            args=args_out,
            train_dataset=encoded["train"],
            eval_dataset=encoded[eval_key],
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            callbacks=[best_callback],
            lambda_vib=args.lambda_vib,
            #lambda_stab=args.lambda_stab,
            lambda_latent_stability=args.lambda_latent_stability,
        )
    else:
        trainer = Trainer(
            model=model,
            args=args_out,
            train_dataset=encoded["train"],
            eval_dataset=encoded[eval_key],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[best_callback],
        )

    trainer.train()
    
    best_main = None
    for log in trainer.state.log_history:
        if f"eval_{main_metric}" in log:
            val = log[f"eval_{main_metric}"]
            best_main = val if best_main is None else max(best_main, val)

    if best_main is not None:
        wandb.run.summary[f"best_{main_metric}"] = best_main

    # 결과 저장
    result_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(result_dir, exist_ok=True)

    result_file = os.path.join(
        result_dir,
        f"commonsense_result_{adapter_type}_{task}_r{args.r}_s{args.seed}.json"
    )

    result_data = {
        "task": task,
        "adapter": adapter_type,
        "seed": args.seed,
        "r": args.r,
        "alpha": final_alpha,
        "lambda_vib": args.lambda_vib,
        "lambda_stab": args.lambda_stab,
        "lambda_latent_stability": args.lambda_latent_stability,
        "best_accuracy": best_main if best_main is not None else 0.0,
        "metric_name": main_metric
    }

    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n[RESULT] Task: {task}, Best {main_metric}: {best_main:.4f}" if best_main else f"\n[RESULT] Task: {task}, No valid metric")

    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    return best_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Task & Model
    parser.add_argument("--task", type=str, required=True,
                        choices=["piqa", "siqa", "arc_easy", "arc_challenge", 
                                "hellaswag", "winogrande","openbookqa"])
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--max_length", type=int, default=256)

    # General Training Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--kl_annealing", action="store_true")
    parser.add_argument("--noise_scale", type=float, default=1.0)

    # LoRA Parameters
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # LAVA Specific Parameters
    parser.add_argument("--lambda_vib", type=float, default=1.0)
    parser.add_argument("--lambda_stab", type=float, default=0.1)
    parser.add_argument("--lambda_latent_stability", type=float, default=1.0)
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="Llama2-CommonsenseReasoning")

    args = parser.parse_args()
    setup_seed(args.seed)
    main(args)