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
from evaluate import load as load_metric
import wandb
import os
import json
import tempfile
from peft import get_peft_model, LoraConfig, AdaLoraConfig
from peft.tuners.lava.config import LavaConfig

from configs.task_config import (
    GLUE_META,
    PISSA_TASK_CONFIG,
    DORA_TASK_CONFIG,
    LORA_TASK_CONFIG,
    MOCA_TASK_CONFIG,
    LAVA_TASK_CONFIG,
    BITFIT_TASK_CONFIG,
    ADALORA_TASK_CONFIG,
)

from trainer import (
    LavaNLUTrainer,
    setup_seed,
    register_lava,
    BestMetricCallback,
    print_trainable_parameters,
)

# LAVA 등록
register_lava()

# ==========================================================
# MaxEnt LAVA Trainer (🔥 CLEAN: FIXED LAMBDA, NO CONSTRAINT)
# ==========================================================

# ==========================================================
# Adapter builder (MODIFIED: Added AdaLoRA, BitFit)
# ==========================================================
def build_adapter(adapter_type, r, alpha, model=None, total_step=None, lora_dropout=0.0):
    at = adapter_type.lower()

    # 1. LoRA 계열 (LoRA, DoRA, PiSSA)
    if at in ["lora", "dora", "pissa"]:
        kwargs = dict(
            r=r,
            lora_alpha=alpha,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            task_type="SEQ_CLS",
            lora_dropout=lora_dropout,
        )
        if at == "pissa":
            kwargs["init_lora_weights"] = "pissa"
        if at == "dora":
            kwargs["use_dora"] = True
        return LoraConfig(**kwargs)

    # 2. AdaLoRA
    if at == "adalora":
        return AdaLoraConfig(
            init_r=r,
            target_r=r // 2,
            lora_alpha=alpha,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            task_type="SEQ_CLS",
            total_step=total_step if total_step else 1000,
            lora_dropout=lora_dropout,
        )

    # 3. LAVA
    if at in ["lava", "lava_init"]:
        return LavaConfig(
            r=r,
            alpha=alpha,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            task_type="SEQ_CLS",
        )

    # 4. BitFit
    if at == "bitfit":
        return "bitfit"

    raise ValueError(f"Unknown adapter type: {adapter_type}")


# ==========================================================
# MAIN
# ==========================================================
def main(args):
    task = args.task
    adapter_type = args.adapter
    
    meta = GLUE_META[task]
    num_labels = meta["num_labels"]
    main_metric = meta["main"]
    eval_key = meta["eval_key"]
    
    raw = load_dataset("nyu-mll/glue", task)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess(batch):
        return tokenizer(
            batch[meta["s1"]],
            batch[meta["s2"]] if meta["s2"] else None,
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    encoded = raw.map(
        preprocess,
        batched=True,
        keep_in_memory=True,  # 디스크 캐시 생성 방지
        load_from_cache_file=False  # 기존 캐시 무시
    )
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # Apply train_data_ratio (use first N% for reproducibility)
    original_train_size = len(encoded["train"])
    if args.train_data_ratio < 100:
        subset_size = int(original_train_size * args.train_data_ratio / 100)
        subset_size = max(1, subset_size)  # At least 1 sample
        encoded["train"] = encoded["train"].select(range(subset_size))
        print(f"[*] Using {args.train_data_ratio}% of training data: {subset_size}/{original_train_size} samples")

    total_train_samples = len(encoded["train"])

    base = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
    )

    # Config 매핑
    at = adapter_type.lower()
    config_map = {
        "pissa": PISSA_TASK_CONFIG,
        "dora": DORA_TASK_CONFIG,
        "lora": LORA_TASK_CONFIG,
        "lava": LAVA_TASK_CONFIG,
        "adalora": ADALORA_TASK_CONFIG,
        "bitfit": BITFIT_TASK_CONFIG
    }
    
    task_configs = config_map.get(at, LORA_TASK_CONFIG)
    cfg = task_configs.get(task, task_configs.get("default", {"epochs": 3, "lr": 2e-4, "batch": 32}))

    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    lr = args.learning_rate if args.learning_rate is not None else cfg.get("lr", 5e-4)
    batch = args.batch if args.batch is not None else cfg.get("batch", 32)
    alpha = args.alpha if args.alpha is not None else cfg.get("alpha", args.r * 2)

    if args.alpha is not None:
        # 사용자가 명시적으로 --alpha를 준 경우
        final_alpha = args.alpha
    elif at in ["lava", "lava_init"]:
        # LAVA일 경우 r=8, alpha=4 기준 sqrt scaling 적용
        # 공식: alpha = 4 * sqrt(r / 8)
        final_alpha = 4 * math.sqrt(args.r / 8)
        print(f"[*] LAVA Optimal Alpha calculated: {final_alpha:.2f} (r={args.r})")
    else:
        # 일반 LoRA 계열은 기존 방식(cfg 또는 r*2) 유지
        final_alpha = cfg.get("alpha", args.r)

    # AdaLoRA용 total_step 계산
    total_step = (total_train_samples // batch) * epochs

    # Adapter 적용
    peft_cfg = build_adapter(adapter_type, r=args.r, alpha=final_alpha, total_step=total_step, lora_dropout=args.lora_dropout)
    
    
    
        
    if at == "pissa":
        cache_dir = ".precomputed"
        os.makedirs(cache_dir, exist_ok=True)
        
        # 모델명과 Rank를 조합해 고유 파일명 생성
        model_name_safe = args.model.replace("/", "_")
        cache_path = os.path.join(cache_dir, f"{model_name_safe}_r{args.r}.pt")

        if os.path.exists(cache_path):
            print(f"[*] Found precomputed PiSSA weights at {cache_path}. Loading...")
            # 캐시가 있으면 SVD 연산을 건너뜀
            peft_cfg.init_lora_weights = False
            model = get_peft_model(base, peft_cfg)
            
            # 저장된 PiSSA 가중치(A, B 및 수정된 base) 주입
            checkpoint = torch.load(cache_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            print(f"[*] PiSSA initialization skipped and weights loaded from cache.")
        else:
            print(f"[*] No precomputed weights found. Computing PiSSA SVD (this may take a while)...")
            peft_cfg.init_lora_weights = "pissa"
            model = get_peft_model(base, peft_cfg)
            
            # 초기화된 가중치 저장
            to_save = {}
            for name, param in model.named_parameters():
                if "lora_" in name or "pissa" in name or any(tm in name for tm in peft_cfg.target_modules):
                    if param.requires_grad or "base_layer" in name:
                        to_save[name] = param.cpu().detach()
            
            torch.save(to_save, cache_path)
            print(f"[*] PiSSA SVD computation finished and saved to {cache_path}")

    elif adapter_type.lower() == "bitfit":
        # BitFit 전용 로직
        model = base
        for name, param in model.named_parameters():
            if "bias" in name or "classifier" in name or "pooler" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("[*] BitFit adapter applied.")

    else:
        # LAVA, LoRA, DoRA 등 일반적인 어댑터 생성
        model = get_peft_model(base, peft_cfg)
        print(f"[*] {adapter_type.upper()} adapter applied.") 
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    frozen_params = all_params - trainable_params
    trainable_percentage = 100 * trainable_params / all_params

    # Count adapter modules and calculate params per adapter
    num_adapter_modules = 0
    adapter_only_params = 0
    for name, module in model.named_modules():
        # PEFT adapter layers (LoRA, LAVA, etc.)
        if hasattr(module, 'lora_A') or hasattr(module, 'W_mu'):
            num_adapter_modules += 1
            # Count params in this adapter module
            for p in module.parameters():
                if p.requires_grad:
                    adapter_only_params += p.numel()

    params_per_adapter = adapter_only_params / num_adapter_modules if num_adapter_modules > 0 else 0

    print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {trainable_percentage:.4f}")
    print(f"adapter modules: {num_adapter_modules} || params per adapter: {params_per_adapter:,.0f}")
    
    metric = load_metric("glue", task)
    

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if num_labels == 1:
            preds = preds.squeeze()
        else:
            preds = preds.argmax(-1)
        
        out = metric.compute(predictions=preds, references=labels)
        out["main"] = out[main_metric]
        return out
    
    
    run_name = (
        f"{adapter_type}_{task}_"
        f"r{args.r}_a{final_alpha:.1f}_"
        f"vb{args.lambda_vib}_"
        f"ls{args.lambda_latent_stability}_"
        f"s{args.seed}"
    )
    
    # Wandb 초기화 (조건부)
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
        # Parameter metrics
        wandb.run.summary["trainable_params"] = trainable_params
        wandb.run.summary["all_params"] = all_params
        wandb.run.summary["frozen_params"] = frozen_params
        wandb.run.summary["trainable_percentage"] = trainable_percentage
        wandb.run.summary["num_adapter_modules"] = num_adapter_modules
        wandb.run.summary["adapter_only_params"] = adapter_only_params
        wandb.run.summary["params_per_adapter"] = params_per_adapter
        # Data metrics
        wandb.run.summary["total_train_samples"] = total_train_samples
        wandb.run.summary["original_train_size"] = original_train_size
        wandb.run.summary["train_data_ratio"] = args.train_data_ratio
    else:
        # wandb를 비활성화하는 경우 offline 모드로 설정
        os.environ["WANDB_MODE"] = "offline"

    best_callback = BestMetricCallback(main_metric)

    # 임시 디렉토리 사용 (폴더 생성 방지)
    tmp_dir = tempfile.mkdtemp()

    args_out = TrainingArguments(
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
        report_to="wandb" if not args.no_wandb else "none",
        seed=args.seed,
        logging_steps=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Use LavaTrainer only when LAVA losses are active (lambda > 0)
    # When lambda_vib=0 and lambda_latent_stability=0, use standard Trainer for fair comparison
    use_lava_trainer = (
        at in ["lava", "lava_init"] and
        (args.lambda_vib > 0 or args.lambda_latent_stability > 0)
    )

    if use_lava_trainer:
        trainer = LavaNLUTrainer(
            model=model,
            args=args_out,
            train_dataset=encoded["train"],
            eval_dataset=encoded[eval_key],
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            callbacks=[best_callback],
            lambda_vib=args.lambda_vib,
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

    if best_main is not None and not args.no_wandb:
        wandb.run.summary[f"best_{main_metric}"] = best_main

    if not args.no_wandb:
        wandb.finish()

    # 결과를 JSON 파일로 저장 (민감도 분석용)
    result_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(result_dir, exist_ok=True)

    result_file = os.path.join(
        result_dir,
        f"result_{task}_s{args.seed}_vib{args.lambda_vib}_lat{args.lambda_latent_stability}.json"
    )

    result_data = {
        "task": task,
        "seed": args.seed,
        "lambda_vib": args.lambda_vib,
        "lambda_latent_stability": args.lambda_latent_stability,
        "best_metric": best_main if best_main is not None else 0.0,
        "metric_name": main_metric
    }

    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n[RESULT] Task: {task}, Best {main_metric}: {best_main:.4f}" if best_main else f"\n[RESULT] Task: {task}, No valid metric")

    # 임시 디렉토리 정리
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    return best_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Task & Model
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")

    # General Training Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler", type=str, default="linear",
                        choices=["linear", "cosine", "constant"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA Parameters
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # LAVA Specific Parameters
    parser.add_argument("--lambda_vib", type=float, default=1.0, help="VIB loss weight")
    parser.add_argument("--lambda_latent_stability", type=float, default=1.0, help="Latent stability weight")
    parser.add_argument("--latent_dim", type=int, default=16, help="LAVA latent dimension")
    parser.add_argument("--kl_annealing", action="store_true", help="Enable KL annealing")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="LAVA noise scale")
    
    # Wandb Parameters
    parser.add_argument("--wandb_project", type=str, default="Deberta-NaturalLanguageUnderstanding", help="Wandb project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    # Data Ratio Parameter
    parser.add_argument("--train_data_ratio", type=int, default=100,
                        help="Percentage of training data to use (1-100). Uses first N%% for reproducibility.")

    args = parser.parse_args()
    setup_seed(args.seed)
    main(args)