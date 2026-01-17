import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
import math
import random
import numpy as np
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

import torch.nn.functional as F

from typing import Dict
import peft.utils.save_and_load
import peft.mapping
from peft.utils.peft_types import PeftType
from peft.tuners.lava.config import LavaConfig
from peft.tuners.lava.model import LavaModel
# ----------------------------------------------------------
# SEED SETUP (UNCHANGED)
# ----------------------------------------------------------
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# 1. PeftType 에 LAVA 열거형 추가
if not hasattr(PeftType, "LAVA"):
    PeftType.LAVA = "LAVA"

# 2. PEFT 내부 매핑 테이블에 LAVA 등록
# 이 과정이 없으면 get_peft_model이 'LAVA' 키를 찾지 못해 KeyError가 발생합니다.
for lava_key in ["LAVA", PeftType.LAVA]:
    peft.mapping.PEFT_TYPE_TO_CONFIG_MAPPING[lava_key] = LavaConfig
    peft.mapping.PEFT_TYPE_TO_TUNER_MAPPING[lava_key] = LavaModel
    
    # 저장 및 로드를 위한 프리픽스 설정
    peft.utils.save_and_load.PEFT_TYPE_TO_PREFIX_MAPPING[lava_key] = "adapter_model"
    peft.mapping.PEFT_TYPE_TO_PREFIX_MAPPING[lava_key] = "adapter_model"

print("✅ LAVA 시스템이 PEFT 매핑에 성공적으로 등록되었습니다.")


def print_trainable_parameters(model):
    """
    모델의 전체 파라미터 대비 학습 가능한 파라미터의 수와 비율을 출력합니다.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    print(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
    )
    
    
class BestMetricCallback(TrainerCallback):
    def __init__(self, main_metric):
        self.main_metric = f"eval_{main_metric}"
        self.best_score = -float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and self.main_metric in metrics:
            current_score = metrics[self.main_metric]
            if current_score > self.best_score:
                self.best_score = current_score
            # WandB에 현재까지의 Best 점수 기록
            wandb.log({"eval/best_main": self.best_score}, step=state.global_step)

# ==========================================================
# MaxEnt LAVA Trainer (🔥 CLEAN: FIXED LAMBDA, NO CONSTRAINT)
# ==========================================================

class StabilityLavaTrainer(Trainer):
    def __init__(self, *args, lambda_vib=0.1, lambda_latent_stability=0.1,lambda_stab=0.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.lambda_vib = lambda_vib
            self.lambda_latent_stability = lambda_latent_stability
            self.lambda_stab = lambda_stab
            
            # Logit Consistency 관련 키 제거
            self.loss_track = {
                "ce_loss": 0,
                "raw_vib_loss": 0,
                "weighted_vib_loss": 0,
                "raw_latent_stab_loss": 0,
                "weighted_latent_stab_loss": 0
            }

    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. 평가(Evaluation) 모드일 때는 기본 Trainer의 loss 계산 방식을 따름
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs)

        # 2. 데이터 준비 (전체 배치 N을 복제하여 2N 생성)
        labels = inputs["labels"]
        # 모든 텐서 입력을 배치 차원(dim=0)으로 두 번 이어 붙임
        concat_inputs = {k: torch.cat([v, v], dim=0) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # 3. Forward Pass (2N 배치 처리)
        outputs = model(**concat_inputs)
        logits = outputs.logits 
        
        # 4. 결과 분할 (원본 몫과 복제본 몫)
        # logits1: 원본 데이터 N개에 대한 결과
        # logits2: 복제된 데이터 N개에 대한 결과 (LAVA 노이즈로 인해 미세하게 다름)
        logits1, logits2 = logits.chunk(2, dim=0)
        
        # 5. Task Loss 계산 (2N 전체에 대해 수행)
        # 라벨도 2N으로 불려서 전체 출력에 대해 정답을 맞히도록 유도
        full_labels = torch.cat([labels, labels], dim=0)
        if labels.dtype in [torch.float32, torch.float64]:
            loss_fct = torch.nn.MSELoss()
            ce_loss = loss_fct(logits.view(-1), full_labels.view(-1))
        else:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), full_labels.view(-1))

        # 6. Stability Loss (Logit Consistency)
        # 동일한 입력에 대해 두 출력이 최대한 같아지도록 제약 (핵심 정규화)
        if labels.dtype in [torch.float32, torch.float64]:
            const_loss = F.mse_loss(logits1, logits2)
        else:
            p = F.log_softmax(logits1, dim=-1)
            q = F.softmax(logits2, dim=-1)
            p_rev = F.log_softmax(logits2, dim=-1)
            q_rev = F.softmax(logits1, dim=-1)
            # 양방향 KL Divergence의 평균 사용
            const_loss = (F.kl_div(p, q, reduction='batchmean') + 
                        F.kl_div(p_rev, q_rev, reduction='batchmean')) / 2

        # 7. LAVA 고유 Loss 수집 (VIB & Latent Stability)
        kl_divs = []
        latent_stabs = []
        for m in model.modules():
            # VIB Loss: 잠재 공간의 분포를 표준 정규분포에 가깝게 (정보 압축)
            if hasattr(m, "_last_mu") and m._last_mu is not None:
                # 2N 데이터 중 앞부분(N개)의 통계치만 사용하여 KL 계산 (효율성)
                mu, logvar = m._last_mu.chunk(2)[0], m._last_logvar.chunk(2)[0]
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
                kl_divs.append(kl)
                # 메모리 누수 방지를 위해 초기화
                m._last_mu, m._last_logvar = None, None

            # Latent Stability: 내부 레이어 표현의 일관성
            if hasattr(m, "_latent_stability") and m._latent_stability is not None:
                latent_stabs.append(m._latent_stability)
                m._latent_stability = None

        vib_loss = torch.stack(kl_divs).mean() if kl_divs else torch.tensor(0.0).to(ce_loss.device)
        latent_stab_loss = torch.stack(latent_stabs).mean() if latent_stabs else torch.tensor(0.0).to(ce_loss.device)

        # 8. 가중 합산 및 최종 Loss 산출
        w_const = self.lambda_stab * const_loss
        w_vib = self.lambda_vib * vib_loss
        w_latent = self.lambda_latent_stability * latent_stab_loss

        loss = ce_loss + w_const + w_vib + w_latent
        
        # 9. 실시간 로깅을 위한 값 업데이트
        self.loss_track.update({
            "ce_loss": ce_loss.item(),
            "raw_const_loss": const_loss.item(),
            "weighted_const_loss": w_const.item(),
            "raw_vib_loss": vib_loss.item(),
            "weighted_vib_loss": w_vib.item(),
            "raw_latent_stab_loss": latent_stab_loss.item(),
            "weighted_latent_stab_loss": w_latent.item()
        })

        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float]) -> None:
        logs["train/ce_loss"] = self.loss_track["ce_loss"]
        logs["train/vib_raw"] = self.loss_track["raw_vib_loss"]
        logs["train/vib_weighted"] = self.loss_track["weighted_vib_loss"]
        logs["train/latent_stab_raw"] = self.loss_track["raw_latent_stab_loss"]
        logs["train/latent_stab_weighted"] = self.loss_track["weighted_latent_stab_loss"]
        
        super().log(logs)
# ==========================================================
# Adapter builder (MODIFIED: Added AdaLoRA, BitFit)
# ==========================================================
def build_adapter(adapter_type, r, alpha, model=None):
    at = adapter_type.lower()

    # 1. LoRA 계열 (LoRA, DoRA, PiSSA)
    if at in ["lora", "dora", "pissa"]:
        kwargs = dict(
            r=r,
            lora_alpha=alpha,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            task_type="SEQ_CLS",
        )
        if at == "pissa":
            kwargs["init_lora_weights"] = "pissa"
        if at == "dora":
            kwargs["use_dora"] = True
        return LoraConfig(**kwargs)

    # 2. AdaLoRA
    if at == "adalora":
        return AdaLoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            task_type="SEQ_CLS",
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
    
    raw = load_dataset("glue", task)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess(batch):
        return tokenizer(
            batch[meta["s1"]],
            batch[meta["s2"]] if meta["s2"] else None,
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    encoded = raw.map(preprocess, batched=True)
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

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


    # Adapter 적용
    peft_cfg = build_adapter(adapter_type, r=args.r, alpha=final_alpha)
    
    
    
        
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
    trainable_percentage = 100 * trainable_params / all_params   
    print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {trainable_percentage:.4f}")
    
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
        f"vb{args.lambda_vib}_st{args.lambda_stab}_"
        f"ls{args.lambda_latent_stability}_"
        f"s{args.seed}"
    )
    
    wandb.init(
        project="Deberta-NaturalLanguageUnderstanding",
        name=run_name,
        config=vars(args)
    )
    wandb.run.summary["trainable_params"] = trainable_params
    wandb.run.summary["all_params"] = all_params
    wandb.run.summary["trainable_percentage"] = trainable_percentage

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
        report_to="wandb",
        seed=args.seed,
        logging_steps=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if at in ["lava", "lava_init"]:
        trainer = StabilityLavaTrainer(
            model=model,
            args=args_out,
            train_dataset=encoded["train"],
            eval_dataset=encoded[eval_key],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            lambda_vib=args.lambda_vib,
            lambda_stab=args.lambda_stab,
            lambda_latent_stability=args.lambda_latent_stability,
            callbacks=[best_callback],
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

    wandb.finish()

    # 결과를 JSON 파일로 저장 (민감도 분석용)
    result_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(result_dir, exist_ok=True)

    result_file = os.path.join(
        result_dir,
        f"result_{task}_s{args.seed}_vib{args.lambda_vib}_stab{args.lambda_stab}_lat{args.lambda_latent_stability}.json"
    )

    result_data = {
        "task": task,
        "seed": args.seed,
        "lambda_vib": args.lambda_vib,
        "lambda_stab": args.lambda_stab,
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
    parser.add_argument("--lambda_stab", type=float, default=0.1, help="Logit stability weight")
    parser.add_argument("--lambda_latent_stability", type=float, default=1.0, help="Latent stability weight")
    parser.add_argument("--latent_dim", type=int, default=16, help="LAVA latent dimension")
    parser.add_argument("--kl_annealing", action="store_true", help="Enable KL annealing")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="LAVA noise scale")

    args = parser.parse_args()
    setup_seed(args.seed)
    main(args)