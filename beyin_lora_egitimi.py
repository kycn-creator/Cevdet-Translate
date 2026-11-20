# -*- coding: utf-8 -*-
"""
Çapa AI - Faz 3 ("Beyin") Qwen-2.5 QLoRA Eğitim Betiği
=====================================================
RTX 5060 Laptop GPU (8 GB VRAM) için optimize edilmiş 4-bit NF4 QLoRA eğitimi.
Veri setleri:
  - beyin_train.jsonl (17.066 kayıt: 7.599 cümle çevirisi + 9.467 sözlük maddesi)
  - beyin_val.jsonl   (897 kayıt: 399 cümle çevirisi + 498 sözlük maddesi)
Canlı Takip:
  - beyin_egitimi.log + goz_egitimi.log (Açık olan terminal penceresine kesintisiz devam eder!)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = BASE_DIR / "beyin_train.jsonl"
DEFAULT_VAL = BASE_DIR / "beyin_val.jsonl"
DEFAULT_OUT = BASE_DIR / "Capa_Beyin_Qwen25_LoRA"
BEYIN_LOG_PATH = BASE_DIR / "beyin_egitimi.log"
GOZ_LOG_PATH = BASE_DIR / "goz_egitimi.log"


class SyncFileHandler(logging.FileHandler):
    """Her log satırında os.fsync çağırarak açık terminal/editör sekmesini anında yeniler."""

    def flush(self) -> None:
        super().flush()
        if self.stream and hasattr(self.stream, "fileno"):
            try:
                os.fsync(self.stream.fileno())
            except OSError:
                pass


log = logging.getLogger("capa_beyin")


class LiveBeyinCallback(TrainerCallback):
    """Terminalde ve log dosyasında Loss + Osmanlıca/Günümüz Türkçesi çeviri testini canlı gösterir."""

    def __init__(self, val_rows: list[dict], tokenizer, sample_every: int = 100):
        self.sentence_rows = [
            r for r in val_rows
            if len(r.get("messages", [])) >= 3
            and str(r["messages"][1].get("content", "")).startswith("Latin Osmanlıca Metin:")
        ] or val_rows
        self.tokenizer = tokenizer
        self.sample_every = sample_every
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or state.global_step == 0:
            return
        if "loss" in logs:
            elapsed = time.time() - self.t0
            steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
            rem_steps = max(0, state.max_steps - state.global_step)
            eta_min = (rem_steps / steps_per_sec) / 60 if steps_per_sec > 0 else 0
            vram_gb = torch.cuda.memory_allocated(0) / (1024**3)
            pct = 100.0 * state.global_step / max(1, state.max_steps)
            log.info(
                "[BEYİN] Adım %d/%d (%.1f%%) | Epoch %.2f | Loss: %.4f | LR: %.2e | VRAM: %.2f GB | Kalan: ~%.1f dk",
                state.global_step,
                state.max_steps,
                pct,
                state.epoch or 0.0,
                float(logs["loss"]),
                float(logs.get("learning_rate", 0.0)),
                vram_gb,
                eta_min,
            )
        if "eval_loss" in logs:
            log.info(
                "=== [BEYİN DOĞRULAMA - Adım %d] | Eval Loss: %.4f ===",
                state.global_step,
                float(logs["eval_loss"]),
            )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.sample_every != 0 or model is None:
            return
        model.eval()
        device = next(model.parameters()).device
        indices = [
            (state.global_step // self.sample_every + k) % len(self.sentence_rows)
            for k in range(2)
        ]
        log.info("--- [BEYİN CANLI ÇEVİRİ TESTİ - Adım %d] Osmanlıca <-> Gerçek Çeviri <-> Model Çevirisi ---", state.global_step)
        with torch.no_grad():
            for idx in indices:
                msgs = self.sentence_rows[idx]["messages"]
                prompt_msgs = msgs[:2]
                gt_text = str(msgs[2]["content"])
                src_text = str(msgs[1]["content"]).replace("Latin Osmanlıca Metin: ", "")
                prompt_str = str(
                    self.tokenizer.apply_chat_template(
                        prompt_msgs, tokenize=False, add_generation_prompt=True
                    )
                )
                inputs = self.tokenizer(prompt_str, return_tensors="pt").to(device)
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                input_len = int(inputs["input_ids"].shape[1])
                new_tokens = gen_ids[0][input_len:]
                pred_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                log.info(
                    "  Osmanlıca : %s\n  Gerçek    : %s\n  Model     : %s",
                    src_text,
                    gt_text,
                    pred_text,
                )
        model.train()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Çapa AI - Faz 3 Beyin QLoRA Eğitimi")
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Temel model (8GB VRAM için Qwen/Qwen2.5-3B-Instruct)",
    )
    p.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val-file", type=Path, default=DEFAULT_VAL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--resume", action="store_true", help="Son checkpoint'ten devam et")
    return p.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            SyncFileHandler(BEYIN_LOG_PATH, mode="w", encoding="utf-8"),
            SyncFileHandler(GOZ_LOG_PATH, mode="a", encoding="utf-8"),
        ],
    )
    for noisy in ("httpx", "urllib3", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    args = parse_args()
    assert torch.cuda.is_available(), "CUDA destekli GPU bulunamadı!"
    device_name = torch.cuda.get_device_name(0)
    bf16_ok = torch.cuda.is_bf16_supported()
    log.info("=========================================================================")
    log.info("FAZ 3: BEYİN (Qwen2.5-3B-Instruct 4-Bit QLoRA) EĞİTİMİ BAŞLIYOR!")
    log.info("GPU: %s | BF16: %s", device_name, bf16_ok)

    train_rows = [
        json.loads(line)
        for line in args.train_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    val_rows = [
        json.loads(line)
        for line in args.val_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    log.info(
        "Beyin Veri Seti -> Eğitim: %d (%s) | Doğrulama: %d (%s)",
        len(train_rows), args.train_file.name, len(val_rows), args.val_file.name,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16 if bf16_ok else torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=300,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=2,
        load_best_model_at_end=True,
        max_length=args.max_seq_len,
        report_to="none",
    )

    live_cb = LiveBeyinCallback(val_rows=val_rows, tokenizer=tokenizer, sample_every=100)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[live_cb],
    )

    if trainer.model is not None:
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        all_param = sum(p.numel() for p in trainer.model.parameters())
        log.info(
            "Beyin Modeli Hazır: %s | Eğitilebilir LoRA: %.2fM / Toplam: %.1fM (%.2f%%)",
            args.model, trainable_params / 1e6, all_param / 1e6, 100 * trainable_params / max(1, all_param),
        )

    trainer.train(resume_from_checkpoint=args.resume)

    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info("BEYİN EĞİTİMİ TAMAMLANDI! LoRA adaptörü kaydedildi: %s", final_dir)


if __name__ == "__main__":
    main()
