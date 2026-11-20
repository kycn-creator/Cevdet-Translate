# -*- coding: utf-8 -*-
"""
Çapa AI - Faz 2 ("Göz") Osmanlıca OCR / HTR Eğitim Betiği
=========================================================
Mimari           : Tam microsoft/trocr-base-stage1 (Önceden eğitilmiş ViT Encoder +
                   684M satırda eğitilmiş hazır TrOCR Decoder ve Cross-Attention)
Yön Hizalaması   : Osmanlıca satır görselleri (Sağdan Sola) yatay aynalanarak (ImageOps.mirror)
                   Soldan Sağa Latin harf sırasıyla ve TrOCR Cross-Attention yönüyle
                   birebir (1-e-1) hizalanır.
Veri Seti        : Goz_Egitim_Seti/goz_train.jsonl (7.616) & goz_val.jsonl (400)
Metrik           : jiwer Character Error Rate (CER), Harf Doğruluğu (%) ve WER
Donanım          : NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM, BF16)
Canlı Takip      : Terminal + goz_egitimi.log (Her 20 adımda Loss/VRAM, her 100 adımda
                   Gerçek Metin <-> Modelin Okuduğu Metin karşılaştırması)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import jiwer
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Goz_Egitim_Seti"
TRAIN_PATH = DATA_DIR / "goz_train.jsonl"
VAL_PATH = DATA_DIR / "goz_val.jsonl"
DEFAULT_OUT = BASE_DIR / "Capa_Goz_TrOCR"
LOG_PATH = BASE_DIR / "goz_egitimi.log"

TROCR_MODEL = "microsoft/trocr-base-stage1"


class SyncFileHandler(logging.FileHandler):
    """Her log satırında os.fsync çağırarak açık editör sekmesini anında yeniler."""

    def flush(self) -> None:
        super().flush()
        if self.stream and hasattr(self.stream, "fileno"):
            try:
                os.fsync(self.stream.fileno())
            except OSError:
                pass


log = logging.getLogger("capa_goz")


class OttomanOCRDataset(Dataset):
    def __init__(
        self,
        items: list[dict],
        root_dir: Path,
        processor: TrOCRProcessor,
        max_target_length: int = 128,
    ):
        self.items = items
        self.root_dir = root_dir
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.items[idx]
        img_path = self.root_dir / entry["file_name"]
        image = Image.open(img_path).convert("RGB")
        # ÖNEMLİ 1: Osmanlıca satırlar sağdan sola yazıldığı için yatay aynalama (mirror)
        # uyguluyoruz. Böylece cümlenin ilk kelimesi görüntünün en soluna, son kelimesi
        # en sağına gelir; TrOCR'nin soldan sağa Cross-Attention taramasıyla birebir örtüşür.
        image = ImageOps.mirror(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        # ÖNEMLİ 2: Fairseq/TrOCR mimarisinde decoder_start_token_id = eos_token_id (2)
        # olarak eğitilmiştir ve hedef dizinin başında <s> (0) bulunmaz.
        # add_special_tokens=False ile tokenize edip yalnız sonuna eos_token_id (2) ekliyoruz.
        token_ids = self.tokenizer(
            entry["text"],
            add_special_tokens=False,
            max_length=self.max_target_length - 1,
            truncation=True,
        ).input_ids
        token_ids = token_ids + [self.tokenizer.eos_token_id]

        pad_len = self.max_target_length - len(token_ids)
        labels = token_ids + ([-100] * pad_len)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class LiveRelationshipCallback(TrainerCallback):
    """Terminalde ve goz_egitimi.log dosyasında Loss + Gerçek/Tahmin ilişkisini canlı gösterir."""

    def __init__(
        self,
        val_rows: list[dict],
        root_dir: Path,
        processor: TrOCRProcessor,
        sample_every: int = 100,
    ):
        self.val_rows = val_rows
        self.root_dir = root_dir
        self.processor = processor
        self.tokenizer = processor.tokenizer
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
            true_loss = logs["loss"] / max(1, args.gradient_accumulation_steps)
            log.info(
                "Adım %d/%d (%.1f%%) | Epoch %.2f | Loss: %.4f | LR: %.2e | VRAM: %.2f GB | Kalan: ~%.1f dk",
                state.global_step,
                state.max_steps,
                pct,
                state.epoch or 0.0,
                true_loss,
                logs.get("learning_rate", 0.0),
                vram_gb,
                eta_min,
            )
        if "eval_cer" in logs:
            cer_pct = logs["eval_cer"] * 100
            acc_pct = max(0.0, 100.0 - cer_pct)
            log.info(
                "=== DOĞRULAMA (Adım %d) | Harf Doğruluğu: %.2f%% | CER (Harf Hatası): %.2f%% | WER: %.2f%% | Eval Loss: %.4f ===",
                state.global_step,
                acc_pct,
                cer_pct,
                logs.get("eval_wer", 0.0) * 100,
                logs.get("eval_loss", 0.0),
            )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.sample_every != 0 or model is None:
            return
        model.eval()
        device = next(model.parameters()).device
        indices = [
            (state.global_step // self.sample_every + k) % len(self.val_rows)
            for k in range(3)
        ]
        log.info("--- [CANLI OKUMA TESTİ - Adım %d] Görsel <-> Gerçek Metin <-> Model Tahmini ---", state.global_step)
        with torch.no_grad():
            for idx in indices:
                item = self.val_rows[idx]
                img_path = self.root_dir / item["file_name"]
                image = ImageOps.mirror(Image.open(img_path).convert("RGB"))
                pix = self.processor(image, return_tensors="pt").pixel_values.to(device)
                gen_ids = model.generate(pix, max_new_tokens=96, num_beams=2)
                pred_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
                gt_text = item["text"]
                sample_cer = jiwer.cer(gt_text, pred_text) if gt_text else 0.0
                sample_acc = max(0.0, (1.0 - sample_cer) * 100)
                log.info(
                    "  [%s] Doğruluk=%.1f%% (CER=%.1f%%)\n    Gerçek : %s\n    Tahmin : %s",
                    item["file_name"],
                    sample_acc,
                    sample_cer * 100,
                    gt_text,
                    pred_text,
                )
        model.train()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Çapa AI - Faz 2 Göz (Tam TrOCR + Yatay Aynalama) Eğitimi")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-len", type=int, default=96)
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
            SyncFileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        ],
    )
    for noisy in ("httpx", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    args = parse_args()
    assert torch.cuda.is_available(), "CUDA destekli GPU bulunamadı!"
    bf16_ok = torch.cuda.is_bf16_supported()
    log.info("GPU: %s | BF16: %s", torch.cuda.get_device_name(0), bf16_ok)

    train_rows = [
        json.loads(line)
        for line in TRAIN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    val_rows = [
        json.loads(line)
        for line in VAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    log.info("Göz Veri Seti -> Eğitim: %d (%s) | Doğrulama: %d (%s)",
             len(train_rows), TRAIN_PATH.name, len(val_rows), VAL_PATH.name)

    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    tokenizer = processor.tokenizer
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)

    # transformers 5.x meta-device yüklemesinde sinusoidal embed_positions.weights
    # meta tensör olarak kaldığı için gerçek CUDA tensörü olarak yeniden oluşturuyoruz:
    pos_emb = model.decoder.model.decoder.embed_positions
    if hasattr(pos_emb, "weights"):
        pos_emb.weights = pos_emb.get_embedding(
            1024, pos_emb.embedding_dim, pos_emb.padding_idx
        ).to("cuda")

    # Fairseq / TrOCR Stage-1 orijinal token ayarları (decoder_start_token_id = 2 = </s>)
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.generation_config.max_length = args.max_len
    model.generation_config.early_stopping = True
    model.generation_config.num_beams = 2
    model.generation_config.decoder_start_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    total_params = sum(p.numel() for p in model.parameters())
    log.info(
        "Model Hazır: %s (Tam Hazır OCR Decoder + Yatay Aynalama) | Toplam: %.1fM param | LR=%.1e",
        TROCR_MODEL, total_params / 1e6, args.lr,
    )

    train_ds = OttomanOCRDataset(train_rows, DATA_DIR, processor, args.max_len)
    val_ds = OttomanOCRDataset(val_rows, DATA_DIR, processor, args.max_len)

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_ids_clean = [
            [tok if tok >= 0 else tokenizer.pad_token_id for tok in seq]
            for seq in pred_ids
        ]
        pred_str = tokenizer.batch_decode(pred_ids_clean, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        cer = jiwer.cer(label_str, pred_str)
        wer = jiwer.wer(label_str, pred_str)
        return {"cer": round(cer, 4), "wer": round(wer, 4)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=300,
        save_strategy="steps",
        save_steps=300,
        logging_steps=20,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        dataloader_num_workers=0,
        report_to="none",
    )

    live_cb = LiveRelationshipCallback(
        val_rows=val_rows,
        root_dir=DATA_DIR,
        processor=processor,
        sample_every=100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        callbacks=[live_cb],
    )

    log.info("Göz (Tam TrOCR-Base-Stage1 + Yatay Aynalama) eğitimi başlıyor...")
    trainer.train(resume_from_checkpoint=args.resume)

    final_dir = args.output_dir / "final_model"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info("Göz OCR modeli başarıyla kaydedildi: %s", final_dir)


if __name__ == "__main__":
    main()
