# -*- coding: utf-8 -*-
"""
Çapa AI — Uçtan Uca Osmanlıca Okuma (Göz) ve Anlamlandırma (Beyin) Web Arayüzü
==============================================================================
- Faz 2 (Göz)   : Capa_Goz_TrOCR/final_model (Tam TrOCR + Yatay Aynalama, %87.83 Harf Doğruluğu)
- Faz 3 (Beyin) : Capa_Beyin_Qwen25_LoRA/final_adapter (Qwen2.5-3B-Instruct 4-Bit QLoRA, %85.02 Token Doğruluğu)
- Sözlük        : osmanlica_turkce_sozluk.json (9.965 Osmanlıca-Türkçe madde)
- Özellikler    :
    1) Tek Satır veya Tam Sayfa / Paragraf Görseli Yükleme (OpenCV Otomatik Satır Dilimleme)
    2) Görsel -> Göz (Latin Transkripsiyon) -> Beyin (Günümüz Türkçesi) Uçtan Uca Okuma
    3) Doğrudan Osmanlıca Metin Çevirisi (Sadece Beyin Modu)
    4) 9.965 Maddelik Osmanlıca Sözlük + Yapay Zeka Açıklaması
    5) Test Setinden (`goz_val.jsonl`) Tek Tıkla Canlı Test Galerisi

Çalıştırma:
    python capa_arayuz.py
Tarayıcı:
    http://localhost:8000
"""

from __future__ import annotations

import base64
import io
import json
import re
import sys
import time
import types

# Transformers & huggingface_hub sürüm uyumsuzluk bypass'ı
_dep_dummy = types.ModuleType("transformers.dependency_versions_check")
_dep_dummy.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = _dep_dummy

from pathlib import Path
from typing import Any

import cv2
import jiwer
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from peft import PeftModel
from PIL import Image, ImageOps
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

BASE_DIR = Path(__file__).resolve().parent
GOZ_MODEL_DIR = BASE_DIR / "Capa_Goz_TrOCR" / "final_model"
BEYIN_ADAPTER_DIR = BASE_DIR / "Capa_Beyin_Qwen25_LoRA" / "final_adapter"
BEYIN_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
VAL_JSONL = BASE_DIR / "Goz_Egitim_Seti" / "goz_val.jsonl"
DATA_DIR = BASE_DIR / "Goz_Egitim_Seti"
DICT_PATH = BASE_DIR / "osmanlica_turkce_sozluk.json"

SYSTEM_PROMPT = (
    "Sen Osmanlıca metinleri bağlamına uygun şekilde günümüz modern Türkçesine "
    "çeviren uzman bir tarih ve dil asistanısın."
)

app = FastAPI(title="Çapa AI — Osmanlıca Araştırma Asistanı")


class EngineState:
    goz_processor: TrOCRProcessor | None = None
    goz_model: VisionEncoderDecoderModel | None = None
    beyin_tokenizer: Any = None
    beyin_model: Any = None
    sozluk: dict[str, str] = {}
    samples: list[dict[str, Any]] = []
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


state = EngineState()


def pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def base64_to_pil(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def segment_lines_opencv(pil_img: Image.Image) -> tuple[list[tuple[int, int, int, int]], Image.Image]:
    """
    Tam sayfa veya çok satırlı bir Osmanlıca belge görselini OpenCV yatay projeksiyon
    ve morfolojik satır birleştirme ile yukarıdan aşağıya satır kutularına böler.
    """
    rgb = np.array(pil_img.convert("RGB"))
    h, w, _ = rgb.shape
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Arka plan aydınlatma farklarını dengele ve ikili (binary) görüntüye çevir
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )

    # Yatayda harfleri ve kelimeleri birleştirerek satır şeritleri oluştur
    kernel_w = max(25, w // 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Yatay projeksiyon profili + bağlı bileşen analizi
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    raw_boxes: list[tuple[int, int, int, int]] = []
    min_w = max(40, int(w * 0.08))
    min_h = max(14, int(h * 0.012))

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if bw < min_w or bh < min_h or area < 200:
            continue
        if bh > h * 0.6:
            continue
        raw_boxes.append((int(x), int(y), int(x + bw), int(y + bh)))

    if not raw_boxes:
        return [(0, 0, w, h)], pil_img

    # Aynı satır hizasındaki kutuları (dikey örtüşmesi yüksek olanları) tek satırda birleştir
    raw_boxes.sort(key=lambda b: b[1])
    merged: list[list[int]] = []
    for box in raw_boxes:
        x1, y1, x2, y2 = box
        placed = False
        for m in merged:
            my1, my2 = m[1], m[3]
            overlap = min(y2, my2) - max(y1, my1)
            min_height = min(y2 - y1, my2 - my1)
            if overlap > 0.45 * min_height:
                m[0] = min(m[0], x1)
                m[1] = min(m[1], y1)
                m[2] = max(m[2], x2)
                m[3] = max(m[3], y2)
                placed = True
                break
        if not placed:
            merged.append([x1, y1, x2, y2])

    # Satır kutularına üst/alt/yan nefes payı (padding) ekle
    final_boxes: list[tuple[int, int, int, int]] = []
    annotated = rgb.copy()
    pad_y = max(4, int(h * 0.006))
    pad_x = max(8, int(w * 0.01))

    for idx, (x1, y1, x2, y2) in enumerate(merged, start=1):
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)
        final_boxes.append((px1, py1, px2, py2))
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (212, 168, 83), 2)
        cv2.putText(
            annotated,
            f"#{idx}",
            (max(4, px1 + 4), max(18, py1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 40, 40),
            2,
            cv2.LINE_AA,
        )

    return final_boxes, Image.fromarray(annotated)


def goz_read_line(line_img: Image.Image) -> str:
    """Tek bir Osmanlıca satır görselini yatay aynalayarak Göz (TrOCR) modeliyle okur."""
    assert state.goz_processor is not None and state.goz_model is not None
    mirrored = ImageOps.mirror(line_img.convert("RGB"))
    pix = state.goz_processor(mirrored, return_tensors="pt").pixel_values.to(state.device)
    with torch.no_grad():
        gen_ids = state.goz_model.generate(pix, max_new_tokens=96, num_beams=2)
    text = state.goz_processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    return text


def beyin_translate(ottoman_text: str, is_word: bool = False) -> str:
    """Latin harfli Osmanlıca cümleyi veya kelimeyi Beyin (Qwen2.5 QLoRA) modeliyle günümüz Türkçesine çevirir."""
    assert state.beyin_tokenizer is not None and state.beyin_model is not None
    clean_text = ottoman_text.strip()
    if not clean_text:
        return ""

    prefix = "Osmanlıca Kelime/Tamlama: " if is_word else "Latin Osmanlıca Metin: "
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{prefix}{clean_text}"},
    ]
    prompt_str = str(
        state.beyin_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    )
    inputs = state.beyin_tokenizer(prompt_str, return_tensors="pt").to(state.device)
    with torch.no_grad():
        gen_ids = state.beyin_model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=state.beyin_tokenizer.pad_token_id,
        )
    in_len = int(inputs["input_ids"].shape[1])
    new_tokens = gen_ids[0][in_len:]
    out = state.beyin_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return out


def load_engines() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    print("=" * 72, flush=True)
    print("Çapa AI Motorları Yükleniyor...", flush=True)
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dt = torch.bfloat16 if bf16_ok else torch.float16

    # 1) GÖZ (TrOCR) Yükle
    print(f"[1/3] Göz (OCR) Modeli yükleniyor: {GOZ_MODEL_DIR}", flush=True)
    state.goz_processor = TrOCRProcessor.from_pretrained(str(GOZ_MODEL_DIR))
    goz_model = VisionEncoderDecoderModel.from_pretrained(str(GOZ_MODEL_DIR))
    pos_emb = goz_model.decoder.model.decoder.embed_positions
    if hasattr(pos_emb, "weights"):
        pos_emb.weights = pos_emb.get_embedding(
            1024, pos_emb.embedding_dim, pos_emb.padding_idx
        ).to(state.device)
    goz_model.to(state.device)
    goz_model.eval()
    state.goz_model = goz_model

    # 2) BEYİN (Qwen2.5-3B 4-bit + LoRA) Yükle
    print(f"[2/3] Beyin (Qwen2.5 QLoRA) Modeli yükleniyor: {BEYIN_ADAPTER_DIR}", flush=True)
    state.beyin_tokenizer = AutoTokenizer.from_pretrained(str(BEYIN_ADAPTER_DIR), use_fast=True)
    if state.beyin_tokenizer.pad_token is None:
        state.beyin_tokenizer.pad_token = state.beyin_tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dt,
    )
    base_llm = AutoModelForCausalLM.from_pretrained(
        BEYIN_BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=dt,
    )
    beyin_model = PeftModel.from_pretrained(base_llm, str(BEYIN_ADAPTER_DIR))
    beyin_model.config.use_cache = True
    beyin_model.eval()
    state.beyin_model = beyin_model

    # 3) Sözlük ve Test Örneklerini Yükle
    print("[3/3] Sözlük ve Test Galerisi hazırlanıyor...", flush=True)
    if DICT_PATH.exists():
        raw_dict = json.loads(DICT_PATH.read_text(encoding="utf-8"))
        if isinstance(raw_dict, dict):
            state.sozluk = {str(k).strip(): str(v).strip() for k, v in raw_dict.items()}
        elif isinstance(raw_dict, list):
            for item in raw_dict:
                if isinstance(item, dict):
                    k = item.get("kelime") or item.get("word") or item.get("osmanlica")
                    v = item.get("anlam") or item.get("meaning") or item.get("turkce")
                    if k and v:
                        state.sozluk[str(k).strip()] = str(v).strip()

    if VAL_JSONL.exists():
        val_rows = [
            json.loads(line)
            for line in VAL_JSONL.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Farklı eser ve fontlardan seçilmiş 8 doğrulama örneği
        chosen_indices = [0, 7, 15, 22, 30, 58, 73, 120]
        for idx in chosen_indices:
            if idx < len(val_rows):
                row = val_rows[idx]
                img_p = DATA_DIR / row["file_name"]
                if img_p.exists():
                    pil_im = Image.open(img_p).convert("RGB")
                    state.samples.append(
                        {
                            "id": row["file_name"],
                            "ground_truth": row["text"],
                            "image_b64": pil_to_base64(pil_im),
                        }
                    )

    vram_gb = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"Çapa AI Hazır! Toplam GPU VRAM Kullanımı: {vram_gb:.2f} GB", flush=True)
    print("Tarayıcıdan açın -> http://localhost:8000", flush=True)
    print("=" * 72, flush=True)


class OCRRequest(BaseModel):
    image_b64: str
    mode: str = "auto"  # "auto" | "single_line" | "full_page"
    ground_truth: str | None = None


class TranslateRequest(BaseModel):
    text: str


class DictRequest(BaseModel):
    query: str


@app.get("/api/samples")
def get_samples() -> JSONResponse:
    return JSONResponse({"samples": state.samples, "dict_size": len(state.sozluk)})


@app.post("/api/ocr")
def api_ocr(req: OCRRequest) -> JSONResponse:
    t0 = time.time()
    pil_img = base64_to_pil(req.image_b64)
    w, h = pil_img.size

    # Mod seçimi: Tek satır mı, çok satırlı tam sayfa mı?
    if req.mode == "single_line" or (req.mode == "auto" and (h < 180 or (w / max(1, h)) > 3.2)):
        boxes = [(0, 0, w, h)]
        annotated_b64 = pil_to_base64(pil_img)
        is_multiline = False
    else:
        boxes, annotated_pil = segment_lines_opencv(pil_img)
        annotated_b64 = pil_to_base64(annotated_pil)
        is_multiline = len(boxes) > 1

    lines_out: list[dict[str, Any]] = []
    full_ottoman_parts: list[str] = []
    full_modern_parts: list[str] = []

    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        crop = pil_img.crop((x1, y1, x2, y2))
        ott_text = goz_read_line(crop)
        mod_text = beyin_translate(ott_text, is_word=False) if ott_text else ""
        full_ottoman_parts.append(ott_text)
        full_modern_parts.append(mod_text)

        line_info: dict[str, Any] = {
            "index": idx,
            "crop_b64": pil_to_base64(crop),
            "ottoman": ott_text,
            "modern": mod_text,
        }
        if req.ground_truth and len(boxes) == 1:
            cer = jiwer.cer(req.ground_truth, ott_text)
            line_info["accuracy"] = round(max(0.0, (1.0 - cer) * 100), 1)
            line_info["ground_truth"] = req.ground_truth
        lines_out.append(line_info)

    elapsed = round(time.time() - t0, 2)
    return JSONResponse(
        {
            "is_multiline": is_multiline,
            "line_count": len(boxes),
            "annotated_b64": annotated_b64,
            "full_ottoman": "\n".join(full_ottoman_parts),
            "full_modern": "\n".join(full_modern_parts),
            "lines": lines_out,
            "elapsed_sec": elapsed,
        }
    )


@app.post("/api/translate")
def api_translate(req: TranslateRequest) -> JSONResponse:
    t0 = time.time()
    raw_lines = [ln.strip() for ln in req.text.splitlines() if ln.strip()]
    translated_lines = [beyin_translate(ln, is_word=False) for ln in raw_lines]
    return JSONResponse(
        {
            "ottoman": req.text,
            "modern": "\n".join(translated_lines),
            "elapsed_sec": round(time.time() - t0, 2),
        }
    )


@app.post("/api/dictionary")
def api_dictionary(req: DictRequest) -> JSONResponse:
    t0 = time.time()
    q = req.query.strip()
    q_low = q.lower()

    exact_match = state.sozluk.get(q) or state.sozluk.get(q_low)
    partial_matches: list[dict[str, str]] = []
    for k, v in state.sozluk.items():
        if q_low in k.lower() or q_low in v.lower():
            partial_matches.append({"word": k, "meaning": v})
            if len(partial_matches) >= 12:
                break

    ai_meaning = beyin_translate(q, is_word=True)
    return JSONResponse(
        {
            "query": q,
            "exact_match": exact_match,
            "ai_meaning": ai_meaning,
            "matches": partial_matches,
            "elapsed_sec": round(time.time() - t0, 2),
        }
    )


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Cevdet-Translate v1.0 — Osmanlıca Yapay Zekâ Çeviri Platformu</title>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --panel-alt: #1f2630;
    --border: #30363d;
    --gold: #d4a853;
    --gold-soft: rgba(212, 168, 83, 0.14);
    --text: #e6edf3;
    --muted: #8b949e;
    --green: #3fb950;
    --blue: #58a6ff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    line-height: 1.5;
    padding-bottom: 60px;
  }
  header {
    background: linear-gradient(135deg, #161b22 0%, #1b222d 100%);
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .brand-badge {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    background: var(--gold-soft);
    border: 1px solid var(--gold);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: 700;
    color: var(--gold);
  }
  .brand h1 {
    font-size: 21px;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: 0.3px;
  }
  .brand p {
    font-size: 13px;
    color: var(--muted);
  }
  .status-pills {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
  }
  .pill {
    font-size: 12px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 999px;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .pill strong { color: var(--gold); }
  .beta-badge {
    background: var(--gold);
    color: #0d1117;
    font-weight: 800;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 4px;
    letter-spacing: 0.5px;
  }
  .beta-banner {
    background: rgba(212, 168, 83, 0.08);
    border: 1px solid rgba(212, 168, 83, 0.35);
    color: var(--text);
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 13.5px;
  }
  .beta-banner-icon {
    font-size: 22px;
    line-height: 1;
  }
  .container {
    max-width: 1320px;
    margin: 24px auto;
    padding: 0 24px;
  }
  .tabs {
    display: flex;
    gap: 10px;
    margin-bottom: 22px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
    flex-wrap: wrap;
  }
  .tab-btn {
    background: var(--panel);
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 10px 18px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.15s;
  }
  .tab-btn.active, .tab-btn:hover {
    color: var(--gold);
    border-color: var(--gold);
    background: var(--gold-soft);
  }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1.25fr;
    gap: 22px;
  }
  @media (max-width: 980px) {
    .grid-2 { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }
  .card h2 {
    font-size: 16px;
    color: var(--gold);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .dropzone {
    border: 2px dashed var(--border);
    border-radius: 10px;
    padding: 26px 18px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: rgba(255,255,255,0.01);
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: var(--gold);
    background: var(--gold-soft);
  }
  .dropzone p { color: var(--muted); font-size: 14px; margin-top: 6px; }
  .controls {
    display: flex;
    gap: 10px;
    margin-top: 14px;
    flex-wrap: wrap;
    align-items: center;
  }
  select, input[type="text"], textarea {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 14px;
    width: 100%;
  }
  .btn {
    background: var(--gold);
    color: #0d1117;
    font-weight: 700;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    transition: opacity 0.15s;
  }
  .btn:hover { opacity: 0.9; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .preview-box {
    margin-top: 16px;
    background: #090c10;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    text-align: center;
    max-height: 440px;
    overflow: auto;
  }
  .preview-box img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
  }
  .samples-bar {
    margin-top: 18px;
  }
  .samples-bar h3 {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .sample-list {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .sample-chip {
    font-size: 12px;
    padding: 6px 11px;
    border-radius: 6px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    color: var(--text);
    cursor: pointer;
  }
  .sample-chip:hover {
    border-color: var(--gold);
    color: var(--gold);
  }
  .result-block {
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 14px;
  }
  .result-label {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
  }
  .label-goz { color: var(--blue); }
  .label-beyin { color: var(--green); }
  .label-gt { color: var(--gold); }
  .result-text {
    font-size: 16px;
    white-space: pre-wrap;
    color: var(--text);
  }
  .line-card {
    background: #12171e;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    margin-top: 10px;
  }
  .line-card img {
    max-width: 100%;
    max-height: 75px;
    border-radius: 4px;
    margin-bottom: 8px;
    border: 1px solid #2b313a;
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="brand-badge">C</div>
    <div>
      <h1>Cevdet-Translate <span style="font-size:13px;color:var(--blue);font-weight:600;margin-left:6px">v1.0.0</span></h1>
      <p>Yapay zekâ ile matbu Osmanlıca belgeleri okuma ve günümüz modern Türkçesine çevirme platformu</p>
    </div>
  </div>
  <div class="status-pills">
    <div class="pill"><span class="beta-badge">BETA</span> <strong>Test Sürümü</strong></div>
    <div class="pill"><strong>Sözlük Kapsamı:</strong> <span id="dictCount">9.965</span> Kelime</div>
  </div>
</header>

<div class="container">
  <div class="beta-banner">
    <span class="beta-banner-icon">🧪</span>
    <div>
      <strong>Sistem Geliştirme ve Test Aşamasındadır:</strong> Çıktılar araştırma amaçlı üretilmektedir. Belgenin matbu netliği, yıpranma durumu ve çözünürlüğe bağlı olarak okuma ve çeviri farklılıkları oluşabilir.
    </div>
  </div>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('ocr')">1. Görselden Oku ve Çevir</button>
    <button class="tab-btn" onclick="switchTab('translate')">2. Metin Çevirisi</button>
    <button class="tab-btn" onclick="switchTab('dict')">3. Osmanlıca Sözlük</button>
  </div>

  <!-- SEKME 1: GÖRSEL OCR + ÇEVİRİ -->
  <div id="tab-ocr" class="tab-panel active">
    <div class="grid-2">
      <div class="card">
        <h2>Belge Görseli Yükle</h2>
        <div id="dropzone" class="dropzone" onclick="document.getElementById('fileInput').click()">
          <strong>Görsel Seçmek İçin Tıklayın, Sürükleyin veya Yapıştırın</strong>
          <p>Tek satır veya çok satırlı matbu Osmanlıca gazete/kitap sayfası (.jpg, .png, .webp)</p>
          <input id="fileInput" type="file" accept="image/*" style="display:none" onchange="handleFile(this.files[0])" />
        </div>

        <div class="controls">
          <select id="ocrMode" style="flex:1">
            <option value="auto">Otomatik Algıla (Tek Satır veya Tam Sayfa)</option>
            <option value="single_line">Tek Satır Belge</option>
            <option value="full_page">Çok Satırlı / Tam Sayfa Belge</option>
          </select>
          <button id="runOcrBtn" class="btn" onclick="runOCR()">Metni Oku ve Çevir</button>
        </div>

        <div class="preview-box" id="previewBox" style="display:none">
          <img id="previewImg" src="" alt="Önizleme" />
        </div>

        <div class="samples-bar">
          <h3>Hızlı Deneme İçin Örnek Satırlar:</h3>
          <div id="sampleList" class="sample-list"></div>
        </div>
      </div>

      <div class="card">
        <h2>
          <span>Okuma ve Çeviri Sonucu</span>
          <span id="ocrTiming" style="font-size:12px;color:var(--muted);font-weight:400"></span>
        </h2>

        <div id="gtBlock" class="result-block" style="display:none">
          <div class="result-label label-gt">
            <span>Referans Metin</span>
            <span id="accBadge"></span>
          </div>
          <div id="gtText" class="result-text"></div>
        </div>

        <div class="result-block">
          <div class="result-label label-goz">
            <span>1. Belgeden Okunan Metin (Osmanlıca / Transkripsiyon)</span>
          </div>
          <div id="gozText" class="result-text" style="color:#c9d1d9">Henüz bir görsel yüklenmedi. Soldaki alandan görsel seçebilir veya hazır örnekleri deneyebilirsiniz.</div>
        </div>

        <div class="result-block">
          <div class="result-label label-beyin">
            <span>2. Günümüz Türkçesi Çevirisi</span>
          </div>
          <div id="beyinText" class="result-text" style="color:#a5d6ff">...</div>
        </div>

        <div id="linesContainer"></div>
      </div>
    </div>
  </div>

  <!-- SEKME 2: METİN ÇEVİRİSİ -->
  <div id="tab-translate" class="tab-panel">
    <div class="grid-2">
      <div class="card">
        <h2>Osmanlıca / Eski Türkçe Metin</h2>
        <textarea id="rawOttomanInput" rows="7" placeholder="Çevirmek istediğiniz Osmanlıca veya eski Türkçe metni buraya yazın veya yapıştırın..."></textarea>
        <div class="controls">
          <button id="runTransBtn" class="btn" onclick="runTranslate()">Günümüz Türkçesine Çevir</button>
        </div>
      </div>
      <div class="card">
        <h2>
          <span>Günümüz Türkçesi Çevirisi</span>
          <span id="transTiming" style="font-size:12px;color:var(--muted)"></span>
        </h2>
        <div class="result-block">
          <div id="transOutput" class="result-text">Çeviri burada görünecektir...</div>
        </div>
      </div>
    </div>
  </div>

  <!-- SEKME 3: SÖZLÜK -->
  <div id="tab-dict" class="tab-panel">
    <div class="card">
      <h2>Osmanlıca Kelime Arama</h2>
      <div class="controls" style="margin-bottom:18px">
        <input id="dictInput" type="text" placeholder="Aramak istediğiniz kelimeyi yazın (Örn: müstelzim, ferzend, şerâit, teessüf, paymâl)..." style="flex:1" onkeydown="if(event.key==='Enter') runDict()" />
        <button class="btn" onclick="runDict()">Ara</button>
      </div>
      <div id="dictResult"></div>
    </div>
  </div>
</div>

<script>
let currentImageB64 = null;
let currentGroundTruth = null;
let samplesData = [];

function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    const map = ['ocr', 'translate', 'dict'];
    b.classList.toggle('active', map[i] === name);
  });
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
}

async function initSamples() {
  const res = await fetch('/api/samples');
  const data = await res.json();
  samplesData = data.samples || [];
  document.getElementById('dictCount').textContent = (data.dict_size || 9965).toLocaleString('tr-TR');
  const box = document.getElementById('sampleList');
  box.innerHTML = '';
  samplesData.forEach((s, idx) => {
    const btn = document.createElement('button');
    btn.className = 'sample-chip';
    btn.textContent = `Örnek #${idx + 1} (${s.id.replace('images/', '')})`;
    btn.onclick = () => selectSample(idx);
    box.appendChild(btn);
  });
}

function selectSample(idx) {
  const s = samplesData[idx];
  currentImageB64 = s.image_b64;
  currentGroundTruth = s.ground_truth;
  document.getElementById('previewImg').src = currentImageB64;
  document.getElementById('previewBox').style.display = 'block';
  runOCR();
}

function handleFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    currentImageB64 = e.target.result;
    currentGroundTruth = null;
    document.getElementById('previewImg').src = currentImageB64;
    document.getElementById('previewBox').style.display = 'block';
    document.getElementById('gtBlock').style.display = 'none';
  };
  reader.readAsDataURL(file);
}

window.addEventListener('paste', e => {
  const items = (e.clipboardData || e.originalEvent.clipboardData).items;
  for (const item of items) {
    if (item.type.indexOf('image') === 0) {
      handleFile(item.getAsFile());
    }
  }
});

const dz = document.getElementById('dropzone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

async function runOCR() {
  if (!currentImageB64) {
    alert('Lütfen önce bir görsel yükleyin veya aşağıdaki hazır örneklerden birine tıklayın.');
    return;
  }
  const btn = document.getElementById('runOcrBtn');
  btn.disabled = true;
  btn.textContent = 'Okunuyor ve Çevriliyor...';
  document.getElementById('gozText').textContent = 'Belgedeki satırlar okunuyor...';
  document.getElementById('beyinText').textContent = 'Günümüz Türkçesine çevriliyor...';

  try {
    const mode = document.getElementById('ocrMode').value;
    const res = await fetch('/api/ocr', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        image_b64: currentImageB64,
        mode: mode,
        ground_truth: currentGroundTruth
      })
    });
    const data = await res.json();
    document.getElementById('previewImg').src = data.annotated_b64;
    document.getElementById('gozText').textContent = data.full_ottoman || '(Boş)';
    document.getElementById('beyinText').textContent = data.full_modern || '(Boş)';
    document.getElementById('ocrTiming').textContent = `${data.line_count} satır • ${data.elapsed_sec} sn`;

    if (data.lines.length === 1 && data.lines[0].ground_truth) {
      document.getElementById('gtBlock').style.display = 'block';
      document.getElementById('gtText').textContent = data.lines[0].ground_truth;
      document.getElementById('accBadge').textContent = `Örnek Doğruluğu: %${data.lines[0].accuracy}`;
    } else {
      document.getElementById('gtBlock').style.display = 'none';
    }

    const lc = document.getElementById('linesContainer');
    lc.innerHTML = '';
    if (data.lines.length > 1) {
      data.lines.forEach(ln => {
        const div = document.createElement('div');
        div.className = 'line-card';
        div.innerHTML = `
          <div style="font-size:12px;color:var(--gold);font-weight:700;margin-bottom:4px">Satır #${ln.index}</div>
          <img src="${ln.crop_b64}" />
          <div style="font-size:14px;color:#c9d1d9"><strong>Okunan Metin:</strong> ${ln.ottoman}</div>
          <div style="font-size:14px;color:#a5d6ff;margin-top:4px"><strong>Günümüz Türkçesi:</strong> ${ln.modern}</div>
        `;
        lc.appendChild(div);
      });
    }
  } finally {
    btn.disabled = false;
    btn.textContent = 'Metni Oku ve Çevir';
  }
}

async function runTranslate() {
  const txt = document.getElementById('rawOttomanInput').value.trim();
  if (!txt) return;
  const btn = document.getElementById('runTransBtn');
  btn.disabled = true;
  document.getElementById('transOutput').textContent = 'Çevriliyor...';
  try {
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: txt})
    });
    const data = await res.json();
    document.getElementById('transOutput').textContent = data.modern;
    document.getElementById('transTiming').textContent = `${data.elapsed_sec} sn`;
  } finally {
    btn.disabled = false;
  }
}

async function runDict() {
  const q = document.getElementById('dictInput').value.trim();
  if (!q) return;
  const box = document.getElementById('dictResult');
  box.innerHTML = '<div class="result-block">Sorgulanıyor...</div>';
  const res = await fetch('/api/dictionary', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: q})
  });
  const data = await res.json();
  let html = `
    <div class="result-block">
      <div class="result-label label-beyin"><span>Yapay Zekâ Anlamlandırması</span><span>${data.elapsed_sec} sn</span></div>
      <div class="result-text">${data.ai_meaning}</div>
    </div>
  `;
  if (data.exact_match) {
    html += `
      <div class="result-block">
        <div class="result-label label-gt"><span>Sözlük Anlamı (${data.query})</span></div>
        <div class="result-text">${data.exact_match}</div>
      </div>
    `;
  }
  if (data.matches && data.matches.length) {
    html += `<div class="result-block"><div class="result-label label-goz"><span>İlgili Sözlük Kelimeleri</span></div>`;
    data.matches.forEach(m => {
      html += `<div style="padding:6px 0;border-bottom:1px solid var(--border)"><strong>${m.word}:</strong> ${m.meaning}</div>`;
    });
    html += `</div>`;
  }
  box.innerHTML = html;
}

initSamples();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


if __name__ == "__main__":
    load_engines()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

