# -*- coding: utf-8 -*-
"""
Çapa AI - Faz 3 (Beyin) Nihai Eğitim ve Doğrulama Veri Seti Hazırlayıcı
======================================================================
1. beyin_egitim_seti.jsonl (7.998 bağlamsal cümle çevirisi)
2. osmanlica_turkce_sozluk.json (9.965 Osmanlıca-Türkçe sözlük maddesi, PDF OCR
   kopuk harf hataları temizlenerek Qwen-2.5 messages formatına dönüştürülür)

Çıktılar:
- beyin_train.jsonl (%95 eğitim seti - karıştırılmış / shuffled)
- beyin_val.jsonl   (%5 doğrulama seti - stratified: hem cümle hem sözlük içerir)
"""

import json
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SENTENCE_PATH = BASE_DIR / "beyin_egitim_seti.jsonl"
DICT_PATH = BASE_DIR / "osmanlica_turkce_sozluk.json"
TRAIN_OUT = BASE_DIR / "beyin_train.jsonl"
VAL_OUT = BASE_DIR / "beyin_val.jsonl"

GOZ_METADATA = BASE_DIR / "Goz_Egitim_Seti" / "metadata.jsonl"
GOZ_TRAIN_OUT = BASE_DIR / "Goz_Egitim_Seti" / "goz_train.jsonl"
GOZ_VAL_OUT = BASE_DIR / "Goz_Egitim_Seti" / "goz_val.jsonl"

SYSTEM_PROMPT = (
    "Sen Osmanlıca metinleri bağlamına uygun şekilde günümüz modern Türkçesine "
    "çeviren uzman bir tarih ve dil asistanısın."
)


def clean_ocr_definition(text: str) -> str:
    """Sözlükteki PDF OCR kaynaklı kopuk Türkçe harfleri ('k izil' -> 'kızıl') onarır."""
    cleaned = re.sub(
        r"(?<=[a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛ])\s+(?=[çğıöşü][a-zA-Zçğıöşüâîû]*\b)",
        "",
        text,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def main() -> None:
    random.seed(42)

    # 1) Cümle çevirilerini yükle (7.998 kayıt)
    sentence_records = []
    with SENTENCE_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentence_records.append(json.loads(line))

    # 2) Sözlük maddelerini yükle, temizle ve Qwen-2.5 messages formatına çevir (9.965 kayıt)
    raw_dict = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    dict_records = []
    for word, definition in raw_dict.items():
        w = word.strip()
        d = clean_ocr_definition(str(definition))
        if not w or not d:
            continue
        dict_records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Osmanlıca Kelime/Tamlama: {w}",
                    },
                    {"role": "assistant", "content": d},
                ]
            }
        )

    # Stratified %95 Train / %5 Validation ayrımı
    random.shuffle(sentence_records)
    random.shuffle(dict_records)

    s_val_n = int(len(sentence_records) * 0.05)  # ~400 cümle
    d_val_n = int(len(dict_records) * 0.05)      # ~498 sözlük maddesi

    val_records = sentence_records[:s_val_n] + dict_records[:d_val_n]
    train_records = sentence_records[s_val_n:] + dict_records[d_val_n:]

    random.shuffle(train_records)
    random.shuffle(val_records)

    with TRAIN_OUT.open("w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with VAL_OUT.open("w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Cümle çevirisi kaydı : {len(sentence_records)}")
    print(f"Temiz sözlük kaydı   : {len(dict_records)}")
    print(f"Toplam birleşik veri : {len(train_records) + len(val_records)}")
    print(f"  -> beyin_train.jsonl : {len(train_records)} kayıt")
    print(f"  -> beyin_val.jsonl   : {len(val_records)} kayıt")

    # 3) Göz (OCR) veri setini de fiziksel olarak %95 Train / %5 Val ayır
    random.seed(42)
    goz_records = [
        json.loads(line)
        for line in GOZ_METADATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    random.shuffle(goz_records)
    g_val_n = int(len(goz_records) * 0.05)  # 400 kayıt
    goz_val = goz_records[:g_val_n]
    goz_train = goz_records[g_val_n:]

    with GOZ_TRAIN_OUT.open("w", encoding="utf-8") as f:
        for r in goz_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with GOZ_VAL_OUT.open("w", encoding="utf-8") as f:
        for r in goz_val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nGöz (OCR) toplam veri : {len(goz_records)}")
    print(f"  -> Goz_Egitim_Seti/goz_train.jsonl : {len(goz_train)} kayıt")
    print(f"  -> Goz_Egitim_Seti/goz_val.jsonl   : {len(goz_val)} kayıt")


if __name__ == "__main__":
    main()

