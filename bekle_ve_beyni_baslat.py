# -*- coding: utf-8 -*-
"""
Çapa AI - Göz Eğitimini Bekleyip Beyin Eğitimini Otomatik Başlatıcı
===================================================================
1. Çalışmakta olan Göz (goz_ocr_egitimi.py) eğitimine ve GPU VRAM'ine
   kesinlikle dokunmaz.
2. Göz eğitimi devam ederken arka planda Qwen/Qwen2.5-3B-Instruct model
   dosyalarını diske ön-indirir (0 MB VRAM kullanır).
3. goz_egitimi.log içinde "Göz OCR modeli başarıyla kaydedildi" mesajını
   ve Capa_Goz_TrOCR/final_model klasörünü gördüğü anda 10 saniye bekleyip
   GPU belleğinin tamamen boşalmasını sağlar ve otomatik olarak
   beyin_lora_egitimi.py eğitimini başlatır!
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
GOZ_LOG = BASE_DIR / "goz_egitimi.log"
GOZ_FINAL = BASE_DIR / "Capa_Goz_TrOCR" / "final_model"
BEYIN_SCRIPT = BASE_DIR / "beyin_lora_egitimi.py"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    print("[Zincirleme Başlatıcı] Qwen/Qwen2.5-3B-Instruct ağırlıkları diske ön-indiriliyor (GPU kullanılmaz)...", flush=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="Qwen/Qwen2.5-3B-Instruct")
        print("[Zincirleme Başlatıcı] Qwen/Qwen2.5-3B-Instruct diske hazırlandı!", flush=True)
    except Exception as e:
        print(f"[Zincirleme Başlatıcı] Ön-indirme uyarısı (eğitim sırasında indirilecek): {e}", flush=True)

    print("[Zincirleme Başlatıcı] Göz eğitiminin (2856 adım) tamamlanması bekleniyor...", flush=True)
    while True:
        if GOZ_LOG.exists():
            try:
                txt = GOZ_LOG.read_text(encoding="utf-8", errors="ignore")
                if "Göz OCR modeli başarıyla kaydedildi" in txt and GOZ_FINAL.exists():
                    print("[Zincirleme Başlatıcı] Göz eğitimi tamamlandı! GPU VRAM'in boşalması için 10 sn bekleniyor...", flush=True)
                    time.sleep(10)
                    break
            except OSError:
                pass
        time.sleep(15)

    print("[Zincirleme Başlatıcı] Faz 3: Beyin (beyin_lora_egitimi.py) başlatılıyor!", flush=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    rc = subprocess.call([sys.executable, str(BEYIN_SCRIPT)], cwd=str(BASE_DIR), env=env)
    sys.exit(rc)


if __name__ == "__main__":
    main()

