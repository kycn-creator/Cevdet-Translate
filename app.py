# -*- coding: utf-8 -*-
"""
Cevdet-Translate v1.0.0
=======================
Osmanlıca Metin ve Belgeler İçin Uçtan Uca Yapay Zekâ Çeviri Platformu.

Kullanım:
    python app.py
Tarayıcı:
    http://localhost:8000
"""

from capa_arayuz import app, load_engines
import uvicorn

if __name__ == "__main__":
    load_engines()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
