# Cevdet Translate — Uçtan Uca Osmanlıca Araştırma Asistanı 📜✨

**Cevdet Translate**, matbu Osmanlıca belgeleri satır satır veya tam sayfa olarak görselden okuyan (**Göz / OCR**) ve bağlama uygun şekilde doğrudan günümüz modern Türkçesine çeviren (**Beyin / LLM**) iki aşamalı hibrit bir yapay zeka araştırma asistanıdır.

---

## 🏛️ Mimari ve Bileşenler

```
                                  [Matbu Osmanlıca Belge]
                                             │
                                             ▼
                       ┌───────────────────────────────────────────┐
                       │   OpenCV Akıllı Satır Dilimleyici         │
                       │   (Yatay Projeksiyon + Morfolojik Filtre) │
                       └─────────────────────┬─────────────────────┘
                                             │  Satır Görselleri (satir_*.jpg)
                                             ▼
                       ┌───────────────────────────────────────────┐
                       │   FAZ 2: "GÖZ" (TrOCR OCR Modeli)         │
                       │   • ViT Encoder + Hazır TrOCR Decoder    │
                       │   • Yatay Aynalama (Mirroring) Hizalaması │
                       │   • 225.7M Parametre                      │
                       │   • Harf Doğruluğu: %87.83 (CER: %12.17)  │
                       └─────────────────────┬─────────────────────┘
                                             │  Latin Transkripsiyon
                                             ▼
                       ┌───────────────────────────────────────────┐
                       │   FAZ 3: "BEYİN" (Qwen-2.5 QLoRA)         │
                       │   • Qwen2.5-3B-Instruct (4-Bit NF4 LoRA)  │
                       │   • 17.066 Eğitim / 897 Doğrulama Kaydı   │
                       │   • 9.965 Maddelik Tarih & Dil Sözlüğü    │
                       │   • Doğrulama Token Doğruluğu: %85.02     │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                       [Kusursuz Günümüz Modern Türkçesi Çevirisi]
```

---

## 🚀 Başarı Metrikleri

| Model | Mimari / Yöntem | Doğruluk | Hata / Kayıp (Loss) |
| :--- | :--- | :---: | :---: |
| **Göz (OCR)** | `microsoft/trocr-base-stage1` + Yatay Aynalama | **%87.83 Harf Doğruluğu** | CER: %12.17 &bull; Eval Loss: 0.6524 |
| **Beyin (Çeviri)** | `Qwen2.5-3B-Instruct` (4-Bit NF4 QLoRA) | **%85.02 Token Doğruluğu** | Eval Loss: 0.7590 &bull; Train Loss: 0.6784 |

---

## 💻 Kurulum

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/KULLANICI_ADINIZ/capa-ai-osmanlica.git
cd capa-ai-osmanlica
```

### 2. Sanal Ortam Oluşturun ve Paketleri Yükleyin
```bash
python -m venv venv
.\venv\Scripts\activate

# CUDA destekli PyTorch ve kütüphaneleri yükleyin
pip install -r requirements.txt
```

---

## 🖥️ Web Arayüzünü Başlatma

Uçtan uca web arayüzünü (Göz + Beyin + 9.965 maddelik sözlük) çalıştırmak için:

```bash
python capa_arayuz.py
```

Tarayıcınızdan açın:
👉 **`http://localhost:8000`**

### Arayüz Özellikleri:
1. **Görselden Okuma & Çeviri:** Tek satır küpür veya tam sayfa Osmanlıca belge yükleyin (ya da `Ctrl + V` ile ekran görüntüsü yapıştırın).
2. **Doğrudan Metin Çevirisi (Sadece Beyin):** Latin harfli Osmanlıca metinleri günümüz Türkçesine aktarın.
3. **Osmanlıca Sözlük & Tahlil:** 9.965 maddelik sözlükten anında arama yapın ve yapay zeka anlamlandırmasını görün.

---

## 🏋️ Modelleri Yeniden Eğitme

* **Göz (TrOCR) Eğitimi:**
  ```bash
  python goz_ocr_egitimi.py --epochs 6 --batch-size 4 --grad-accum 4
  ```

* **Beyin (Qwen2.5 QLoRA) Eğitimi:**
  ```bash
  python beyin_lora_egitimi.py --epochs 2 --batch-size 4 --grad-accum 4
  ```

---

## 📂 Dizin Yapısı

```
├── capa_arayuz.py                 # FastAPI + Uvicorn modern web arayüzü
├── goz_ocr_egitimi.py             # Faz 2 TrOCR OCR eğitim betiği
├── beyin_lora_egitimi.py          # Faz 3 Qwen-2.5 QLoRA eğitim betiği
├── bekle_ve_beyni_baslat.py       # Zincirleme otomatik eğitim nöbetçisi
├── sentetik_veri_uretimi.py       # 8.000 satırlık sentetik veri üretim betiği
├── veri_birlestirme.py            # Train / Validation veri seti bölücü
├── osmanlica_turkce_sozluk.json   # 9.965 maddelik Osmanlıca-Türkçe sözlük
├── beyin_train.jsonl              # Beyin eğitim seti (17.066 kayıt)
├── beyin_val.jsonl                # Beyin doğrulama seti (897 kayıt)
├── Goz_Egitim_Seti/               # Göz eğitim seti (8.016 satır görseli ve jsonl)
├── requirements.txt               # Gerekli Python bağımlılıkları
├── LICENSE                        # MIT Lisansı
└── .gitignore                     # Git tarafından göz ardı edilecek dosyalar
```

---

## ⚖️ Lisans

Bu proje [MIT Lisansı](file:///c:/Users/ensar/Desktop/ottoman/LICENSE) ile lisanslanmıştır. Detaylı bilgi için [LICENSE](file:///c:/Users/ensar/Desktop/ottoman/LICENSE) dosyasını inceleyebilirsiniz.
