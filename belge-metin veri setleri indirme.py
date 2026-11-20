import os
import json
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm

# 1. Hugging Face Token'ını buraya yapıştır
login("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")

# Klasör mimarisini kuralım
base_dir = "Goz_Egitim_Seti"
img_dir = os.path.join(base_dir, "images")
os.makedirs(img_dir, exist_ok=True)

print("🚀 1. Akis ve Asparius setleri çekiliyor (Görseller de indiği için internet hızına bağlı olarak 1-2 dakika sürebilir)...")

# Hugging Face'ten setleri çekiyoruz
# Not: Akis setinin verileri "test", Asparius'un verileri genelde "train" split'inde tutulur.
ds_akis = load_dataset("OttomanNLP/Akis-Ottoman-Dataset", split="test")
ds_asparius = load_dataset("asparius/ottoman-ocr-8k", split="train")

print("🧹 2. Kopyalar temizleniyor ve veriler birleştiriliyor...")

# Benzersiz metinleri takip etmek için bir Set (Küme) ve asıl veriyi tutacağımız bir Liste
gorulmus_metinler = set()
essiz_veriler = []

# Akis verilerini tarayalım
for item in tqdm(ds_akis, desc="Akis Taranıyor"):
    metin = item["text"].strip()
    if metin not in gorulmus_metinler:
        gorulmus_metinler.add(metin)
        essiz_veriler.append({"image": item["image"], "text": metin, "kaynak": "akis"})

# Asparius verilerini tarayalım
for item in tqdm(ds_asparius, desc="Asparius Taranıyor"):
    metin = item["text"].strip()
    # Eğer bu metin Akis'te zaten varsa, atla! (Kopya temizliği)
    if metin not in gorulmus_metinler:
        gorulmus_metinler.add(metin)
        essiz_veriler.append({"image": item["image"], "text": metin, "kaynak": "asparius"})

toplam_eski_sati = len(ds_akis) + len(ds_asparius)
toplam_yeni_satir = len(essiz_veriler)
print(f"\n📊 Analiz: Toplam {toplam_eski_sati} satır incelendi.")
print(f"✅ {toplam_eski_sati - toplam_yeni_satir} adet kopya satır silindi!")
print(f"🎯 Elimizde {toplam_yeni_satir} adet tertemiz, eşsiz eğitim verisi kaldı.")

print("\n💾 3. Görseller yerel diske kaydediliyor ve metadata.jsonl oluşturuluyor...")

metadata_list = []

# Görselleri klasöre kaydedip modelin okuyacağı metadata dosyasını hazırlıyoruz
for idx, data in enumerate(tqdm(essiz_veriler, desc="Diske Yazılıyor")):
    img_filename = f"satir_{idx}.jpg"
    img_path = os.path.join(img_dir, img_filename)
    
    # Görseli diske kaydet
    data["image"].convert("RGB").save(img_path)
    
    # Metadata kaydını oluştur
    metadata_list.append({
        "file_name": f"images/{img_filename}",
        "text": data["text"]
    })

# Modellerin standart eğitim formatı olan JSONL (JSON Lines) formatında kaydedelim
metadata_path = os.path.join(base_dir, "metadata.jsonl")
with open(metadata_path, "w", encoding="utf-8") as f:
    for entry in metadata_list:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\n🎉 İŞLEM TAMAM! Tüm görseller '{base_dir}/images' klasörüne indi.")
print(f"📜 Model eğitim dosyası '{metadata_path}' konumunda hazır!")