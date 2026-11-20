import json
import re
from datasets import load_dataset
from tqdm import tqdm

print("📚 10.000 kelimelik Osmanlıca-Türkçe Sözlük indiriliyor...")
# Sözlük veri setini çekiyoruz
ds_sozluk = load_dataset("TurkOpenDataOrg/Ottoman2Turkish_dictionary", split="train")

temiz_sozluk = {}

for item in tqdm(ds_sozluk, desc="Sözlük İşleniyor"):
    messages = item["messages"]
    
    # 2. mesaj kullanıcı sorusu: "'âb' kelimesinin anlamı nedir?"
    user_msg = messages[1]["content"]
    # 3. mesaj asistan cevabı: "'âb' kelimesinin anlamı: 1. su. 2. deniz..."
    ast_msg = messages[2]["content"]
    
    # User mesajından tırnak içindeki Osmanlıca kelimeyi cımbızla
    kelime_match = re.search(r"'(.*?)'", user_msg)
    
    if kelime_match:
        kelime = kelime_match.group(1).strip()
        
        # Asistan mesajından "anlamı: " sonrasındaki asıl Türkçe karşılığı cımbızla
        if "anlamı:" in ast_msg:
            anlam = ast_msg.split("anlamı:")[-1].strip()
        else:
            anlam = ast_msg.strip() # Alternatif yapı varsa direkt al
            
        temiz_sozluk[kelime] = anlam

# Kusursuz JSON formatında (key-value) kaydet
with open("osmanlica_turkce_sozluk.json", "w", encoding="utf-8") as f:
    json.dump(temiz_sozluk, f, ensure_ascii=False, indent=2)

print(f"\n✅ İŞLEM TAMAM! Toplam {len(temiz_sozluk)} kelime/tamlama temizlenip 'osmanlica_turkce_sozluk.json' dosyasına kaydedildi.")