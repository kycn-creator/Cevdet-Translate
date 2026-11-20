# -*- coding: utf-8 -*-
"""
Çapa AI - Faz 3 (Beyin) Sentetik Paralel Veri Üretimi
=====================================================

Goz_Egitim_Seti/metadata.jsonl içindeki Latin harfli Osmanlıca satırları,
osmanlica_turkce_sozluk.json'dan çıkarılan bağlamla zenginleştirip Gemini ile
günümüz Türkçesine çevirir ve Qwen-2.5 LoRA (chat/messages) formatında
beyin_egitim_seti.jsonl dosyasına yazar.

Özellikler
----------
* Diakritik / Türkçe karakter bağımsız sözlük eşleştirme (â→a, ş→s, ı→i ...),
  izafet (-i / -ı / -u / -ü) tamlamaları için n-gram (3/2/1) arama ve
  ek atma (en uzun kök öneki) ile kök tahmini.
* Toplu (batch) istek + JSON şemalı yapılandırılmış çıktı. Toplu yanıt bozuksa
  satır satır yedek (fallback) çeviri.
* tenacity ile exponential backoff + jitter (429 / 5xx / timeout / bağlantı).
* İstemci tarafı RPM sınırlayıcı (ücretsiz katman kotaları için).
* Kesinti koruması: her batch sonrası append + fsync, atomik checkpoint.json.
  Yeniden çalıştırıldığında kaldığı index'ten devam eder; checkpoint ile çıktı
  dosyası arasında oluşabilecek tutarsızlığı (yarım yazılmış batch) otomatik
  olarak onarır (mükerrer kayıt oluşmaz).

Kullanım
--------
    set GEMINI_API_KEY=...            (PowerShell: $env:GEMINI_API_KEY="...")
    python sentetik_veri_uretimi.py                      # tümünü üret / devam et
    python sentetik_veri_uretimi.py --dry-run --limit 3  # API'siz prompt önizleme
    python sentetik_veri_uretimi.py --model gemini-2.0-flash --batch-size 10 --rpm 15
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import re
import sys
import threading
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA = BASE_DIR / "Goz_Egitim_Seti" / "metadata.jsonl"
DEFAULT_DICT = BASE_DIR / "osmanlica_turkce_sozluk.json"
DEFAULT_OUTPUT = BASE_DIR / "beyin_egitim_seti.jsonl"
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoint.json"
DEFAULT_FAILED = BASE_DIR / "basarisiz_satirlar.jsonl"

# Nihai eğitim setindeki (Qwen-2.5) system mesajı
QWEN_SYSTEM_PROMPT = (
    "Sen Osmanlıca metinleri bağlamına uygun şekilde günümüz modern Türkçesine "
    "çeviren uzman bir tarih ve dil asistanısın."
)
USER_PREFIX = "Latin Osmanlıca Metin: "

# Veri üretici (Gemini) için talimat
GEMINI_SYSTEM_INSTRUCTION = """\
Sen Osmanlı Türkçesi filolojisi, Arapça-Farsça kökenli söz varlığı ve tarihî metin \
neşri konusunda uzman bir akademisyensin. Görevin, Latin harflerine transkribe \
edilmiş Osmanlıca metin satırlarını günümüz modern Türkçesine çevirmektir.

KESİN KURALLAR:
1. SİYAK-SİBAK (BAĞLAM): Bir kelimenin sözlükte birden fazla anlamı varsa, cümlenin \
ve komşu satırların bağlamına en uygun TEK anlamı seç. Numaralı anlamları asla \
sıralama, alternatif verme, parantez içinde açıklama yazma.
2. TRANSKRİPSİYON HATALARINA TOLERANS: Metinler OCR veya elle transkripsiyondan \
gelir. Şapka eksikliği (a/â, i/î, u/û), ı/i, s/ş, g/ğ, c/ç, o/ö, u/ü karışıklıkları, \
eksik/fazla/yer değiştirmiş harfler, hatalı bitişik/ayrık yazımlar olabilir. Bağlamdan \
kastedilen doğru kelimeyi çıkar ve onu çevir. Düzeltme notu düşme.
3. KESİK SATIRLAR: Satırlar bir sayfanın tek bir satırıdır; cümle ortasında başlayıp \
bitebilir. Eksik kısmı UYDURMA veya tamamlama; yalnızca satırda bulunan içeriği, aynı \
kesiklikte, doğal bir modern Türkçe ile karşıla.
4. KORUMA: Özel isimler, yer adları, tarihler, sayılar, madde numaraları ve noktalama \
yapısı korunur (gerekirse günümüz yazımına uyarlanır).
5. ÜSLUP: Akıcı, sade ve akademik modern Türkçe kullan. Günümüzde zaten anlaşılan \
kelimeleri gereksiz yere değiştirme; Arapça-Farsça terkip ve tamlamaları (ör. \
"elviye-i mütecavire" → "komşu sancaklar") günümüz Türkçesine aç. Anlam ekleme, \
yorum yapma, özetleme.
6. SÖZLÜK BAĞLAMI YARDIMCIDIR, BAĞLAYICI DEĞİLDİR: Sözlük tanımları PDF'den \
çıkarıldığı için bozuk harf boşlukları içerebilir (ör. "k izil" = "kızıl", \
"yeti smesine" = "yetişmesine"). "(kök tahmini)" ile işaretli eşleşmeler yanlış \
olabilir; bağlama uymayan sözlük önerilerini yok say.
7. ÇIKTI: Yalnızca istenen JSON şemasında, her satır için bir çeviri döndür. \
"id" değerlerini girdideki ile birebir aynı tut.
"""

# Sözlükte aranması anlamsız veya sesteşliğiyle yanıltıcı yaygın Türkçe kelimeler (normalize hâlde)
STOPWORDS = {
    "ve", "ile", "bir", "bu", "su", "o", "da", "de", "ki", "mi", "mu", "ne",
    "icin", "gibi", "kadar", "daha", "en", "her", "hem", "ya", "veya", "ise",
    "olan", "olarak", "oldu", "olur", "olup", "idi", "imis", "dahi", "pek",
    "ben", "sen", "biz", "siz", "onlar", "bunlar", "sunlar", "ama", "fakat",
    "lakin", "ancak", "diye", "sonra", "evvel", "bile", "yine", "artik",
    # Yaygın Türkçe kelimeler (sözlükteki âlî, vâr, şüş, karîn, âtıl, hakk vb. ile karışmaması için)
    "ali", "var", "yok", "sus", "karin", "kadin", "kiz", "ogul", "baba", "ana",
    "anne", "el", "bel", "bas", "goz", "yuz", "ayak", "kol", "kapi", "yol",
    "ev", "koy", "sehir", "su", "ates", "toprak", "hava", "gun", "gece", "yil",
    "sene", "ay", "yaz", "kis", "bugun", "yarin", "dun", "simdi", "burada",
    "orada", "nerede", "nasil", "niye", "neden", "kim", "kimse", "baska",
    "cok", "az", "buyuk", "kucuk", "uzun", "kisa", "iyi", "kotu", "guzel",
    "eski", "yeni", "genc", "yasli", "kara", "ak", "beyaz", "sari", "yesil",
    "kirmizi", "icinde", "ustunde", "altinda", "arasinda", "etrafta", "karsi",
    "dogru", "yanlis", "sadece", "yalniz", "hemen", "belki", "galiba",
    "dedi", "diyor", "demis", "oldum", "olmus", "olacak", "geldi", "gitti",
    "verdi", "aldi", "bakti", "gordu", "biliyorum", "bilirim", "bilmem",
    "kendi", "kendisi", "benim", "senin", "onun", "bizim", "sizin", "onlarin",
    "bana", "sana", "ona", "bize", "size", "onlara", "beni", "seni", "onu",
    "atin", "atilmis", "hakkindaki", "hakkinda", "uzerinde", "uzerine", "surusu",
    "gencine", "genci", "gence", "turk", "arap", "fars", "kabil", "taraf",
    "tarafin", "halde", "olmaz", "olur", "soz", "sozu", "sozler", "ses", "sesi",
}

TURKISH_ASCII_MAP = str.maketrans({
    "ı": "i", "İ": "i", "I": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ç": "c", "Ç": "c", "ö": "o", "Ö": "o", "ü": "u", "Ü": "u",
})
APOSTROPHES = "'’‘ʻʼʿʾ`´"
IZAFET_RE = re.compile(r"\s*-\s*([iıuüîû])(?=\s|$)", re.IGNORECASE)
TOKEN_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)

log = logging.getLogger("capa.sentetik")


# ---------------------------------------------------------------------------
# Normalizasyon ve sözlük indeksi
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Eşleştirme anahtarı: NFC → Türkçe-ASCII → diakritiksiz, küçük harf,
    kesme işaretsiz, izafet '-i' biçiminde, tek boşluklu."""
    text = unicodedata.normalize("NFC", text).translate(TURKISH_ASCII_MAP)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    for ap in APOSTROPHES:
        text = text.replace(ap, "")
    text = IZAFET_RE.sub("-i", text)  # "ab - ı adalet" / "ruy-ı" → "-i"
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class DictHit:
    key: str            # sözlükteki orijinal madde başı
    definition: str
    matched: str        # metinde eşleşen parça
    is_stem_guess: bool


class OttomanDictionary:
    def __init__(self, path: Path, min_stem_len: int = 4, max_def_chars: int = 220):
        with path.open(encoding="utf-8") as f:
            raw: dict[str, str] = json.load(f)
        self.min_stem_len = min_stem_len
        self.max_def_chars = max_def_chars
        self.index: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.max_ngram = 1
        for key, definition in raw.items():
            if not key or not definition:
                continue
            nk = normalize(key)
            if not nk:
                continue
            self.index[nk].append((unicodedata.normalize("NFC", key.strip()),
                                   unicodedata.normalize("NFC", str(definition).strip())))
            self.max_ngram = max(self.max_ngram, len(nk.split(" ")))
        self.max_ngram = min(self.max_ngram, 4)

        self._morph = None
        self._lemma_cache: dict[str, list[str]] = {}
        try:
            import zeyrek
            logging.getLogger("zeyrek").setLevel(logging.CRITICAL)
            self._morph = zeyrek.MorphAnalyzer()
        except Exception:
            pass
        log.info("Sözlük yüklendi: %d madde, %d normalize anahtar, max n-gram=%d, zeyrek=%s",
                 len(raw), len(self.index), self.max_ngram, bool(self._morph))

    def _clip(self, d: str) -> str:
        d = re.sub(r"\s+", " ", d)
        return d if len(d) <= self.max_def_chars else d[: self.max_def_chars].rstrip() + "…"

    @staticmethod
    def _has_circumflex(s: str) -> bool:
        return any(ch in s for ch in "âîûÂÎÛāīū")

    def _lemmatize_word(self, orig_word: str) -> list[str]:
        """Zeyrek morfolojik çözümleyici ile kelimenin kök/gövde adaylarını döndürür."""
        if not self._morph:
            return []
        w = orig_word.lower()
        if w in self._lemma_cache:
            return self._lemma_cache[w]
        try:
            lemmas = self._morph.lemmatize(w)[0][1]
        except Exception:
            lemmas = []
        # Özel isim (büyük harfli) kökleri ele, uzundan kısaya sırala
        valid = sorted(
            {normalize(lm) for lm in lemmas if not lm[:1].isupper() and len(lm) >= 4},
            key=len,
            reverse=True,
        )
        self._lemma_cache[w] = valid
        return valid

    def lookup_line(self, text: str, max_hits: int = 25,
                    allow_stem_guess: bool = True) -> list[DictHit]:
        norm = normalize(text)
        tokens = TOKEN_RE.findall(norm)
        orig_tokens = TOKEN_RE.findall(IZAFET_RE.sub(r"-\1", unicodedata.normalize("NFC", text)))
        if len(orig_tokens) != len(tokens):
            orig_tokens = tokens

        hits: list[DictHit] = []
        seen_keys: set[str] = set()

        def add(nk: str, matched: str, stem: bool) -> None:
            if nk in seen_keys:
                return
            for key, definition in self.index[nk][:2]:
                # ŞAPKA VE ÜNLÜ UYUMU KURALI (Sesteşlik Tuzağı Engeli):
                # 1) Eğer sözlükte aynı anahtarın hem şapkalı hem şapkasız maddesi varsa,
                #    metindeki kelimenin şapka durumuna uyanı seç (ör. âlemleri → âlem, alem değil).
                if len(self.index[nk]) > 1 and self._has_circumflex(matched) != self._has_circumflex(key):
                    continue
                # 2) Birebir eşleşmede (stem=False) metindeki kelime şapkasız ve <=5 harfliyse,
                #    sözlükteki şapkalı veya farklı ünlülü (ör. bütün != butûn, bazı != bâzî) maddelerle eşleştirme!
                if not stem and len(nk) <= 5 and not self._has_circumflex(matched):
                    if self._has_circumflex(key) or matched.lower()[:len(key)] != key.lower()[:len(key)]:
                        continue
                seen_keys.add(nk)
                # PDF OCR bozukluklarını ("k izil", "sava ş", "saklad ığı", "as ılı") temizle
                clean_def = re.sub(
                    r"(?<=[a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛ])\s+(?=[çğıöşü][a-zA-Zçğıöşüâîû]*\b)",
                    "",
                    definition,
                )
                hits.append(DictHit(key, self._clip(clean_def), matched, stem))

        i = 0
        while i < len(tokens) and len(hits) < max_hits:
            matched_len = 0
            # 1) Tamlama / çok kelimeli maddeler (uzundan kısaya)
            for n in range(min(self.max_ngram, len(tokens) - i), 1, -1):
                cand = " ".join(tokens[i:i + n])
                if cand in self.index:
                    add(cand, " ".join(orig_tokens[i:i + n]), False)
                    matched_len = n
                    break
            if matched_len:
                i += matched_len
                continue

            tok, orig = tokens[i], orig_tokens[i]
            i += 1
            if tok in STOPWORDS or len(tok) < 3:
                continue
            # Cümle ortasında büyük harfle başlayan özel isimleri (Hacer, Ali, Hürmüz, Viktor) sözlükte arama
            if i > 1 and orig[:1].isupper() and "-" not in orig:
                continue
            # 2) İzafetli kelime: "kabil-i" → "kabil"; "elviye-i" → "elviye"
            if "-" in tok:
                parts = [p for p in tok.split("-") if p and p != "i"]
                for p in parts:
                    if p in self.index and p not in STOPWORDS:
                        add(p, orig, False)
                continue
            # 3) Birebir kelime
            if len(tok) >= 4 and tok in self.index:
                add(tok, orig, False)
                continue
            # 4) Zeyrek Morfolojik Kök Bulucu (ör. izdivaçtan → izdivaç, intizardayım → intizar)
            lemmas = self._lemmatize_word(orig)
            # Eğer kelimenin kendisi zaten Türkçe bir kök olarak tanınıyorsa (ör. rafadan), başka köke bakma
            if tok in lemmas:
                continue
            for lm in lemmas:
                if lm in self.index and lm not in STOPWORDS and len(lm) >= 4:
                    add(lm, orig, True)
                    break
        return hits[:max_hits]


# ---------------------------------------------------------------------------
# Prompt oluşturma
# ---------------------------------------------------------------------------
def format_hits(hits: list[DictHit]) -> str:
    if not hits:
        return "    (eşleşme yok)"
    lines = []
    for h in hits:
        tag = " (kök tahmini)" if h.is_stem_guess else ""
        lines.append(f'    - "{h.matched}" → {h.key}{tag}: {h.definition}')
    return "\n".join(lines)


def build_batch_prompt(items: list[tuple[int, str]], dictionary: OttomanDictionary) -> str:
    blocks = []
    for idx, text in items:
        hits = dictionary.lookup_line(text)
        blocks.append(
            f"[id={idx}]\n"
            f"  Metin: {text}\n"
            f"  Sözlük bağlamı:\n{format_hits(hits)}"
        )
    return (
        "Aşağıdaki Latin harfli Osmanlıca satırları günümüz Türkçesine çevir. "
        "Satırlar aynı belgeden ardışık olabilir; komşu satırları yalnızca bağlamı "
        "anlamak için kullan, her satırı kendi içeriğiyle ayrı ayrı çevir.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nToplam {len(items)} satır var; tam olarak {len(items)} çeviri döndür."
    )


# ---------------------------------------------------------------------------
# Gemini istemcisi + retry
# ---------------------------------------------------------------------------
class RateLimiter:
    """Basit istemci taraflı RPM sınırlayıcı."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        delta = time.monotonic() - self._last
        if delta < self.interval:
            time.sleep(self.interval - delta)
        self._last = time.monotonic()


class BadModelOutput(Exception):
    """Model yanıtı boş / şemaya aykırı."""


class FatalAPIError(Exception):
    """Yeniden denemenin anlamsız olduğu hatalar (geçersiz anahtar, model yok...)."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (BadModelOutput, TimeoutError, ConnectionError)):
        return True
    try:
        from google.genai import errors as genai_errors
        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", None)
            return code in (404, 408, 409, 429) or (code is not None and code >= 500)
    except ImportError:
        pass
    try:
        import httpx
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True
    except ImportError:
        pass
    return False


class GeminiTranslator:
    DEFAULT_POOL = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]

    def __init__(self, model: str, temperature: float, rpm: float,
                 timeout_s: int, max_attempts: int):
        try:
            from google import genai
            from google.genai import types
            from pydantic import BaseModel
        except ImportError as e:  # pragma: no cover
            sys.exit(f"Eksik paket: {e}. Kurulum: pip install google-genai tenacity")

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            sys.exit("GEMINI_API_KEY (veya GOOGLE_API_KEY) ortam değişkeni tanımlı değil.")

        class Ceviri(BaseModel):
            id: int
            ceviri: str

        class BatchCevirisi(BaseModel):
            ceviriler: list[Ceviri]

        self._types = types
        self._schema = BatchCevirisi
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_s * 1000),
        )
        user_models = [m.strip() for m in model.split(",") if m.strip()]
        self.models = list(dict.fromkeys(user_models + self.DEFAULT_POOL))
        self.model_idx = 0
        self._lock = threading.Lock()
        self.model = self.models[0]
        self.temperature = temperature
        self.limiter = RateLimiter(rpm)

        self._call = retry(
            reraise=True,
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential_jitter(initial=0.5, max=10, jitter=1),
            stop=stop_after_attempt(max_attempts),
            before_sleep=before_sleep_log(log, logging.WARNING),
        )(self._call_once)

    def _current_model(self) -> str:
        with self._lock:
            m = self.models[self.model_idx % len(self.models)]
            self.model = m
            return m

    def _rotate_model(self) -> str:
        with self._lock:
            self.model_idx = (self.model_idx + 1) % len(self.models)
            m = self.models[self.model_idx]
            self.model = m
            return m

    def _call_once(self, prompt: str, expected_ids: list[int]) -> dict[int, str]:
        self.limiter.wait()
        types = self._types
        cur_model = self._current_model()
        is_gemma = cur_model.startswith("gemma-")
        try:
            if is_gemma:
                full_prompt = (
                    f"{GEMINI_SYSTEM_INSTRUCTION}\n\n"
                    f"{prompt}\n\n"
                    'Yanıtı YALNIZCA geçerli bir JSON nesnesi olarak ver: '
                    '{"ceviriler": [{"id": <int>, "ceviri": "<metin>"}, ...]}'
                )
                resp = self.client.models.generate_content(
                    model=cur_model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(temperature=self.temperature),
                )
            else:
                resp = self.client.models.generate_content(
                    model=cur_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=GEMINI_SYSTEM_INSTRUCTION,
                        temperature=self.temperature,
                        response_mime_type="application/json",
                        response_schema=self._schema,
                    ),
                )
        except Exception as exc:
            msg = str(exc)
            if "GenerateRequestsPerDay" in msg or "RESOURCE_EXHAUSTED" in msg or "404" in msg:
                with self._lock:
                    if cur_model in self.models and len(self.models) > 1:
                        self.models.remove(cur_model)
                        self.model_idx = self.model_idx % len(self.models)
                        log.warning("Model günlük kotası doldu / devre dışı, havuzdan çıkarıldı: %s (kalan=%s)",
                                    cur_model, self.models)
            else:
                self._rotate_model()
            if _is_retryable(exc):
                raise
            raise FatalAPIError(msg) from exc

        parsed = getattr(resp, "parsed", None)
        if parsed is None:
            txt = (getattr(resp, "text", None) or "").strip()
            if not txt:
                raise BadModelOutput(f"Boş yanıt ({cur_model}).")
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if m:
                txt = m.group(0)
            try:
                parsed = self._schema.model_validate_json(txt)
            except Exception as exc:
                raise BadModelOutput(f"JSON ayrıştırılamadı ({cur_model}): {exc}") from exc

        result: dict[int, str] = {}
        for item in parsed.ceviriler:
            text = re.sub(r"\s+", " ", (item.ceviri or "")).strip()
            if item.id in expected_ids and text:
                result[item.id] = text
        if not result:
            raise BadModelOutput(f"Yanıtta geçerli çeviri yok ({cur_model}).")
        return result

    def translate(self, items: list[tuple[int, str]],
                  dictionary: OttomanDictionary) -> dict[int, str]:
        ids = [i for i, _ in items]
        prompt = build_batch_prompt(items, dictionary)
        return self._call(prompt, ids)


# ---------------------------------------------------------------------------
# Checkpoint ve G/Ç
# ---------------------------------------------------------------------------
def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("metadata.jsonl satır %d bozuk, atlandı.", ln)
    return rows


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return {"next_index": 0, "output_lines": 0, "written": 0, "skipped": 0, "failed": 0}


def save_checkpoint(path: Path, state: dict) -> None:
    state = {**state, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomik


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


def reconcile_output(path: Path, expected_lines: int) -> None:
    """Checkpoint kaydedilmeden önce kesinti olduysa çıktıdaki fazla satırları kırp."""
    actual = count_lines(path)
    if actual == expected_lines:
        return
    if actual < expected_lines:
        log.warning("Çıktı dosyasında %d satır var, checkpoint %d bekliyor. "
                    "Mevcut dosya korunuyor.", actual, expected_lines)
        return
    log.warning("Yarım kalmış batch tespit edildi: %d fazla satır kırpılıyor.",
                actual - expected_lines)
    with path.open(encoding="utf-8") as f:
        kept = [ln for ln in f if ln.strip()][:expected_lines]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.writelines(kept)
    os.replace(tmp, path)


def make_record(osmanlica: str, turkce: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": f"{USER_PREFIX}{osmanlica}"},
            {"role": "assistant", "content": turkce},
        ]
    }


def append_jsonl(path: Path, records: Iterable[dict]) -> int:
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    return n


def clean_source_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def is_translatable(text: str) -> bool:
    return sum(ch.isalpha() for ch in text) >= 2


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    rows = load_rows(args.metadata)
    dictionary = OttomanDictionary(args.dictionary)
    total = len(rows)
    end = min(total, args.start + args.limit) if args.limit else total

    if args.dry_run:
        items = [(i, clean_source_text(rows[i].get("text", ""))) for i in range(args.start, end)]
        items = [(i, t) for i, t in items if is_translatable(t)]
        for b in range(0, len(items), args.batch_size):
            print("=" * 80)
            print(build_batch_prompt(items[b:b + args.batch_size], dictionary))
        return

    if args.reset:
        for p in (args.output, args.checkpoint):
            if p.exists():
                p.unlink()
        log.info("Checkpoint ve çıktı sıfırlandı.")

    state = load_checkpoint(args.checkpoint)
    reconcile_output(args.output, state.get("output_lines", 0))
    start = max(state.get("next_index", 0), args.start)
    if start >= end:
        log.info("Tamamlanmış görünüyor (next_index=%d / %d).", start, end)
        return
    log.info("Başlangıç index: %d / %d (model=%s, batch=%d)",
             start, end, args.model, args.batch_size)

    translator = GeminiTranslator(args.model, args.temperature, args.rpm,
                                  args.timeout, args.max_attempts)
    t0 = time.time()
    processed_since_start = 0
    idx = start

    def _process_one_batch(b_start: int, b_end: int):
        items, skipped = [], 0
        for i in range(b_start, b_end):
            text = clean_source_text(rows[i].get("text", ""))
            if is_translatable(text):
                items.append((i, text))
            else:
                skipped += 1
        results: dict[int, str] = {}
        if items:
            try:
                results = translator.translate(items, dictionary)
            except FatalAPIError as exc:
                log.error("Kurtarılamaz API hatası: %s", exc)
                raise
            except Exception as exc:
                log.warning("Batch %d-%d başarısız (%s); satır satır deneniyor.",
                            b_start, b_end - 1, exc)
            missing = [(i, t) for i, t in items if i not in results]
            for i, t in missing:
                try:
                    results.update(translator.translate([(i, t)], dictionary))
                except FatalAPIError:
                    raise
                except Exception as exc:
                    log.error("Satır %d çevrilemedi: %s", i, exc)
        return b_start, b_end, items, skipped, results

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while idx < end:
                futures = []
                cur = idx
                for _ in range(args.workers):
                    if cur >= end:
                        break
                    b_end = min(cur + args.batch_size, end)
                    futures.append(pool.submit(_process_one_batch, cur, b_end))
                    cur = b_end

                for fut in futures:
                    b_start, b_end, items, skipped, results = fut.result()
                    records = [make_record(t, results[i]) for i, t in items if i in results]
                    for i, t in items:
                        if i in results:
                            log.info("[%d] %s  →  %s", i, t, results[i])
                    failed = [{"index": i, "file_name": rows[i].get("file_name"), "text": t}
                              for i, t in items if i not in results]

                    written = append_jsonl(args.output, records) if records else 0
                    if failed:
                        append_jsonl(args.failed, failed)

                    state.update(
                        next_index=b_end,
                        output_lines=state.get("output_lines", 0) + written,
                        written=state.get("written", 0) + written,
                        skipped=state.get("skipped", 0) + skipped,
                        failed=state.get("failed", 0) + len(failed),
                        model=args.model,
                        total=end,
                    )
                    save_checkpoint(args.checkpoint, state)
                    processed_since_start += b_end - b_start
                    idx = b_end

                elapsed = time.time() - t0
                rate = processed_since_start / elapsed if elapsed else 0
                eta_min = (end - idx) / rate / 60 if rate else float("inf")
                log.info("İlerleme %d/%d (%.1f%%) | yazılan=%d atlanan=%d başarısız=%d | "
                         "%.2f satır/sn | kalan ~%.1f dk",
                         idx, end, 100 * idx / end, state["written"], state["skipped"],
                         state["failed"], rate, eta_min)
    except KeyboardInterrupt:
        log.warning("Kullanıcı tarafından durduruldu. Son checkpoint: next_index=%d. "
                    "Tekrar çalıştırınca buradan devam edilecek.", state.get("next_index", 0))
        return

    log.info("Tamamlandı. Çıktı: %s (%d kayıt). Başarısız satırlar: %s",
             args.output, state["written"], args.failed if state["failed"] else "yok")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Çapa AI - Faz 3 sentetik paralel veri üretimi")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--dictionary", type=Path, default=DEFAULT_DICT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--failed", type=Path, default=DEFAULT_FAILED)
    p.add_argument("--model", default="gemini-3.5-flash-lite,gemini-3.1-flash-lite",
                   help="Virgülle ayrılmış model havuzu")
    p.add_argument("--batch-size", type=int, default=20,
                   help="İstek başına satır sayısı (1 = her satır ayrı istek)")
    p.add_argument("--workers", type=int, default=4,
                   help="Paralel çalışacak iş parçacığı sayısı")
    p.add_argument("--rpm", type=float, default=0,
                   help="Dakikadaki maksimum istek (0 = sınırsız; ücretsiz katman için ~10-15)")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout", type=int, default=25, help="İstek zaman aşımı (sn)")
    p.add_argument("--max-attempts", type=int, default=12, help="Retry üst sınırı")
    p.add_argument("--start", type=int, default=0, help="Minimum başlangıç index'i")
    p.add_argument("--limit", type=int, default=0, help="İşlenecek satır sayısı (0 = tümü)")
    p.add_argument("--reset", action="store_true", help="Checkpoint ve çıktıyı sil, baştan başla")
    p.add_argument("--dry-run", action="store_true", help="API çağırmadan promptları yazdır")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


class SyncFileHandler(logging.FileHandler):
    """Her log satırında os.fsync çağırarak Windows editörlerinin anında yenilenmesini sağlar."""

    def flush(self) -> None:
        super().flush()
        if self.stream and hasattr(self.stream, "fileno"):
            try:
                os.fsync(self.stream.fileno())
            except OSError:
                pass


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    _args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if _args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            SyncFileHandler(BASE_DIR / "sentetik_veri_uretimi.log", encoding="utf-8"),
        ],
    )
    for noisy in ("httpx", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(_args)
