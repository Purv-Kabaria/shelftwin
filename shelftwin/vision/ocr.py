"""OCR tie-break for ambiguous crops (spec Section 3.6 / 6.6).

Token matching is dependency-free; PaddleOCR is optional and loaded lazily.
"""
from __future__ import annotations

import re

_UNIT_ALIASES = {
    "gm": "g", "gms": "g", "gram": "g", "grams": "g", "gr": "g",
    "kgs": "kg", "kilo": "kg", "ltr": "l", "litre": "l", "liter": "l", "ltrs": "l",
    "mls": "ml", "millilitre": "ml",
}


def normalise_tokens(text: str) -> set[str]:
    t = text.lower()
    t = re.sub(r"(\d+(?:\.\d+)?)\s*([a-z]+)", lambda m: m.group(1) + _UNIT_ALIASES.get(m.group(2), m.group(2)), t)
    toks = re.findall(r"[a-z0-9.]+", t)
    return {_UNIT_ALIASES.get(x, x) for x in toks if len(x) > 1 or x.isdigit()}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def ocr_tiebreak(ocr_text: str, candidates: list[tuple[str, str]], min_score: float = 0.2) -> tuple[str | None, float]:
    """candidates: [(sku_id, catalogue_text)]. Returns best sku (or None) and its Jaccard score."""
    toks = normalise_tokens(ocr_text)
    scored = sorted(((jaccard(toks, normalise_tokens(txt)), sku) for sku, txt in candidates), reverse=True)
    if not scored or scored[0][0] < min_score:
        return None, scored[0][0] if scored else 0.0
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.05:
        return None, scored[0][0]  # still ambiguous: abstain
    return scored[0][1], scored[0][0]


class PackOCR:
    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(lang=lang)

    def read(self, bgr) -> str:
        res = self.ocr.predict(bgr) if hasattr(self.ocr, "predict") else self.ocr.ocr(bgr)
        texts = []
        for r in res or []:
            if isinstance(r, dict):
                texts += r.get("rec_texts", [])
            elif isinstance(r, list):
                texts += [x[1][0] for x in r if x]
        return " ".join(texts)
