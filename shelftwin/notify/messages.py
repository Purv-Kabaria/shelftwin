"""Staff task messages in English, Hindi and Gujarati (spec 3.9 / 3.10)."""
from __future__ import annotations

import json
from pathlib import Path

_I18N = Path(__file__).parent / "i18n"
_cache: dict[str, dict] = {}


def strings(lang: str) -> dict:
    if lang not in _cache:
        p = _I18N / f"{lang}.json"
        _cache[lang] = json.loads((p if p.exists() else _I18N / "en.json").read_text(encoding="utf-8"))
    return _cache[lang]


def task_message(task: dict, slot: dict, lang: str = "en") -> str:
    s = strings(lang)
    kind = s["kinds"].get(task["kind"], task["kind"])
    parts = [f"{kind} · {s['aisle']} {slot.get('aisle', '?')} · {s['shelf']} {slot.get('shelf', '?')} · "
             f"{s['slot']} {task['slot_id'].split('-')[-1]} · {s['priority']} {task['priority']:.0f}"]
    if slot.get("sku"):
        parts.append(slot["sku"] + ".")
    if task.get("backroom_location"):
        parts.append(s["bring_from"].format(loc=task["backroom_location"]))
    r = task.get("reason") or {}
    if r.get("clean_views"):
        parts.append(s["why"].format(views=r["clean_views"], span=r.get("span_s", 0),
                                     cov=round(100 * (r.get("coverage_now") or 0)),
                                     blocked=round(100 * (r.get("share_blocked") or 0))))
    return " ".join(parts)
