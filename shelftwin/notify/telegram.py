"""Telegram Bot API notifier (optional). Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."""
from __future__ import annotations

import json
import os
import threading
import urllib.request

from .messages import task_message


class TelegramNotifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None, lang: str = "en", slot_lookup=None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.lang = lang
        self.slot_lookup = slot_lookup or (lambda sid: {})

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def __call__(self, task, event: str):
        if not self.enabled or event not in ("created", "escalated", "reopened"):
            return
        text = task_message(task.to_dict(), self.slot_lookup(task.slot_id), self.lang)
        if event != "created":
            text = f"[{event.upper()}] " + text
        threading.Thread(target=self._send, args=(text,), daemon=True).start()

    def _send(self, text: str):
        data = json.dumps({"chat_id": self.chat_id, "text": text}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{self.token}/sendMessage", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:
            print("telegram send failed:", e)
