"""SQLite response cache keyed by a hash of (model, messages, params).

With temperature 0 and pinned models this makes eval reruns free and reproducible,
and lets CI replay a smoke eval without any API key.
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


class CacheMiss(RuntimeError):
    """Raised in replay-only mode (CI) when a call is not in the committed cache."""


def prompt_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(value))
            )
            self._conn.commit()
