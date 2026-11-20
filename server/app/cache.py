import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def digest(*parts: str) -> str:
    """Short stable hash of the inputs, so a cache entry is invalidated when any input changes."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:24]


class DiskCache:
    """Tiny JSON-file cache: loaded once, written on every set (atomic replace)."""

    def __init__(self, name: str):
        self.path = CACHE_DIR / f"{name}.json"
        self._data: dict | None = None

    def _load(self) -> dict:
        if self._data is None:
            try:
                self._data = json.loads(self.path.read_text())
            except (OSError, ValueError):
                self._data = {}
        return self._data

    def get(self, key: str):
        return self._load().get(key)

    def set(self, key: str, value) -> None:
        self._load()[key] = value
        CACHE_DIR.mkdir(exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.replace(self.path)
