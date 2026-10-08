"""Hashing and manifest helpers. Original sources are never mutated."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CHUNK = 1 << 20


def sha256_bytes(data):
    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"generated_utc": now_iso(), **payload}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return body


def source_entry(path):
    path = Path(path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
