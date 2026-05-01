"""Lineage helpers for ingested artifacts (hashes, paths, timestamps)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactRef(BaseModel):
    path: str
    sha256: str | None = None
    size_bytes: int | None = None

    @classmethod
    def from_path(cls, path: Path, root: Path | None = None, hash_file: bool = True) -> ArtifactRef:
        resolved = path.resolve()
        display_path = resolved if root is None else resolved.relative_to(root.resolve())
        if not resolved.exists():
            return cls(path=str(display_path))
        return cls(
            path=str(display_path),
            sha256=sha256_file(resolved) if hash_file and resolved.is_file() else None,
            size_bytes=resolved.stat().st_size if resolved.is_file() else None,
        )


