"""Lineage schemas used to keep generated studio-copilot artifacts traceable."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


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


class RunLineage(BaseModel):
    run_id: int
    source_dataset: str
    dataset_version: str
    source_files: list[ArtifactRef] = Field(default_factory=list)
    ingested_at: str = Field(default_factory=utc_now_iso)


class CandidateLineage(BaseModel):
    candidate_id: str
    baseline_run_id: int
    model_version: str
    feature_schema_version: str
    rule_set_version: str
    generated_at: str = Field(default_factory=utc_now_iso)
    source_run: RunLineage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
