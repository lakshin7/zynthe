"""Structured package manifest utilities."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = "1.0"


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_environment() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def compute_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ArtifactRecord:
    """Describes a single artifact tracked in a package."""

    logical_name: str
    relative_path: str
    size: int
    sha256: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        logical_name: str,
        relative_path: Path,
        source_path: Path,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ArtifactRecord":
        stats = source_path.stat()
        return cls(
            logical_name=logical_name,
            relative_path=str(relative_path).replace("\\", "/"),
            size=int(stats.st_size),
            sha256=compute_sha256(source_path),
            metadata=metadata or {},
        )


@dataclass
class Manifest:
    """High-level manifest describing a packaged delivery."""

    package_name: str
    version: str
    created_at: str
    artifacts: List[ArtifactRecord]
    environment: Dict[str, Any] = field(default_factory=_default_environment)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        package_name: str,
        artifacts: Iterable[ArtifactRecord],
        *,
        version: str = MANIFEST_VERSION,
        metadata: Optional[Dict[str, Any]] = None,
        environment: Optional[Dict[str, Any]] = None,
    ) -> "Manifest":
        return cls(
            package_name=package_name,
            version=version,
            created_at=_now_iso(),
            artifacts=list(artifacts),
            environment=environment or _default_environment(),
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_name": self.package_name,
            "version": self.version,
            "created_at": self.created_at,
            "environment": self.environment,
            "metadata": self.metadata,
            "artifacts": [
                {
                    "logical_name": record.logical_name,
                    "relative_path": record.relative_path,
                    "size": record.size,
                    "sha256": record.sha256,
                    "metadata": record.metadata,
                }
                for record in self.artifacts
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, source: Path) -> "Manifest":
        payload = json.loads(source.read_text(encoding="utf-8"))
        artifacts = [
            ArtifactRecord(
                logical_name=item["logical_name"],
                relative_path=item["relative_path"],
                size=int(item["size"]),
                sha256=item["sha256"],
                metadata=item.get("metadata", {}),
            )
            for item in payload.get("artifacts", [])
        ]
        return cls(
            package_name=payload["package_name"],
            version=payload.get("version", MANIFEST_VERSION),
            created_at=payload.get("created_at", _now_iso()),
            artifacts=artifacts,
            environment=payload.get("environment", _default_environment()),
            metadata=payload.get("metadata", {}),
        )

    def validate(self, base_dir: Path) -> List[str]:
        errors: List[str] = []
        for record in self.artifacts:
            file_path = base_dir / record.relative_path
            if not file_path.exists():
                errors.append(f"Missing artifact: {record.relative_path}")
                continue
            if int(file_path.stat().st_size) != record.size:
                errors.append(f"Size mismatch for {record.relative_path}")
                continue
            actual_digest = compute_sha256(file_path)
            if actual_digest != record.sha256:
                errors.append(f"Digest mismatch for {record.relative_path}")
        return errors

    def artifact_map(self) -> Dict[str, ArtifactRecord]:
        return {record.logical_name: record for record in self.artifacts}

    def diff(self, other: "Manifest") -> Dict[str, Any]:
        delta: Dict[str, Any] = {"added": [], "removed": [], "changed": []}
        self_map = self.artifact_map()
        other_map = other.artifact_map()

        for key in other_map:
            if key not in self_map:
                delta["added"].append(key)
            elif other_map[key].sha256 != self_map[key].sha256:
                delta["changed"].append(key)

        for key in self_map:
            if key not in other_map:
                delta["removed"].append(key)

        return delta


__all__ = [
    "ArtifactRecord",
    "Manifest",
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "compute_sha256",
]
