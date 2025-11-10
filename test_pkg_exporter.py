import json
from pathlib import Path

import pytest

from core.pkg import (
    MANIFEST_FILENAME,
    ArtifactRecord,
    Manifest,
    PackageExporter,
    PackageVerifier,
)


def test_package_exporter_creates_archive_with_manifest(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    sample_file = source_dir / "weights.bin"
    sample_file.write_bytes(b"binary-weights")

    exporter = PackageExporter(
        package_name="test_package",
        output_dir=tmp_path / "dist",
    )
    exporter.add_file(
        sample_file,
        logical_name="model_weights",
        dest_path=Path("artifacts") / sample_file.name,
        metadata={"type": "weights"},
    )

    package_path = exporter.finalize()

    assert package_path.exists()
    verifier = PackageVerifier(package_path)
    errors = verifier.verify()
    assert errors == []

    manifest = verifier.extract_manifest()
    assert manifest.package_name == "test_package"
    assert manifest.artifacts[0].logical_name == "model_weights"
    assert manifest.artifacts[0].metadata["type"] == "weights"


def test_manifest_validation_detects_corrupted_artifact(tmp_path):
    artifact_path = tmp_path / "artifact.bin"
    artifact_path.write_bytes(b"original")

    record = ArtifactRecord.from_file(
        logical_name="artifact",
        relative_path=Path("artifact.bin"),
        source_path=artifact_path,
    )
    manifest = Manifest.create("test_package", [record])
    manifest_path = tmp_path / MANIFEST_FILENAME
    manifest.save(manifest_path)

    assert manifest.validate(tmp_path) == []

    artifact_path.write_bytes(b"tampered")
    errors = manifest.validate(tmp_path)
    assert errors and "Digest mismatch" in errors[0]
