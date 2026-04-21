"""Packaging utilities for exporting trained artifacts."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .manifest import ArtifactRecord, Manifest, MANIFEST_FILENAME


@dataclass
class RegisteredArtifact:
	source_path: Path
	logical_name: str
	target_path: Path
	metadata: Dict[str, Any] = field(default_factory=dict)


class PackageExporter:
	"""Coordinates staging, manifest creation, and archive export."""

	def __init__(
		self,
		package_name: str,
		output_dir: Path | str,
		*,
		compression: str = "zip",
		metadata: Optional[Dict[str, Any]] = None,
		keep_staging: bool = False,
	) -> None:
		self.package_name = package_name
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.compression = compression.lower()
		self.metadata = metadata or {}
		self.keep_staging = keep_staging
		self._registered: List[RegisteredArtifact] = []
		self._staging_dir: Optional[Path] = None

	@property
	def staging_dir(self) -> Path:
		if self._staging_dir is None:
			self._staging_dir = Path(
				tempfile.mkdtemp(prefix=f"{self.package_name}_", dir=str(self.output_dir))
			)
		return self._staging_dir

	def add_file(
		self,
		source: Path | str,
		*,
		logical_name: Optional[str] = None,
		dest_path: Optional[Path | str] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		source_path = Path(source)
		if not source_path.exists():
			raise FileNotFoundError(f"Artifact does not exist: {source_path}")

		relative_target = Path(dest_path) if dest_path else Path(source_path.name)
		logical = logical_name or relative_target.as_posix()
		staging_target = self.staging_dir / relative_target
		staging_target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(source_path, staging_target)

		self._registered.append(
			RegisteredArtifact(
				source_path=source_path,
				logical_name=logical,
				target_path=staging_target.relative_to(self.staging_dir),
				metadata=metadata or {},
			)
		)

	def add_directory(
		self,
		source: Path | str,
		*,
		logical_prefix: Optional[str] = None,
		include_hidden: bool = False,
	) -> None:
		source_path = Path(source)
		if not source_path.is_dir():
			raise NotADirectoryError(f"Directory required: {source_path}")

		for item in source_path.rglob("*"):
			if item.is_dir():
				continue
			if not include_hidden and item.name.startswith("."):
				continue
			relative = item.relative_to(source_path)
			if logical_prefix:
				logical_name = f"{logical_prefix}/{relative.as_posix()}"
				dest_path = Path(logical_prefix) / relative
			else:
				logical_name = relative.as_posix()
				dest_path = relative
			self.add_file(item, logical_name=logical_name, dest_path=dest_path)

	def build_manifest(self) -> Manifest:
		artifacts: List[ArtifactRecord] = []
		for entry in self._registered:
			staged_path = self.staging_dir / entry.target_path
			artifacts.append(
				ArtifactRecord.from_file(
					entry.logical_name,
					entry.target_path,
					staged_path,
					metadata=entry.metadata,
				)
			)
		return Manifest.create(
			package_name=self.package_name,
			artifacts=artifacts,
			metadata=self.metadata,
		)

	def finalize(self) -> Path:
		if not self._registered:
			raise RuntimeError("No artifacts registered for export")

		manifest = self.build_manifest()
		manifest_path = self.staging_dir / MANIFEST_FILENAME
		manifest.save(manifest_path)

		package_path = self._build_archive()

		if not self.keep_staging:
			shutil.rmtree(self.staging_dir, ignore_errors=True)
			self._staging_dir = None

		return package_path

	def _build_archive(self) -> Path:
		package_filename = f"{self.package_name}.{self._archive_extension()}"
		package_path = self.output_dir / package_filename

		if self.compression == "zip":
			with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
				for file_path in self.staging_dir.rglob("*"):
					archive.write(file_path, file_path.relative_to(self.staging_dir).as_posix())
		elif self.compression in {"tar", "tar.gz", "tgz"}:
			mode = "w:gz" if self.compression != "tar" else "w"
			with tarfile.open(package_path, mode) as archive:
				for file_path in self.staging_dir.rglob("*"):
					archive.add(file_path, arcname=file_path.relative_to(self.staging_dir))
		else:
			raise ValueError(f"Unsupported compression type: {self.compression}")

		return package_path

	def _archive_extension(self) -> str:
		if self.compression == "zip":
			return "zip"
		if self.compression == "tar":
			return "tar"
		if self.compression in {"tar.gz", "tgz"}:
			return "tar.gz"
		return self.compression


class PackageVerifier:
	"""Utility for verifying exported packages."""

	def __init__(self, package_path: Path | str) -> None:
		self.package_path = Path(package_path)
		if not self.package_path.exists():
			raise FileNotFoundError(package_path)

	def extract_manifest(self) -> Manifest:
		with tempfile.TemporaryDirectory() as tmp_dir:
			tmp_path = Path(tmp_dir)
			manifest_path = tmp_path / MANIFEST_FILENAME
			self._extract_manifest(tmp_path)
			if not manifest_path.exists():
				raise FileNotFoundError("Manifest not found in package")
			return Manifest.load(manifest_path)

	def verify(self) -> List[str]:
		with tempfile.TemporaryDirectory() as tmp_dir:
			tmp_path = Path(tmp_dir)
			self._extract_all(tmp_path)
			manifest_file = tmp_path / MANIFEST_FILENAME
			if not manifest_file.exists():
				return ["Manifest missing from package"]
			manifest = Manifest.load(manifest_file)
			return manifest.validate(tmp_path)

	def _extract_manifest(self, destination: Path) -> None:
		if self.package_path.suffix == ".zip":
			with zipfile.ZipFile(self.package_path, "r") as archive:
				for name in archive.namelist():
					if name.endswith(MANIFEST_FILENAME):
						archive.extract(name, destination)
						return
		else:
			mode = "r:gz" if self.package_path.suffix in {".gz", ".tgz"} else "r"
			with tarfile.open(self.package_path, mode) as archive:
				try:
					archive.extract(MANIFEST_FILENAME, destination)
				except KeyError:
					pass

	def _extract_all(self, destination: Path) -> None:
		if self.package_path.suffix == ".zip":
			with zipfile.ZipFile(self.package_path, "r") as archive:
				archive.extractall(destination)
		else:
			mode = "r:gz" if self.package_path.suffix in {".gz", ".tgz"} else "r"
			with tarfile.open(self.package_path, mode) as archive:
				archive.extractall(destination)


__all__ = [
	"PackageExporter",
	"PackageVerifier",
]
