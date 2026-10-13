from __future__ import annotations

import hashlib
import importlib.util
import stat
import zipfile
from pathlib import Path


def _load_builder(repo_root: Path):
    spec = importlib.util.spec_from_file_location(
        "build_release", repo_root / "scripts" / "build_release.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_bundle_is_reproducible_and_minimal(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    builder = _load_builder(repo_root)
    first, sums1 = builder.build_release(repo_root, tmp_path / "one", "0.1.0")
    second, sums2 = builder.build_release(repo_root, tmp_path / "two", "0.1.0")

    assert first.read_bytes() == second.read_bytes()
    assert sums1.read_text() == sums2.read_text()
    assert _sha(first) in sums1.read_text()

    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        expected = {
            "LICENSE",
            "SHA256SUMS",
            "rag-sqlite",
            "rag-sqlite.cmd",
            "rag_sqlite.py",
            "requirements.txt",
        }
        assert {Path(name).name for name in names} == expected
        assert not any(
            token in name.lower()
            for name in names
            for token in ("activity.jsonl", "kb.sqlite", "__pycache__", ".env")
        )
        launcher = archive.getinfo("rag-sqlite-v0.1.0/rag-sqlite")
        assert stat.S_IMODE(launcher.external_attr >> 16) == 0o755


def test_internal_checksums_cover_runtime_payload(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    builder = _load_builder(repo_root)
    archive_path, _ = builder.build_release(repo_root, tmp_path, "0.1.0")
    with zipfile.ZipFile(archive_path) as archive:
        prefix = "rag-sqlite-v0.1.0/"
        sums = archive.read(prefix + "SHA256SUMS").decode()
        for line in sums.splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read(prefix + name)).hexdigest() == digest
