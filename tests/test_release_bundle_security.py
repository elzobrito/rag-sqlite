from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pytest


def _load_builder(repo_root: Path):
    spec = importlib.util.spec_from_file_location(
        "build_release_security", repo_root / "scripts" / "build_release.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validated_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
            raise ValueError(f"unsafe archive member: {info.filename}")
        members.append(info)
    return members


def _verify_internal_checksums(root: Path) -> None:
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"checksum mismatch: {name}")


def _run(runtime: Path, db: Path, *args: str) -> dict:
    process = subprocess.run(
        [sys.executable, str(runtime), "--db", str(db), "--compact", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(process.stdout)


def test_archive_policy_rejects_traversal_and_symlink(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape", "bad")
    with zipfile.ZipFile(unsafe) as archive:
        with pytest.raises(ValueError, match="unsafe archive member"):
            _validated_members(archive)

    symlink = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("bundle/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(info, "../../target")
    with zipfile.ZipFile(symlink) as archive:
        with pytest.raises(ValueError, match="unsafe archive member"):
            _validated_members(archive)


def test_checksum_corruption_is_rejected(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    builder = _load_builder(repo_root)
    archive_path, _ = builder.build_release(repo_root, tmp_path / "dist", "0.1.0")
    extract_root = tmp_path / "extract"
    with zipfile.ZipFile(archive_path) as archive:
        members = _validated_members(archive)
        archive.extractall(extract_root, members=members)
    root = extract_root / "rag-sqlite-v0.1.0"
    _verify_internal_checksums(root)
    (root / "rag_sqlite.py").write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _verify_internal_checksums(root)


def test_clean_bundle_runs_schema_and_offline_hash_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    builder = _load_builder(repo_root)
    archive_path, _ = builder.build_release(repo_root, tmp_path / "dist", "0.1.0")
    extract_root = tmp_path / "extract"
    with zipfile.ZipFile(archive_path) as archive:
        members = _validated_members(archive)
        archive.extractall(extract_root, members=members)

    root = extract_root / "rag-sqlite-v0.1.0"
    _verify_internal_checksums(root)
    runtime = root / "rag_sqlite.py"
    db = tmp_path / "kb.sqlite"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "alpha.txt").write_text(
        "Data mesh uses domain-oriented ownership and data products.",
        encoding="utf-8",
    )

    schema = _run(runtime, db, "schema")
    assert schema["ok"] is True
    configured = _run(runtime, db, "--create", "config", "set", "embedding_provider", "hash")
    assert configured["ok"] is True
    indexed = _run(runtime, db, "index", str(fixtures))
    assert indexed["ok"] is True
    query = _run(runtime, db, "query", "data mesh", "--top-k", "3")
    assert query["ok"] is True
    assert query["meta"]["hit_count"] >= 1
    assert query["hits"][0]["filename"] == "alpha.txt"
