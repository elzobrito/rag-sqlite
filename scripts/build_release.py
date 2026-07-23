#!/usr/bin/env python3
"""Build a deterministic, data-free rag-sqlite release archive."""

from __future__ import annotations

import argparse
import hashlib
import stat
import zipfile
from pathlib import Path

FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)
PAYLOAD_FILES = ("rag_sqlite.py", "requirements.txt", "LICENSE")

SH_LAUNCHER = """#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
exec "$PYTHON_BIN" "$SCRIPT_DIR/rag_sqlite.py" "$@"
"""

CMD_LAUNCHER = """@echo off
setlocal
if defined PYTHON_BIN (
  "%PYTHON_BIN%" "%~dp0rag_sqlite.py" %*
  exit /b %ERRORLEVEL%
)
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  py -3 "%~dp0rag_sqlite.py" %*
  exit /b %ERRORLEVEL%
)
python "%~dp0rag_sqlite.py" %*
"""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def zip_info(name: str, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    mode = (stat.S_IFREG | (0o755 if executable else 0o644)) << 16
    info.external_attr = mode
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    return info


def build_release(repo_root: Path, output_dir: Path, version: str) -> tuple[Path, Path]:
    repo_root = repo_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"rag-sqlite-v{version}"
    archive_path = output_dir / f"{prefix}.zip"
    sums_path = output_dir / "SHA256SUMS"

    payload: dict[str, bytes] = {}
    for relative in PAYLOAD_FILES:
        source = repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"required release file missing: {relative}")
        payload[relative] = source.read_bytes()
    payload["rag-sqlite"] = SH_LAUNCHER.encode("utf-8")
    payload["rag-sqlite.cmd"] = CMD_LAUNCHER.replace("\n", "\r\n").encode("utf-8")

    internal_sums = "".join(
        f"{sha256_bytes(content)}  {name}\n" for name, content in sorted(payload.items())
    ).encode("utf-8")
    payload["SHA256SUMS"] = internal_sums

    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, content in sorted(payload.items()):
            archive.writestr(
                zip_info(f"{prefix}/{name}", executable=name == "rag-sqlite"),
                content,
            )

    archive_hash = sha256_bytes(archive_path.read_bytes())
    sums_path.write_text(
        f"{archive_hash}  {archive_path.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    return archive_path, sums_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("--version", default="0.1.0")
    args = parser.parse_args()
    archive, sums = build_release(args.repo_root, args.output_dir, args.version)
    print(archive)
    print(sums)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
