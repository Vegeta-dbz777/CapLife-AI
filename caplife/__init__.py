"""CapLife AI: research-only AEC neural RUL prototype."""
from __future__ import annotations

import hashlib
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def source_fingerprint() -> str:
    """Hash of every module's path, size and modification time in this package, as on disk now."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        stat = path.stat()
        digest.update(f"{path.relative_to(PACKAGE_DIR).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


# The code this process imported. ``app.py`` compares it with the files on disk to notice a
# repository update that the running server has not loaded yet.
LOADED_FINGERPRINT = source_fingerprint()
