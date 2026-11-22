"""A fingerprint of the app's code, models and sample data, used to key Streamlit caches.

Streamlit Community Cloud applies a repository update to the running server
without restarting it, and ``st.cache_data`` / ``st.cache_resource`` only
invalidate when the cached function's own source changes. A cache whose
inputs live in other modules or in artifact files would otherwise keep
serving results computed by the old code (for example the old trained model).
Passing ``code_version()`` as an argument makes every such cache refresh after
any update to these files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WATCHED_FOLDERS = ("caplife", "artifacts", "sample_data")


def code_version(root: Path = ROOT) -> str:
    """Short hash of every watched file's path, size and modification time."""
    digest = hashlib.sha256()
    for folder in WATCHED_FOLDERS:
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            stat = path.stat()
            digest.update(f"{path.relative_to(root).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:16]
