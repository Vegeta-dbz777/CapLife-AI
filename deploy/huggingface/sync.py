"""Mirror this repository to a Hugging Face Space (run by .github/workflows/huggingface-space.yml).

Copies every tracked file except the GitHub workflows and tests into a staging folder, puts the
Space's settings (deploy/huggingface/space-header.md) in front of README.md there, and uploads the
folder to the Space in one commit; files removed from GitHub are removed from the Space too. The
GitHub README itself never carries the Space settings.

    HF_TOKEN=... HF_SPACE=user/caplife-ai python deploy/huggingface/sync.py
    python deploy/huggingface/sync.py --dry-run   # build the staging folder and list it, upload nothing
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = (".github/", "tests/")
KEEP_ON_SPACE = {".gitattributes"}  # Hugging Face's own file


def stage(target: Path) -> list[str]:
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.split()
    files = [name for name in files if not name.startswith(SKIP)]
    for name in files:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target / name)
    header = (ROOT / "deploy/huggingface/space-header.md").read_text(encoding="utf-8")
    readme = target / "README.md"
    readme.write_text(header + readme.read_text(encoding="utf-8"), encoding="utf-8")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        files = stage(Path(folder))
        if args.dry_run:
            print(f"{len(files)} files staged; README starts with:")
            print((Path(folder) / "README.md").read_text(encoding="utf-8")[:200])
            return 0
        space, token = os.environ.get("HF_SPACE", ""), os.environ.get("HF_TOKEN", "")
        if not space or not token:
            print("Set HF_SPACE (e.g. user/caplife-ai) and HF_TOKEN (a Hugging Face write token).", file=sys.stderr)
            return 1
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        remote = set(api.list_repo_files(space, repo_type="space"))
        stale = sorted(remote - set(files) - KEEP_ON_SPACE)
        commit = os.environ.get("GITHUB_SHA", "local")[:7]
        api.upload_folder(folder_path=folder, repo_id=space, repo_type="space",
                          commit_message=f"Mirror GitHub {commit}", delete_patterns=stale or None)
        print(f"Uploaded {len(files)} files to {space}; removed {len(stale)} stale file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
