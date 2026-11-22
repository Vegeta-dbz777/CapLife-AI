import os

from caplife.versioning import code_version


def test_version_changes_when_code_models_or_data_change(tmp_path):
    for folder in ("caplife", "artifacts", "sample_data"):
        (tmp_path / folder).mkdir()
    model = tmp_path / "artifacts" / "model.pt"
    model.write_bytes(b"weights")
    (tmp_path / "caplife" / "sim.py").write_text("x = 1\n")
    first = code_version(tmp_path)
    assert code_version(tmp_path) == first  # stable while nothing changes

    model.write_bytes(b"new weights")
    os.utime(model, ns=(model.stat().st_atime_ns, model.stat().st_mtime_ns + 1_000_000))
    assert code_version(tmp_path) != first

    cache = tmp_path / "caplife" / "__pycache__"
    cache.mkdir()
    before = code_version(tmp_path)
    (cache / "sim.cpython-311.pyc").write_bytes(b"compiled")
    assert code_version(tmp_path) == before  # bytecode caches are ignored
