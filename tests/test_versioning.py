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



def test_app_reloads_caplife_modules_left_over_from_older_code():
    # Streamlit Cloud reruns app.py after an update without restarting the server, so
    # modules imported by the previous code must not keep serving old behaviour.
    import sys
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import caplife.six_rul_page  # noqa: F401  (make sure it is loaded)

    app = str(Path(__file__).resolve().parents[1] / "app.py")
    saved = {name: module for name, module in sys.modules.items() if name.split(".")[0] == "caplife"}
    try:
        current = sys.modules["caplife.six_rul_page"]
        AppTest.from_file(app, default_timeout=120).run()
        assert sys.modules["caplife.six_rul_page"] is current  # unchanged code is not reloaded

        sys.modules["caplife"].LOADED_FINGERPRINT = "from-an-older-deploy"
        AppTest.from_file(app, default_timeout=120).run()
        assert sys.modules["caplife"] is not saved["caplife"]  # the package was imported afresh
        assert sys.modules.get("caplife.six_rul_page") is not current  # pages load the new code
        assert sys.modules["caplife"].LOADED_FINGERPRINT == sys.modules["caplife"].source_fingerprint()
    finally:
        for name in [name for name in sys.modules if name.split(".")[0] == "caplife"]:
            del sys.modules[name]
        sys.modules.update(saved)
        saved["caplife"].LOADED_FINGERPRINT = saved["caplife"].source_fingerprint()
