"""Launcher checks that never leave a server process running."""
from pathlib import Path

import pytest

from backend import demo

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts" / "phase3"


def test_real_release_check_only():
    if not (RELEASE / "manifest.json").is_file():
        pytest.skip("Promote Phase 3 first to exercise the real release")
    assert demo.main(["--check-only", "--release", str(RELEASE)]) == 0


def test_check_only_rejects_unavailable_artifacts(tmp_path, capsys):
    assert demo.main(["--check-only", "--release", str(tmp_path / "missing")]) == 2
    output = capsys.readouterr().err
    assert "model preflight failed" in output.lower()
    assert str(tmp_path) not in output


def test_invalid_host_fails_closed(capsys):
    assert demo.main(["--check-only", "--host", "0.0.0.0"]) == 2
    assert "loopback" in capsys.readouterr().err.lower()


def test_run_is_configured_without_starting_a_persistent_server(monkeypatch):
    calls = []

    class FakeUvicorn:
        @staticmethod
        def run(app, **kwargs):
            calls.append((app, kwargs))

    monkeypatch.setitem(__import__("sys").modules, "uvicorn", FakeUvicorn)
    # The app is constructed only after preflight; fake run returns immediately.
    assert demo.main(["--release", str(RELEASE), "--host", "localhost", "--port", "8123"]) == 0
    assert len(calls) == 1
    assert calls[0][1] == {"host": "127.0.0.1", "port": 8123, "access_log": False}
