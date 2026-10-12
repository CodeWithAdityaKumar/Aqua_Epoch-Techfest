"""Focused tests for the Phase 6 real-subprocess checker."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/tmp/omnirush/aqua-epoch-full-venv/bin/python")


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("phase6_check", ROOT / "scripts/phase6_check.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_direct_script_import_can_load_repo_packages(checker):
    assert checker.ROOT == ROOT
    assert checker._http.__name__ == "_http"


def test_json_get_is_decoded_without_assuming_request_body(checker, monkeypatch):
    class Response:
        code = 200
        headers = type("Headers", (), {"get_content_type": lambda self: "application/json"})()

        def read(self):
            return b'{"ok": true}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Opener:
        def open(self, request, timeout):
            assert request.data is None
            return Response()

    monkeypatch.setattr(checker, "build_opener", lambda _: Opener())
    assert checker._http("http://127.0.0.1", "/health") == (200, {"ok": True})


@pytest.mark.parametrize("args", [("--samples", "0"), ("--concurrency", "17")])
def test_invalid_arguments_fail_before_server(args, tmp_path):
    output = tmp_path / "evidence.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/phase6_check.py"), *args, "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert not output.exists()


def test_existing_output_is_never_overwritten(checker, tmp_path):
    output = tmp_path / "evidence.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        checker.main(["--samples", "1", "--concurrency", "1", "--output", str(output)])
    assert error.value.code == 2
    assert output.read_text(encoding="utf-8") == "keep"


@pytest.mark.skipif(not PYTHON.is_file(), reason="project release virtualenv is unavailable")
def test_small_real_release_smoke_is_artifact_only(tmp_path):
    output = tmp_path / "phase6-evidence.json"
    result = subprocess.run(
        [str(PYTHON), str(ROOT / "scripts/phase6_check.py"), "--samples", "2",
         "--concurrency", "2", "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["scope"]["dataset_required"] is False
    assert report["scope"]["network_isolation_verified"] is False
    assert report["concurrency"]["requests"] == 2
    assert report["concurrency"]["latency_ms"]["p95"] >= 0
    assert report["history"]["actual_local_db_unchanged"] is True
