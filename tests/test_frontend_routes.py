from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_frontend_is_same_origin_and_private_files_are_not_exposed(tmp_path):
    client = TestClient(create_app(artifacts_dir=ROOT / "artifacts" / "phase3", history_db=tmp_path / "history.sqlite3"))
    page = client.get("/")
    assert page.status_code == 200
    assert "Aqua Epoch" in page.text
    assert client.get("/static/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/../README.md").status_code in {404, 400}
    assert client.get("/models").status_code == 200
    assert client.get("/docs").status_code == 200
