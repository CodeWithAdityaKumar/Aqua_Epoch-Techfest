from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts" / "phase3"


def _quick(client, save=False):
    return client.post("/classify/quick" + ("?save_history=true" if save else ""), json={"Turbidity": 3, "Hardness": 150, "TDS": 1200})


def test_history_is_opt_in_and_round_trips_result(tmp_path):
    db = tmp_path / "nested" / "history.sqlite3"
    client = TestClient(create_app(artifacts_dir=RELEASE, history_db=db))
    assert not db.exists()
    response = _quick(client)
    assert response.status_code == 200
    assert "history" not in response.json()
    assert not db.exists()

    response = _quick(client, save=True)
    assert response.status_code == 200
    receipt = response.json()["history"]
    assert receipt["saved"] is True and receipt["id"]
    assert db.exists()
    listing = client.get("/history?limit=1")
    assert listing.status_code == 200
    assert listing.json()["items"][0]["id"] == receipt["id"]
    record = client.get(f"/history/{receipt['id']}")
    assert record.status_code == 200
    assert record.json()["inputs"] == {"Turbidity": 3.0, "Hardness": 150.0, "TDS": 1200.0}
    assert "history" not in record.json()["result"]


def test_history_deletion_and_retention_are_bounded(tmp_path):
    client = TestClient(create_app(artifacts_dir=RELEASE, history_db=tmp_path / "history.sqlite3"))
    ids = []
    for index in range(205):
        response = _quick(client, save=True)
        assert response.status_code == 200
        ids.append(response.json()["history"]["id"])
    records = client.get("/history?limit=200").json()
    assert len(records["items"]) == 200
    assert ids[0] not in {item["id"] for item in records["items"]}
    assert client.delete(f"/history/{ids[-1]}").json() == {"deleted": True}
    assert client.get(f"/history/{ids[-1]}").status_code == 404
    cleared = client.delete("/history")
    assert cleared.status_code == 200 and cleared.json()["deleted_count"] == 199
    assert client.get("/history").json()["items"] == []


def test_history_routes_do_not_save_rejected_requests(tmp_path):
    db = tmp_path / "history.sqlite3"
    client = TestClient(create_app(artifacts_dir=RELEASE, history_db=db))
    response = client.post("/classify/quick?save_history=true", json={"Turbidity": True, "Hardness": 150, "TDS": 1200})
    assert response.status_code == 422
    assert not db.exists()
    assert client.get("/history/not-a-uuid").status_code == 404
    assert client.delete("/history/not-a-uuid").status_code == 404
