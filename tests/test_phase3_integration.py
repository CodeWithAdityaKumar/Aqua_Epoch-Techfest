"""Optional real-release checks, distinct from synthetic contract unit tests."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.artifacts import ArtifactStore
from ml.quality import DATASET_FILENAME, load_quality_data
from ml.serving_contract import MODEL_SPECS, SOURCE_IDS

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts" / "phase3"


@pytest.fixture
def real_store():
    if not (RELEASE / "manifest.json").is_file():
        pytest.skip("Promote Phase 3 first to exercise real-model integration")
    return ArtifactStore(RELEASE)


def test_real_released_models_match_unmodified_phase2_candidates(real_store):
    frame, _ = load_quality_data(ROOT / DATASET_FILENAME)
    for model_id, (task, _, features) in MODEL_SPECS.items():
        source = ROOT / "reports" / "phase2_validated" / "candidates" / f"{SOURCE_IDS[model_id]}.joblib"
        if not source.exists():
            pytest.skip("Phase 2 candidates needed for exact preservation test")
        original = joblib.load(source)
        serving = real_store.load(model_id)
        assert serving.metadata["dataset_sha256"] == original["metadata"]["dataset_sha256"]
        assert serving.metadata["training_row_ids"] == original["metadata"]["training_row_ids"]
        x = frame.loc[serving.metadata["holdout_row_ids"][:10], features]
        if task == "classification":
            np.testing.assert_allclose(serving.pipeline.predict_proba(x), original["pipeline"].predict_proba(x), atol=1e-12)
            assert serving.metadata["threshold"] == original["metadata"]["threshold"]["value"]
        else:
            np.testing.assert_allclose(serving.pipeline.predict(x), original["pipeline"].predict(x), atol=1e-12)
            assert "pH" not in features and "Potability" not in features


def test_real_http_direct_frame_and_null_parity(real_store):
    client = TestClient(create_app(artifact_store=real_store))
    assert client.get("/health").json()["ready"]
    for mode in ("full", "quick"):
        classifier = real_store.load(f"classification_{mode}")
        values = {name: bounds["median"] for name, bounds in classifier.metadata["reference_ranges"].items()}
        result = client.post(f"/predict/{mode}", json=values)
        assert result.status_code == 200, result.text
        body = result.json()
        x = pd.DataFrame([values], columns=classifier.metadata["feature_order"])
        probability = float(classifier.pipeline.predict_proba(x)[0, 1])
        assert body["classification"]["probability_safe"] == pytest.approx(probability)
        assert body["classification"]["class_id"] == int(probability >= classifier.metadata["threshold"])
        if mode == "quick":
            assert body["classification"]["assessment"] == "insufficient_evidence"
        else:
            values["pH"] = None
            missing = client.post("/predict/full", json=values)
            assert missing.status_code == 200
            assert missing.json()["classification"]["assessment"] == "insufficient_evidence"
            assert missing.json()["regression"]["estimate"] == body["regression"]["estimate"]
    catalog = client.get("/models")
    assert catalog.status_code == 200
    assert "training_row_ids" not in catalog.text and str(ROOT) not in catalog.text
    client.close()


def test_phase3_notebook_executes_shared_backend_and_candidate_parity(real_store, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    monkeypatch.setenv("AQUA_REBUILD_RELEASE", "0")
    monkeypatch.setenv("AQUA_SKIP_INSTALL", "1")
    notebook = nbformat.read(ROOT / "notebooks" / "phase3_backend.ipynb", as_version=4)
    nbformat.validate(notebook)
    executed = nbclient.NotebookClient(
        notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    ).execute()
    cells = [c for c in executed.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in cells)
    assert not any(output.output_type == "error" for c in cells for output in c.outputs)


def test_http_verifier_rejects_invalid_or_overwriting_run_before_starting_server(tmp_path):
    from backend.verify_release import verify_release
    with pytest.raises(ValueError):
        verify_release(RELEASE, "absent.csv", tmp_path / "evidence.json", samples=0)
    output = tmp_path / "exists.json"
    output.write_text("preserve")
    with pytest.raises(FileExistsError):
        verify_release(RELEASE, "absent.csv", output)
    assert output.read_text() == "preserve"
