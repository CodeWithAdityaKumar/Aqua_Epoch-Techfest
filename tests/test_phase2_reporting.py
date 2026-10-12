"""End-to-end evidence/notebook checks; no training is hidden inside these tests."""
import json
from pathlib import Path

import pytest

from ml.report import compact_summary

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "phase2_validated"


def test_incomplete_evaluation_cannot_be_published_as_completed():
    with pytest.raises(ValueError, match="completed"):
        compact_summary({"status": "running"})


def test_saved_evidence_has_twelve_completed_experiments_and_preserved_protocol():
    if not (RESULTS / "results.json").exists():
        pytest.skip("Generate Phase 2 evidence first")
    report = json.loads((RESULTS / "results.json").read_text())
    summary = compact_summary(report)
    assert len(summary["experiments"]) == 12
    assert summary["audit"]["labelled_rows"] == 22611
    assert summary["cohort_counts"]["regression/physical_valid_targets"]["retained"] == 22366
    for item in report["experiments"]:
        config = item["selected_configuration"]
        assert config["approved"] is False
        assert set(config["training_row_ids"]).isdisjoint(config["holdout_row_ids"])
        assert "Potability" not in config["feature_order"]
        if item["task"] == "classification":
            assert item["threshold"]["value"] >= .5
            s = item["selected_internal_holdout"]
            assert s["conservative_threshold"]["safe_prediction_coverage"] <= s["baseline_threshold_0_5"]["safe_prediction_coverage"]
        else:
            assert "pH" not in config["feature_order"]
    # Negative concentrations are sanitized; positive-spike exclusion is EDA only.
    assert summary["audit"]["tds_turbidity_correlations"]["raw"] > .7
    assert abs(summary["audit"]["tds_turbidity_correlations"]["spike_excluded"]) < .1


def test_notebook_executes_local_audit_report_plots_and_real_candidate_inference(monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    if not (RESULTS / "results.json").exists():
        pytest.skip("Generate Phase 2 evidence first")
    # Training was verified separately by the real CLI experiment run. Reuse it
    # here to test the entire remaining notebook flow without 276 extra fits.
    monkeypatch.setenv("AQUA_RUN_EXPERIMENTS", "0")
    monkeypatch.setenv("AQUA_SKIP_INSTALL", "1")
    notebook = nbformat.read(ROOT / "notebooks" / "phase2_validation.ipynb", as_version=4)
    nbformat.validate(notebook)
    executed = nbclient.NotebookClient(
        notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    ).execute()
    code_cells = [c for c in executed.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    figures = list((RESULTS / "plots").glob("*.png"))
    assert len(figures) == 6
    assert all(p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in figures)
