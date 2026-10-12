"""Loopback HTTP release check and bounded, sequential latency measurement.

Starts a temporary local-only uvicorn server, verifies HTTP/direct-pipeline
parity, writes aggregate evidence, and shuts down. Not a load/production SLA test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import socket
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import uvicorn

from backend.app import create_app
from backend.config import FEATURE_BOUNDS
from ml.quality import FULL_FEATURES, QUICK_FEATURES, REGRESSION_FEATURES, load_quality_data
from ml.serving_contract import runtime_versions


def _request(base, route, payload=None, *, raw=None):
    body = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    req = Request(base + route, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=20) as response:
            code, data = response.status, json.loads(response.read())
    except HTTPError as exc:
        code, data = exc.code, json.loads(exc.read())
    return code, data, (time.perf_counter() - started) * 1000


def verify_release(release_dir, dataset, output, samples=30, warmup=5):
    if not 1 <= samples <= 200 or not 0 <= warmup <= 20:
        raise ValueError("Use 1–200 samples and 0–20 warmup requests")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite verification evidence: {output}")
    frame, audit = load_quality_data(dataset)
    app = create_app(artifacts_dir=str(release_dir))
    store = app.state.artifact_store
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    report = {"status": "running", "protocol": {
        "transport": "HTTP over loopback TCP to an actual uvicorn server in a local thread",
        "scope": "Single process; sequential requests; not concurrency, internet latency or a production SLA",
        "timing": "Client request through response JSON decoding; excludes request encoding and direct-pipeline parity computation",
        "samples_per_endpoint": samples, "warmup_per_endpoint": warmup,
        "inputs": "First physically admissible complete full-classifier holdout rows; selected without labels; reused across endpoints",
        "interpretation": "Integration/latency only, not new unbiased ML performance evidence",
    }, "runtime": {**runtime_versions(), "platform": platform.platform(), "uvicorn": uvicorn.__version__},
        "dataset_sha256": audit["source_sha256"], "routes": {}, "boundary_checks": {}}
    try:
        deadline = time.monotonic() + 30
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.02)
        if not server.started:
            raise RuntimeError("Local verification server did not start")
        code, health, cold_ms = _request(base, "/health")
        if code != 200 or not health["ready"]:
            raise RuntimeError(f"Release not ready: {health}")
        report.update(release_id=health["release_id"], cold_health_ms=cold_ms)
        models = {model_id: store.load(model_id) for model_id in health["models"]}
        if any(m.metadata["dataset_sha256"] != audit["source_sha256"] for m in models.values()):
            raise ValueError("Verification dataset differs from the promoted release")
        ids = models["classification_full"].metadata["holdout_row_ids"]
        inputs = frame.loc[ids, FULL_FEATURES]
        valid = inputs.notna().all(axis=1)
        for name in FULL_FEATURES:
            low, high = FEATURE_BOUNDS[name]
            valid &= inputs[name].between(low, high)
        rows = inputs.loc[valid].iloc[:max(samples, warmup, 1)]
        if len(rows) < samples:
            raise ValueError("Insufficient admissible holdout rows for the requested sample count")
        report["source_row_ids"] = rows.index[:samples].tolist()
        routes = {
            "/classify/full": ("full", "classification", FULL_FEATURES),
            "/classify/quick": ("quick", "classification", QUICK_FEATURES),
            "/ph/full": ("full", "regression", REGRESSION_FEATURES),
            "/ph/quick": ("quick", "regression", QUICK_FEATURES),
            "/predict/full": ("full", "combined", FULL_FEATURES),
            "/predict/quick": ("quick", "combined", QUICK_FEATURES),
        }
        for route, (mode, task, features) in routes.items():
            timings = []
            for iteration in range(warmup + samples):
                row = rows.iloc[iteration % len(rows)]
                payload = {name: float(row[name]) for name in features}
                code, response, elapsed = _request(base, route, payload)
                if code != 200:
                    raise AssertionError(f"{route} failed with HTTP {code}: {response}")
                if response["release_id"] != report["release_id"]:
                    raise AssertionError("Release changed during verification")
                if iteration < warmup:
                    continue
                timings.append(elapsed)
                for kind in (("classification", "regression") if task == "combined" else (task,)):
                    result = response[kind] if task == "combined" else response
                    model = models[f"{kind}_{mode}"]
                    names = model.metadata["feature_order"]
                    x = pd.DataFrame([[float(row[name]) for name in names]], columns=names)
                    if kind == "classification":
                        probability = float(model.pipeline.predict_proba(x)[0, 1])
                        assert np.isclose(probability, result["probability_safe"], atol=1e-12, rtol=0)
                        assert result["class_id"] == int(probability >= model.metadata["threshold"])
                        if mode == "quick":
                            assert result["assessment"] == "insufficient_evidence"
                    else:
                        assert np.isclose(float(model.pipeline.predict(x)[0]), result["estimate"], atol=1e-12, rtol=0)
                        assert result["reliability"] == "limited"
            report["routes"][route] = {
                "requests": samples, "http_200": samples, "direct_pipeline_parity": True,
                "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
                "min_ms": min(timings), "max_ms": max(timings),
            }
        payload = {name: float(rows.iloc[0][name]) for name in FULL_FEATURES}
        quick = {name: payload[name] for name in QUICK_FEATURES}
        checks = {
            "boolean": ("/predict/quick", {**quick, "Turbidity": True}, 422),
            "numeric_string": ("/predict/quick", {**quick, "TDS": "200"}, 422),
            "omitted_field": ("/predict/quick", {"TDS": 200}, 422),
            "negative": ("/predict/quick", {**quick, "TDS": -1}, 422),
            "null_all": ("/predict/quick", dict.fromkeys(QUICK_FEATURES), 422),
            "ph_target_as_input": ("/ph/full", payload, 422),
        }
        for name, (route, data, expected) in checks.items():
            status, _, _ = _request(base, route, data)
            assert status == expected, name
            report["boundary_checks"][name] = status
        for name, raw in (("nonfinite", b'{"Turbidity":NaN,"Hardness":100,"TDS":200}'),
                          ("malformed_json", b'{"TDS":')):
            status, _, _ = _request(base, "/predict/quick", raw=raw)
            assert status == 422, name
            report["boundary_checks"][name] = status
        status, missing, _ = _request(base, "/classify/full", {**payload, "pH": None})
        assert status == 200 and missing["assessment"] == "insufficient_evidence"
        report["boundary_checks"]["missing_ph_abstains"] = True
        status, outside, _ = _request(base, "/classify/full", {**payload, "Hardness": 4500})
        assert status == 200 and outside["assessment"] == "insufficient_evidence"
        report["boundary_checks"]["out_of_range_abstains"] = True
        report["status"] = "passed"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        listener.close()
        if thread.is_alive():
            raise RuntimeError("Verification server did not shut down cleanly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path("artifacts/phase3"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/phase3/http_verification.json"))
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    report = verify_release(args.release, args.dataset, args.output, args.samples)
    print(json.dumps({"status": report["status"], "routes": report["routes"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
