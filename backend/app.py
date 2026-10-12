"""Version 2 demo API with independent model tasks and safe error responses."""
from __future__ import annotations

import os
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ml.serving_contract import MODEL_SPECS, SCHEMA_VERSION
from .artifacts import ArtifactStore, ArtifactUnavailable
from .config import FEATURE_BOUNDS
from .inference import classify, regress
from .frontend import mount_frontend
from .history import HistoryStore, HistoryUnavailable, RETENTION_LIMIT, valid_id
from .schemas import (ClassificationResponse, CombinedFeatureRequest, FullFeatureRequest, PHFeatureRequest,
                       PredictionResponse, QuickFeatureRequest, RegressionResponse)


def create_app(artifacts_dir: str | None = None,
               artifact_store: ArtifactStore | None = None,
               history_db: str | None = None,
               mount_ui: bool = True) -> FastAPI:
    store = artifact_store or ArtifactStore(artifacts_dir or os.getenv("AQUA_EPOCH_ARTIFACTS_DIR"))
    history = HistoryStore(history_db or os.getenv("AQUA_EPOCH_HISTORY_DB"))
    app = FastAPI(title="Aqua Epoch Water Quality Inference API", version="2.0.0",
                  description="Competition demo only. Engineering input bounds are not BIS limits. Units are undocumented.")
    app.state.artifact_store = store
    app.state.history_store = history

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        # Pydantic ctx can contain ValueError, and input can contain NaN or raw
        # request data. Neither belongs in JSON responses or logs.
        details = [{"loc": list(item["loc"]), "type": item["type"], "msg": item["msg"]}
                   for item in exc.errors()]
        return JSONResponse(status_code=422, content={"error": "invalid_request",
                            "message": "Supply the exact required numeric fields or explicit nulls, within the documented engineering bounds.",
                            "details": details})

    @app.exception_handler(ArtifactUnavailable)
    async def artifact_error_handler(request: Request, exc: ArtifactUnavailable):
        return JSONResponse(status_code=503, content={"detail": exc.public()})

    @app.exception_handler(HistoryUnavailable)
    async def history_error_handler(request: Request, exc: HistoryUnavailable):
        return JSONResponse(status_code=503, content={"error": "history_unavailable", "message": "Local history is unavailable; the inference result may still be usable."})

    @app.exception_handler(StarletteHTTPException)
    async def body_error_handler(request: Request, exc: StarletteHTTPException):
        # FastAPI wraps undecodable body bytes in HTTP 400 instead of a
        # RequestValidationError. Keep malformed JSON consistently at 422.
        if exc.status_code == 400 and exc.detail == "There was an error parsing the body":
            return JSONResponse(status_code=422, content={"error": "invalid_request",
                                "message": "The request body must be valid JSON.",
                                "details": [{"loc": ["body"], "type": "json_invalid",
                                             "msg": "Invalid JSON body"}]})
        return await http_exception_handler(request, exc)

    @app.get("/health")
    def health():
        return store.readiness()

    @app.get("/models")
    def models():
        public = {}
        for model_id, (task, mode, features) in MODEL_SPECS.items():
            entry = {"model_id": model_id, "task": task, "mode": mode,
                     "features": [{"name": name, "required": True, "nullable": True,
                                   "units": "unknown", "engineering_bounds": list(FEATURE_BOUNDS[name])}
                                  for name in features], "feature_order": features}
            try:
                loaded = store.load(model_id)
            except ArtifactUnavailable as exc:
                entry.update(available=False, reason=exc.reason)
            else:
                metadata = loaded.metadata
                # Explicit allowlist: never expose provenance/training IDs,
                # local paths, arbitrary evaluation text, or bundle contents.
                entry.update(available=True, reason=None, release_id=metadata["release_id"],
                             status="demo_only", calibrated=False,
                             reference_ranges={name: {key: bounds[key]
                                                     for key in ("min", "q01", "median", "q99", "max")}
                                               for name, bounds in metadata["reference_ranges"].items()},
                             decision_threshold=metadata.get("threshold"),
                             probability_semantics="P(Potability=1)" if task == "classification" else None)
            public[model_id] = entry
        return {"api_version": "2.0.0", "contract_version": SCHEMA_VERSION,
                "intended_use": "competition_demo_only", "bounds_policy": "engineering_checks_not_BIS",
                "units": "unknown", "models": public}

    def envelope(model, mode):
        return {"release_id": model.metadata["release_id"], "mode": mode,
                "contract_version": SCHEMA_VERSION, "api_version": "2.0.0"}

    def history_receipt(save_history, task, mode, payload, result):
        if not save_history:
            return None
        try:
            return history.save(task, mode, payload.model_dump(), result)
        except HistoryUnavailable:
            return {"saved": False, "id": None, "error": "history_unavailable"}

    def combined(mode, payload, save_history=False):
        classifier, regressor = store.models(mode)
        values = payload.model_dump()
        result = {**envelope(classifier, mode), "classification": classify(classifier, values),
                  "regression": regress(regressor, values)}
        receipt = history_receipt(save_history, "combined", mode, payload, result)
        if receipt is not None: result["history"] = receipt
        return result

    def classification(mode, payload, save_history=False):
        model = store.load(f"classification_{mode}")
        result = {**envelope(model, mode), **classify(model, payload.model_dump())}
        receipt = history_receipt(save_history, "classification", mode, payload, result)
        if receipt is not None: result["history"] = receipt
        return result

    def regression(mode, payload, save_history=False):
        model = store.load(f"regression_{mode}")
        result = {**envelope(model, mode), **regress(model, payload.model_dump())}
        receipt = history_receipt(save_history, "regression", mode, payload, result)
        if receipt is not None: result["history"] = receipt
        return result

    @app.post("/predict", response_model=PredictionResponse, response_model_exclude_none=True)
    @app.post("/predict/full", response_model=PredictionResponse, response_model_exclude_none=True)
    def predict_full(payload: CombinedFeatureRequest, save_history: bool = Query(False)):
        return combined("full", payload, save_history)

    @app.post("/predict/quick", response_model=PredictionResponse, response_model_exclude_none=True)
    def predict_quick(payload: QuickFeatureRequest, save_history: bool = Query(False)):
        return combined("quick", payload, save_history)

    @app.post("/classify/full", response_model=ClassificationResponse, response_model_exclude_none=True)
    def classify_full(payload: FullFeatureRequest, save_history: bool = Query(False)):
        return classification("full", payload, save_history)

    @app.post("/classify/quick", response_model=ClassificationResponse, response_model_exclude_none=True)
    def classify_quick(payload: QuickFeatureRequest, save_history: bool = Query(False)):
        return classification("quick", payload, save_history)

    @app.post("/ph/full", response_model=RegressionResponse, response_model_exclude_none=True)
    def ph_full(payload: PHFeatureRequest, save_history: bool = Query(False)):
        return regression("full", payload, save_history)

    @app.post("/ph/quick", response_model=RegressionResponse, response_model_exclude_none=True)
    def ph_quick(payload: QuickFeatureRequest, save_history: bool = Query(False)):
        return regression("quick", payload, save_history)

    @app.get("/history")
    def list_history(limit: Annotated[int, Query(ge=1, le=RETENTION_LIMIT)] = 20):
        return history.recent(limit)

    @app.get("/history/{record_id}")
    def get_history(record_id: str):
        if not valid_id(record_id):
            raise HTTPException(status_code=404, detail="History record not found")
        record = history.get(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="History record not found")
        return record

    @app.delete("/history/{record_id}")
    def delete_history(record_id: str):
        if not valid_id(record_id):
            raise HTTPException(status_code=404, detail="History record not found")
        return {"deleted": bool(history.delete(record_id))}

    @app.delete("/history")
    def clear_history():
        return {"deleted_count": history.delete()}

    if mount_ui:
        mount_frontend(app)

    return app


app = create_app()
