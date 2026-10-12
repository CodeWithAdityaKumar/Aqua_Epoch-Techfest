"""Fail-closed loading of administrator-promoted, local Phase3 releases.

Joblib is executable serialization. A hash is an integrity check, not a trust
boundary: this service must never accept uploaded releases or model paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import threading
from typing import Any
import warnings

import joblib
import numpy as np

from ml.serving_contract import MODEL_SPECS, SCHEMA_VERSION, check_hash, validate_metadata
from .config import DEFAULT_ARTIFACTS_DIR


class ArtifactUnavailable(RuntimeError):
    """Only fixed, public reason codes cross the HTTP boundary."""

    def __init__(self, reason: str, model_id: str | None = None):
        self.reason = reason
        self.model_id = model_id
        super().__init__(reason)

    def public(self) -> dict[str, Any]:
        return {"error": "model_unavailable", "reason": self.reason,
                "model_id": self.model_id,
                "message": "The requested demo model is not ready for inference."}


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Any
    metadata: dict[str, Any]
    manifest_token: tuple
    runtime_warning: bool = False


def _token(path: Path) -> tuple:
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite manifest number")


class ArtifactStore:
    """Cache successful models only, with file and manifest change tokens."""

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory or DEFAULT_ARTIFACTS_DIR).expanduser().resolve()
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[tuple, LoadedModel]] = {}

    def _path(self, name: str) -> Path:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactUnavailable("invalid_artifact_path")
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory):
            raise ArtifactUnavailable("invalid_artifact_path")
        return path

    def _manifest(self) -> tuple[dict, tuple]:
        try:
            path = self._path("manifest.json")
            token = _token(path)
            content = path.read_bytes()
            if _token(path) != token:
                raise ArtifactUnavailable("release_changed")
            manifest = json.loads(content, parse_constant=_reject_constant,
                                  object_pairs_hook=_json_object)
        except ArtifactUnavailable:
            raise
        except FileNotFoundError:
            raise ArtifactUnavailable("missing_manifest") from None
        except Exception:
            raise ArtifactUnavailable("invalid_manifest") from None
        if (not isinstance(manifest, dict)
                or manifest.get("schema_version") != SCHEMA_VERSION
                or manifest.get("intended_use") != "competition_demo_only"
                or not isinstance(manifest.get("release_id"), str)
                or not manifest["release_id"].strip()
                or not isinstance(manifest.get("models"), dict)
                or not set(manifest["models"]).issubset(MODEL_SPECS)):
            raise ArtifactUnavailable("incompatible_manifest")
        return manifest, token

    def load(self, model_id: str) -> LoadedModel:
        with self._lock:
            try:
                return self._load(model_id)
            except ArtifactUnavailable as exc:
                self._cache.pop(model_id, None)
                if exc.model_id is None:
                    exc.model_id = model_id
                raise
            except Exception:
                self._cache.pop(model_id, None)
                raise ArtifactUnavailable("invalid_artifact", model_id) from None

    def _load(self, model_id: str) -> LoadedModel:
        if model_id not in MODEL_SPECS:
            raise ArtifactUnavailable("unsupported_model", model_id)
        manifest, manifest_token = self._manifest()
        entry = manifest["models"].get(model_id)
        if entry is None:
            raise ArtifactUnavailable("missing_model")
        if not isinstance(entry, dict):
            raise ArtifactUnavailable("invalid_model_metadata")
        # Metadata, runtime, and filenames are checked BEFORE deserialization.
        metadata = entry.get("metadata")
        try:
            validate_metadata(metadata, model_id)
            check_hash(entry.get("sha256"))
            if metadata["release_id"] != manifest["release_id"]:
                raise ValueError("Release mismatch")
            if any(not {"min", "q01", "median", "q99", "max"}.issubset(bounds)
                   for bounds in metadata["reference_ranges"].values()):
                raise ValueError("Incomplete reference ranges")
        except Exception:
            raise ArtifactUnavailable("incompatible_model_metadata") from None
        if entry.get("file") != f"{model_id}.joblib":
            raise ArtifactUnavailable("invalid_artifact_path")
        path = self._path(entry["file"])
        try:
            file_token = _token(path)
        except FileNotFoundError:
            raise ArtifactUnavailable("missing_artifact") from None
        cache_token = (manifest_token, file_token)
        cached = self._cache.get(model_id)
        if cached is not None and cached[0] == cache_token:
            return cached[1]
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            raise ArtifactUnavailable("missing_artifact") from None
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ArtifactUnavailable("artifact_hash_mismatch")
        if _token(path) != file_token or _token(self._path("manifest.json")) != manifest_token:
            raise ArtifactUnavailable("release_changed")
        try:
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always")
                # Deserialize the exact bytes whose hash was checked, not a
                # reopened path that could have changed between operations.
                bundle = joblib.load(io.BytesIO(content))
            if not isinstance(bundle, dict) or bundle.get("metadata") != metadata:
                raise ValueError("Embedded metadata mismatch")
            pipeline = bundle["pipeline"]
            features = np.asarray(getattr(pipeline, "feature_names_in_", None))
            if features.ndim != 1 or features.tolist() != MODEL_SPECS[model_id][2]:
                raise ValueError("Feature schema mismatch")
            if not callable(getattr(pipeline, "predict", None)):
                raise ValueError("Missing predict")
            if MODEL_SPECS[model_id][0] == "classification":
                classes = np.asarray(getattr(pipeline, "classes_", None))
                if (classes.shape != (2,) or classes.dtype.kind not in "iuf"
                        or not np.array_equal(classes, [0, 1])
                        or not callable(getattr(pipeline, "predict_proba", None))):
                    raise ValueError("Classification schema mismatch")
        except Exception:
            raise ArtifactUnavailable("incompatible_artifact") from None
        loaded = LoadedModel(pipeline, metadata, manifest_token, bool(emitted))
        self._cache[model_id] = (cache_token, loaded)
        return loaded

    def models(self, mode: str) -> tuple[LoadedModel, LoadedModel]:
        with self._lock:
            classifier = self.load(f"classification_{mode}")
            regressor = self.load(f"regression_{mode}")
            if classifier.manifest_token != regressor.manifest_token:
                raise ArtifactUnavailable("release_changed")
            return classifier, regressor

    def readiness(self) -> dict[str, Any]:
        with self._lock:
            models = {}
            releases = set()
            for model_id in MODEL_SPECS:
                try:
                    model = self.load(model_id)
                except ArtifactUnavailable as exc:
                    models[model_id] = {"available": False, "reason": exc.reason}
                else:
                    releases.add(model.metadata["release_id"])
                    models[model_id] = {"available": True, "reason": None}
            profiles = {}
            for mode in ("full", "quick"):
                classification = models[f"classification_{mode}"]["available"]
                regression = models[f"regression_{mode}"]["available"]
                profiles[mode] = {"available": classification and regression,
                                  "classification": classification, "regression": regression}
            ready = all(item["available"] for item in models.values()) and len(releases) == 1
            return {"status": "ok" if ready else "degraded", "ready": ready,
                    "release_id": next(iter(releases)) if len(releases) == 1 else None,
                    "contract_version": SCHEMA_VERSION, "api_version": "2.0.0",
                    "profiles": profiles, "models": models}
