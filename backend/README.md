# Aqua Epoch inference service — API v2

This is a **competition demo**, not BIS certification or a drinking-water
recommendation. Measurement units are undocumented and reported as `unknown`.

## Run

From the repository root:

```bash
python -m pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

For a promoted release, reproduce its recorded Python major/minor and exact
numpy, pandas, scikit-learn and joblib versions. The dependency ranges above
are installation bounds, not a promise of serialization compatibility. A
runtime mismatch fails closed before loading joblib; use the recorded runtime
or retrain and promote.

The release directory defaults to `artifacts/phase3`. Override it with
`AQUA_EPOCH_ARTIFACTS_DIR` or `create_app(artifacts_dir=...)` in Python. Startup
does not load models; missing releases return degraded health and inference
HTTP 503. Installing a valid release allows recovery without a server restart.

## HTTP routes and inputs

- `GET /health`: readiness of all four models and full/quick profiles. HTTP 200
  also when degraded; use `ready` for readiness checks.
- `GET /models`: public feature order, required/nullable flags, unknown units,
  engineering bounds, training reference ranges and decision thresholds.
  No filepaths, training IDs or arbitrary provenance are exposed.
- `POST /classify/full`: full 18-input dataset-label classification.
- `POST /classify/quick`: independent three-input classification.
- `POST /ph/full`: pH estimation from 17 fields, **excluding pH and Potability**.
- `POST /ph/quick`: independent three-input pH estimation.
- `POST /predict/full` and its alias `POST /predict`: combined classification
  and regression from all 18 fields.
- `POST /predict/quick`: combined three-input inference.
- `GET /history?limit=20`, `GET /history/{id}`, `DELETE /history/{id}` and
  `DELETE /history`: bounded local-demo history. Reads/deletes do not create a
  database; these unauthenticated routes are not suitable for multi-user use.

Independent routes load only their own artifact. Combined routes require both
models from the same manifest snapshot. pH regression never supplies measured
pH to the classifier.

Bodies are flat JSON with exact original-case keys. Full classification order:
`pH`, `Hardness`, `TDS`, `Chloramines`, `Sulfate`, `Conductivity`,
`Organic_carbon`, `Trihalomethanes`, `Turbidity`, `Calcium`, `Magnesium`,
`Sodium`, `Potassium`, `Chloride`, `Nitrate`, `Fluoride`, `Iron`, `Alkalinity`.
Full regression uses this order with `pH` removed. Quick order is `Turbidity`,
`Hardness`, `TDS`. All model calls use named pandas DataFrames in recorded order.

Every key is required. Explicit JSON `null` represents a genuinely missing
measurement and becomes `numpy.nan` for the fitted preprocessing pipeline;
omission is not equivalent to null. All-null requests, extras (including
`Potability`), strings, booleans, arrays, negative non-pH values, pH outside
`[0,14]`, nonfinite/nonstandard JSON numbers, and malformed JSON receive HTTP
422. Validation responses omit raw inputs and exception context. Request bodies
are not logged by this application.

Combined full requests need at least one non-pH measurement: supplying only
pH cannot be used to estimate pH back from entirely missing regression inputs.
Standalone classification may accept that partial input but abstains.

All prediction routes accept optional `?save_history=true`; it is off by
default. Successful validated results can be stored in local SQLite, retained
to 200 records, listed/viewed/deleted/cleared, and returned with a receipt.
Opt-out inference does not create the database. Failed validation/model requests
are not recorded. This is unauthenticated local-demo history, not a public or
multi-user account.

The existing broad `FEATURE_BOUNDS` in `backend/config.py` remain engineering
input checks, **not BIS limits**. `/models` publishes each range. Within those
bounds, unusual and out-of-training-range inputs produce evidence warnings.

Example quick request:

```bash
curl -X POST http://127.0.0.1:8000/classify/quick \
  -H 'Content-Type: application/json' \
  -d '{"Turbidity":2,"Hardness":100,"TDS":200}'
```

## Breaking changes from API v1

Every inference response includes `release_id`, `mode`, `contract_version`
(`aqua-epoch-serving-v1`), and `api_version` (`2.0.0`). Individual task responses
put their task fields at the top level; combined responses put them inside
`classification` and `regression`.

Classification fields are `model_id`, `class_id`, `model_label`,
`probability_safe`, `decision_threshold`, `assessment`, and `warnings`.
`probability_safe` always means **uncalibrated P(Potability=1)**, not confidence
in the returned label. `class_id` is 1 when that probability meets the stored
release threshold, otherwise 0; `model_label` is its dataset label `safe` or
`unsafe`. The estimator's default `predict()` threshold is not used. These raw
thresholded dataset predictions remain visible even when assessment abstains.

- Quick classification always returns `assessment: insufficient_evidence` and
  `limited_inputs` because of limited features and poor Phase2 precision.
- Full classification abstains on explicit missing values, inputs outside
  training min/max, or unavailable reference ranges.
- Values within min/max but outside q01/q99 produce `unusual_measurements`
  warnings without automatically abstaining.
- `abs(probability_safe - decision_threshold) < 0.05` causes `near_threshold`
  abstention. This is a documented heuristic, **not calibrated confidence**.
- Otherwise assessment is `predicted_safe` or `predicted_unsafe`.
- All classifications warn `uncalibrated_probability` and `not_certification`.

Regression fields are `model_id`, `estimate`, `reliability: limited`, and
`warnings`. Every estimate warns `weak_regression` and `not_lab_measurement`,
alongside any input-quality warnings. No confidence or prediction interval is
fabricated. Nonfinite estimates or estimates outside `[0,14]` fail with HTTP
503; estimates are never clipped. Invalid probability shapes, classes, ranges
or probability sums also fail closed rather than returning certainty.

Warnings are objects `{code, message, features: [...]}`. Model runtime warnings
are surfaced as a fixed `model_runtime_warning` message without exposing raw
runtime details. v1's ambiguous `probability` and scalar `pH_prediction` fields
are removed; v1's artifact HTTP 424 is replaced by structured HTTP 503:

```json
{"detail":{"error":"model_unavailable","reason":"missing_artifact","model_id":"classification_full","message":"The requested demo model is not ready for inference."}}
```

## Trusted release loading

Only a trusted administrator-promoted local release is supported. There is no
upload or user-provided model-path endpoint. Joblib can execute code: SHA-256
detects corruption, it does **not** make an untrusted release safe to load.

`manifest.json` must have the shared schema version, a nonempty `release_id`,
`intended_use: competition_demo_only`, and a `models` map. Model keys are
`classification_full`, `classification_quick`, `regression_full`,
`regression_quick`. Each entry contains `file: <model_id>.joblib`, `sha256`,
and metadata validated by `ml.serving_contract.validate_metadata` **before
deserialization**. Each serialized bundle contains `pipeline` and exactly the
same metadata. Embedded/manifest release IDs, exact feature order, `[0,1]`
classification classes, and callable task methods are enforced.

Paths cannot traverse or escape the release root through symlinks. SHA-256 is
checked over the same bytes passed to joblib. There is no Phase1, alternate
filename, bare-estimator, or pickle fallback. Successful models are cached
under a lock using manifest and artifact change tokens; failures are not
permanently cached. For a new immutable directory, restart with its path in
`AQUA_EPOCH_ARTIFACTS_DIR`; changing an environment variable after app creation
does not retarget the existing store.

Tests use synthetic local releases and a small actual sklearn sanitizer /
imputer pipeline; they do not load project candidate artifacts:

```bash
python -m pytest tests/test_backend.py tests/test_serving.py
```

`tests/test_phase3_integration.py` additionally loads the real local release and
compares it with original candidates. `python -m backend.verify_release --help`
describes the temporary loopback HTTP parity/latency check. No permanent server
or public tunnel is created by that verifier.
