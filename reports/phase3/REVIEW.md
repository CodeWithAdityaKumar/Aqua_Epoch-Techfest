# Phase 3 — verified local serving release

## Outcome

API v2 now serves a demo-only immutable release from `artifacts/phase3`.
Release ID: `d92c7561-4a48-44d5-8c51-339c743a452f`.
Phase 1 models and Phase 2 experimental evidence are preserved. This is not a
new ML score, public deployment, calibrated risk model, or water certification.

## Model and operating-policy choice

- Full and quick classifiers: physical-sanitized Phase 2 CV-selected Random
  Forest pipelines, unchanged. Thresholds: full `0.5`, quick approximately
  `0.7003935201872995`; taken from Phase 2 training-only OOF evidence.
- Full and quick pH estimators: physical-valid-target, physical-sanitized,
  CV-selected Ridge pipelines. Raw-target-cohort metrics are retained in release
  metadata so filtered-target performance is not presented as a pure model gain.
- Policy is an explicit operational choice, not a new ranking by holdout scores.
  No estimator is refitted; the original heldout rows remain outside fitting.
- Training-only min/q01/median/q99/max values support input-quality warnings,
  not regulatory decisions. Measurement units remain undocumented.

## Serving semantics

- `probability_safe` is uncalibrated P(Potability=1). `class_id` and `model_label`
  use the stored threshold rather than the estimator's default `predict()`.
- Quick mode always reports `insufficient_evidence`. The raw dataset prediction
  remains visible, but must never override this assessment in the frontend.
- Full mode abstains for missing inputs, values outside its training min/max,
  unavailable reference ranges, or probability within 0.05 of the threshold.
  This last rule is an explicit heuristic, not a calibrated confidence interval.
- Within-min/max percentile outliers receive warnings, not automatic rejection.
- Regression returns an estimate with `reliability: limited`, weak-regression
  warning and no invented confidence interval. Out-of-[0,14]/nonfinite model
  outputs fail with HTTP 503; they are not silently clipped.
- Standalone pH inference excludes measured pH and Potability. A combined
  request containing only measured pH is rejected rather than estimating pH
  from entirely missing non-pH inputs.

## Release integrity

`ml.promote` validates source evidence and runtime before loading trusted
candidate files. It stages and atomically publishes four bundles, a manifest,
`promotion.json`, and exact serialization dependency pins. It refuses existing
nonempty outputs. `ml.serving_contract` is shared with the loader.

The backend checks SHA-256 over the exact bytes deserialized, exact feature order,
binary class schema, metadata agreement, source contract and runtime versions.
It rejects Phase 1/2 direct-file fallback and path escape. Successful loads are
cached under change tokens/locking; missing-release failures can recover.
Hashes prove consistency, not authenticity: untrusted joblib files must never
be uploaded or loaded.

## Verification

- Full test suite: **157 passed**, one external Starlette/TestClient deprecation
  warning. No old positional-feature-name inference warnings remain.
- Tests cover synthetic schema/hash/version/path failures, immutable promotion,
  training-only reference ranges, missing/error recovery, numeric boundaries,
  invalid model outputs, direct candidate preservation and named-frame parity.
- Both Phase 2 and Phase 3 notebooks execute locally. Phase 3 reuses the actual
  promoted release in the automated notebook test. Actual evaluation/promotion
  were separately run through the CLI. Google-hosted Colab remains unverified.
- Real local uvicorn HTTP verification: six endpoints, 30 measured requests each,
  **180/180 HTTP 200** with direct-pipeline numeric parity. Five warmups per
  endpoint are excluded. Ten additional boundary/abstention checks passed.
- `http_verification.json` is the initial run (tests ran concurrently).
  `http_verification_quiet.json` is a repeat run without concurrent agent tests;
  use the latter for the following descriptive latency figures. No claim of a
  software speedup between runs is made.

Measured loopback latency (median / p95, milliseconds):

- Full classification: **37.8 / 64.0**.
- Quick classification: **31.1 / 48.0**.
- Full pH: **25.1 / 34.8**.
- Quick pH: **20.5 / 43.2**.
- Full combined: **59.1 / 87.2**.
- Quick combined: **50.1 / 73.7**.

Timings include HTTP and response decoding, exclude request encoding and
separate direct-pipeline comparisons. Sequential single-process local test,
not concurrent load, internet latency or a production SLA. All verification
servers were shut down after checks; no persistent server was left running.

## Next phase

Phase 4 is the functional frontend, not another model-score claim. Use API v2
`/models`, assessment and warnings; show limited quick-mode evidence and pH
reliability explicitly. UI polish, public deployment/Colab rehearsal, and the
technical PDF are later phases. Four phases remain after this one.
