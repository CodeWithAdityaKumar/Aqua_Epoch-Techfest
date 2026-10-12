# Phased delivery and handoff

## Agreement

- User wants a strong, technically defensible competition entry, not fabricated
  performance or a first-prize guarantee.
- Work one phase at a time; start the next only when user says `next`.
- Round 2 ML/backend first. Frontend polish and technical PDF are deferred.
- Hybrid inputs approved: full chemical profile plus a separate three-field
  quick mode. Never silently pretend three measurements equal a full profile.
- Colab/Jupyter must share the same testable Python implementation.

## Phase 1 — baseline (existing)

Four baseline Random Forest pipelines in `artifacts/`, initial notebook and
FastAPI combined prediction routes. Results are baseline internal holdout
metrics. Invalid finite measurements were retained in baseline training.

## Phase 2 — completed validation work

Implementation:
- `ml/quality.py`: strict CSV contract, stable source IDs, audited coercion,
  anomaly/sentinel/duplicate analysis, stateless physical sanitizer.
- `ml/evaluate.py`: full/quick, raw/sanitized features, raw/valid-pH target
  cohorts, four candidates each, training-only CV selection, fold-local
  preprocessing, unsafe-to-safe metrics, threshold fitting, source-order and
  repeated split sensitivity, permutation importance, experimental bundles.
- `ml/report.py`: reproducible compact review and six plots.
- `notebooks/phase2_validation.ipynb`: Drive/local setup, audit, evaluation,
  figures and candidate inference with the actual feature/threshold contract.

Evidence to use: `reports/phase2_validated/`. The first `reports/phase2/` run
is preserved but superseded for threshold policy: a safety-oriented threshold
must never go below 0.5. No holdout-optimised model/policy winner was chosen.
Neither experiment directory changes backend serving models.

Key results (physical-sanitized features):
- Full classification: Random Forest selected by CV; CV F1 0.99004 +/- 0.00216;
  primary internal-holdout F1 0.98885, AUC 0.99969, unsafe-to-safe FPR 0.47%.
- Quick classification: default-threshold FPR 35.37%. OOF-selected threshold
  about 0.70039 reduces holdout FPR to 1.59%, but safe recall is only 3.99%,
  safe precision 61.40%, and safe-call coverage 2.52%. Not reliable certification.
- Full regression, raw pH targets: Ridge RMSE 1.14471 vs dummy 1.15361.
- Full regression, [0,14] target cohort: Ridge RMSE 0.67838 vs dummy 0.68649;
  R2 0.02347. Different cohorts; do not claim that change as pure model gain.
- Raw TDS/Turbidity correlation 0.74257 becomes 0.02772 when exploratory spikes
  are excluded. Strong original correlation was largely spike-driven.

Verified commands (from project root):

```bash
/tmp/omnirush/aqua-epoch-full-venv/bin/python -m ml.evaluate --dataset "Aqua_Epoch_BIS_Water_Quality_Dataset.xls - Aqua_Epoch_BIS_Water_Quality_Dataset.csv" --output reports/phase2_validated --folds 3 --trees 80 --permutation-repeats 3
/tmp/omnirush/aqua-epoch-full-venv/bin/python -m ml.report --results-dir reports/phase2_validated
/tmp/omnirush/aqua-epoch-full-venv/bin/python -m pytest -q
```

Results: all 12 experiments completed, report/six PNG plots generated, **44
tests passed** (four existing backend/dependency warnings). Notebook was
executed in a local Jupyter kernel against completed results with training and
installation disabled; real training was run separately through the CLI.
Google-hosted Colab execution has not been performed. No linter/typechecker is
configured; Python compile and notebook schema checks passed.

Temporary test environment is not a deliverable; users create their own virtual
environment using root requirements. Full row-level evidence/candidate binaries
are generated and excluded from git to avoid oversized accidental commits.

## Phase 3 — completed local demo release and backend integration

- Shared schema: `ml/serving_contract.py`, `aqua-epoch-serving-v1`.
- Immutable promotion: `ml/promote.py`; generated `artifacts/phase3/` contains
  four demo-only bundles, manifest, exact runtime pins and promotion evidence.
- Operational choice: physical-sanitized classification; physical-valid-target
  regression. Each retains its Phase 2 training-CV winner and fitted pipeline;
  classifier thresholds are preserved. No model refit or holdout-based policy
  ranking. Original artifacts/evidence remain unchanged.
- Reference ranges are derived from sanitized **training rows only**.
- API v2 implements independent classification/pH routes and combined routes.
  Strict names, explicit nullable inputs, SHA/version/feature checks, safe error
  responses, artifact recovery and named-DataFrame inference are verified.
- `probability_safe` means P(class 1). No fabricated confidence fallback,
  measured-pH substitution, output clipping, or confidence interval.
- Quick assessments always abstain. Full assessments abstain for missing/
  outside-training-range/unavailable-range inputs and a documented uncalibrated
  near-threshold heuristic. Raw thresholded labels remain available with warnings.
- pH estimates always have limited reliability; poor Phase 2 regression evidence
  is not hidden. Abstention itself has not been externally calibrated/validated.
- Notebook: `notebooks/phase3_backend.ipynb`, locally executed against the actual
  release. Colab defaults to rebuilding in its runtime for pickle compatibility.

Verified from project root:

```bash
/tmp/omnirush/aqua-epoch-full-venv/bin/python -m ml.promote --source reports/phase2_validated --dataset "Aqua_Epoch_BIS_Water_Quality_Dataset.xls - Aqua_Epoch_BIS_Water_Quality_Dataset.csv" --output artifacts/phase3
/tmp/omnirush/aqua-epoch-full-venv/bin/python -m pytest -q
/tmp/omnirush/aqua-epoch-full-venv/bin/python -m backend.verify_release --release artifacts/phase3 --dataset "Aqua_Epoch_BIS_Water_Quality_Dataset.xls - Aqua_Epoch_BIS_Water_Quality_Dataset.csv" --output reports/phase3/http_verification_quiet.json --samples 30
```

Results: **157 tests passed**, one Starlette dependency deprecation warning.
Real loopback HTTP: 30 measured requests for each of six routes (180 total),
all HTTP 200 and direct-pipeline parity. Five warmups per route excluded.
Full combined median/p95: about 59.1/87.2 ms; quick combined: 50.1/73.7 ms.
Additional invalid/missing/OOD HTTP checks passed. This is sequential local
latency, not production/concurrency/internet performance. Temporary servers shut
down automatically. See `reports/phase3/REVIEW.md` for interpretation.

Still unresolved: source provenance, near-duplicate/source-group contamination,
official units and safe-limit mapping, independent external validation. No
claim that current potability labels establish microbial safety is permissible.

## Phase 4 — completed functional frontend and opt-in local history

- Same-origin vanilla frontend under `frontend/`; FastAPI serves `/` and allowlisted
  `/static/*`, so local/Colab operation needs no Node server, CDN or CORS setup.
- Schema-driven task/mode form reads `/models`; it supports combined,
  classification and pH tasks, full/quick modes, explicit `Not measured` nulls,
  numeric/range validation, refresh/reset, synthetic training-median examples
  clearly labelled as non-real, loading and stale-response cancellation.
- Results make `assessment` primary, keep raw dataset labels secondary, display
  uncalibrated P(class 1), threshold, limited pH reliability, every server warning,
  input snapshots and per-feature training q01–q99 reference bars. It never calls
  engineering ranges BIS limits or claims safety certification.
- Optional local SQLite history is off by default. `?save_history=true` stores
  server-validated inputs/results only after successful inference, keeps at most
  200 newest records, supports list/view/delete/clear and returns a receipt.
  History is local-demo unauthenticated storage, not a secure multi-user database.
- Frontend tests cover schema serialization, null/blank/range checks, pH-only
  rejection, assessment priority and routes. Backend tests cover opt-in/out,
  round-trip, retention/deletion, rejected requests and static serving.

Verified:

```bash
/tmp/omnirush/aqua-epoch-full-venv/bin/python -m pytest -q
npm test --prefix frontend
```

Results: **161 Python tests passed**, **3 frontend tests passed**, one existing
Starlette TestClient deprecation warning. No public hosting or browser automation
was performed.

## Phase 5 — completed UI polish and usability

- Refined visual hierarchy: branded header, demo-step navigation, readiness,
  setup and interpretation sections, compact result summary and clearer status
  states.
- Responsive layout for narrow screens, keyboard-visible focus, skip link,
  labelled controls, live regions, accessible error links, reduced-motion support
  and print-friendly output are in place.
- Loading, success, warning, error, empty-history and unavailable-model states
  now use consistent visual treatment. Warning rows are severity-aware without
  hiding any server-provided text.
- Results remain scientifically conservative: quick/insufficient evidence is
  caution-styled, raw labels remain secondary, pH is limited reliability, and no
  UI range is presented as a BIS or safe-water limit.
- Existing API v2, local history and schema-driven field behavior are unchanged.

Verified from project root:

```bash
/tmp/omnirush/aqua-epoch-full-venv/bin/python -m pytest -q
npm test --prefix frontend
```

Results: **161 Python tests passed**, **3 frontend tests passed**, one existing
Starlette TestClient deprecation warning. No public hosting or browser automation
was performed; visual polish is covered by static/DOM tests and the same-origin
smoke checks from Phase 4.

## Phase 6 — completed deployment readiness and rehearsal

- `backend.demo` is a cross-platform loopback-only launcher with check-only
  preflight for all four models, runtime compatibility and frontend assets.
- `scripts/serve_demo.sh` delegates to the launcher and detects an installed
  runtime; it never installs packages or opens a public bind.
- `scripts/phase6_check.py` runs the real Uvicorn subprocess with isolated history,
  checks UI/API/security/error flows, bounded concurrency, startup, RSS and
  latency, then shuts the server down and writes new evidence only.
- No dataset is needed for inference rehearsal. No public hosting, TLS, browser,
  internet/offline network isolation or multi-user security was claimed.

Verified from project root:

```bash
/tmp/omnirush/aqua-epoch-full-venv/bin/python -m backend.demo --check-only --release artifacts/phase3
PYTHON_BIN=/tmp/omnirush/aqua-epoch-full-venv/bin/python bash scripts/serve_demo.sh --check-only --release artifacts/phase3
/tmp/omnirush/aqua-epoch-full-venv/bin/python scripts/phase6_check.py --samples 4 --concurrency 2 --output reports/phase6/deployment_check.json
/tmp/omnirush/aqua-epoch-full-venv/bin/python -m pytest -q
npm test --prefix frontend
```

Targeted deployment tests: **10 passed**. Real checker passed with four-model
readiness, same-origin UI/API routes, isolated opt-in history, private-path and
invalid-input checks, and bounded concurrent requests. Full suite and frontend
suite remain the required final checks; see `reports/phase6/REVIEW.md`.

## Remaining phase — one, start only after `next`

1. **Phase 7: technical PDF and submission.** Report/deck after the prototype,
   final README/team information, repository packaging and submission checklist.
   Public publishing/upload requires user authorization.

Do not start Phase 7 during Phase 6 completion. A first-prize result cannot be
guaranteed; keep all model limitations and evidence visible.
