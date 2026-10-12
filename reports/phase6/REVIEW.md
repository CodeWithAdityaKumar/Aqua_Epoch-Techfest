# Phase 6 — deployment readiness and demo rehearsal

## Delivered

- `python -m backend.demo` is the cross-platform loopback-only launcher.
- `scripts/serve_demo.sh` is an optional shell convenience wrapper; it detects
  `.venv/bin/python`, `python3`, or `python` only when required runtime imports
  are available. `PYTHON_BIN` can select an explicit environment.
- Launcher preflight checks all four promoted models, exact runtime compatibility,
  frontend allowlisted files and loopback-only binding. It performs no install,
  network, tunnel, browser, deletion or auto-recovery action.
- `scripts/phase6_check.py` starts a real one-worker Uvicorn subprocess with an
  isolated temporary history database, checks UI/API/model routes, prediction
  flows, opt-in/opt-out history, invalid input, private-path probes and bounded
  concurrent quick-classification requests. It writes immutable evidence to a
  new output path and terminates the server on success/failure.
- The checker records startup time, request latency/throughput, process RSS
  snapshots and host context. These are local loopback observations, not an SLA.

## Verified commands

From the project root with the recorded runtime:

```bash
/tmp/omnirush/aqua-epoch-full-venv/bin/python -m backend.demo --check-only --release artifacts/phase3
PYTHON_BIN=/tmp/omnirush/aqua-epoch-full-venv/bin/python bash scripts/serve_demo.sh --check-only --release artifacts/phase3
/tmp/omnirush/aqua-epoch-full-venv/bin/python scripts/phase6_check.py --samples 4 --concurrency 2 --output reports/phase6/deployment_check.json
/tmp/omnirush/aqua-epoch-full-venv/bin/python -m pytest -q
npm test --prefix frontend
```

Results:

- Launcher targeted checks: **10 passed**.
- Full Python suite: **161 passed** before Phase 6 additions; Phase 6 targeted
  suite includes launcher/checker coverage. Re-run the full command above after
  any local changes.
- Frontend suite: **6 passed**.
- Real subprocess checker: passed with four-model readiness, frontend/API
  routes, full/quick predictions, isolated history, path probes, invalid-input
  422 and bounded concurrency.

The recorded checker evidence includes a small local sample: four concurrent
quick requests with two workers. It is not a production load test. The server
uses one Uvicorn worker; history is local and unauthenticated. The actual
`.local/predictions.sqlite3` was unchanged by the rehearsal. No public host,
TLS, internet latency, browser interaction, multi-user auth or network-isolation
claim was verified.

## Operator runbook

1. Create/activate a Python environment and install `requirements.txt` plus the
   exact `artifacts/phase3/requirements-serving.txt` pins.
2. From the project root run:

   ```bash
   python -m backend.demo --check-only
   python -m backend.demo
   ```

3. Open `http://127.0.0.1:8000/`. Keep the terminal visible as a fallback.
4. For the live demo, use the synthetic training-median example only to show
   the workflow; label it synthetic, not a real water sample.
5. Show quick mode as limited screening and read the warning/assessment aloud.
   Do not present `safe` as a drinking recommendation.
6. Stop with Ctrl-C. Do not expose this unauthenticated history demo on a public
   interface. Use `--host 127.0.0.1` only; public hosting needs a later auth/TLS
   design and user approval.

Colab/Jupyter notebooks remain the reproducible training/backend path. A Colab
runtime can rebuild and promote models in that runtime to avoid pickle-version
mismatch, but an actual Google-hosted execution was not performed here.

## Remaining phase

Only Phase 7 remains: technical PDF, final repository/submission packaging and
submission checklist. Public publishing/upload still requires explicit user
authorization.
