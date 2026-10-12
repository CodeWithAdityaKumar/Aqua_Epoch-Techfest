# Phase 4 — functional frontend and local demo history

The FastAPI app serves a schema-driven frontend at `/` and allowlisted assets at
`/static/`; no Node server, CDN or CORS setup is needed at runtime. It supports
full/quick combined inference, independent classification/pH tasks,
`/models`-driven fields, explicit missing values, numeric validation,
loading/refresh/reset, clearly labelled synthetic training-median examples and
stale-request cancellation.

Results make `assessment` primary, keep raw dataset labels secondary, display
uncalibrated P(class 1), thresholds, limited pH reliability, all warning objects,
input snapshots and per-feature training q01–q99 reference bars. Engineering
ranges are not shown as BIS/safe-water limits. Quick mode remains an
insufficient-evidence screening flow.

Prediction routes accept `?save_history=true`; the checkbox is off by default.
Successful validated results can be stored in local SQLite, retained to the
newest 200 records, listed/viewed/deleted/cleared, and returned with a receipt.
Opt-out inference does not create the database. Failed requests are not stored.
This is unauthenticated local-demo history, not a multi-user account. Storage
defaults to `.local/predictions.sqlite3` and is git-ignored.

Verification:

```bash
/tmp/omnirush/aqua-epoch-full-venv/bin/python -m pytest -q
npm test --prefix frontend
```

Results: **161 Python tests passed**, **3 frontend tests passed**, one existing
Starlette TestClient dependency deprecation warning. No browser automation,
public hosting or multi-user security claim was made.

Three phases remain: Phase 5 UI polish/accessibility, Phase 6 deployment and
rehearsal, and Phase 7 technical PDF/submission package.
