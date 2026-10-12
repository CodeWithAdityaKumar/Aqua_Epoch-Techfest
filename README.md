# Aqua Epoch — reproducible water-quality ML

Competition prototype: potability classification and pH regression, with full
chemical-profile and three-input quick modes. Core predictions use scikit-learn,
not LLM APIs. **Not a certified drinking-water safety system.**

## Current stage

Phase 6 is implemented: the backend serves a version-checked **demo-only**
release in `artifacts/phase3/`. Original Phase 1 artifacts and Phase 2 evidence
are preserved. Quick mode always returns insufficient evidence for a reliable
safety assessment; pH estimates explicitly have limited reliability. The working
frontend is served at `/` with responsive/accessibility polish. Public deployment
and the PDF are not finished. Portable launcher/checker scripts are included.
See [PHASES.md](PHASES.md).

Start with:

- [Phase 3 Colab/Jupyter backend notebook](notebooks/phase3_backend.ipynb)
- [Frontend UI](frontend/index.html)
- [Phase 3 verification and policy](reports/phase3/REVIEW.md)
- [API v2 contract](backend/README.md)
- [Phase 4 verification](reports/phase4/REVIEW.md)
- [Phase 5 UI review](reports/phase5/REVIEW.md)
- [Phase 6 deployment review](reports/phase6/REVIEW.md)
- [Phase 2 Colab/Jupyter notebook](notebooks/phase2_validation.ipynb)
- [Validated experiment review](reports/phase2_validated/REVIEW.md)
- `reports/phase2_validated/comparison.csv`: all candidates' CV scores
- `reports/phase2_validated/summary.json`: compact metrics/audit/stress results
- `reports/phase2_validated/plots/`: six EDA/evaluation figures

## Local environment

Python 3.10+ is required (verified locally with 3.12). Use a project-local
environment, not global installs:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

`requirements.txt` is the runtime/plotting set; `requirements-dev.txt` adds tests
and notebook execution tools. Ranges support fresh local/Colab installs; exact
versions used for the recorded experiments are in
`reports/phase2_validated/runtime_config.json`. Retrain using the serving
environment: cross-version sklearn pickle compatibility is not promised.

To reuse the supplied Phase 3 release, use Python 3.12 and install its recorded
versions in the same environment:

```bash
python -m pip install -r requirements-dev.txt -r artifacts/phase3/requirements-serving.txt
```

The loader rejects incompatible runtime versions. Alternatively evaluate and
promote in your own environment as described below.

## Run Phase 2

Keep the original CSV in the project root. No source data is overwritten.
Use a fresh output directory on each run (the evaluator refuses an existing
nonempty directory):

```bash
python -m ml.evaluate \
  --dataset "Aqua_Epoch_BIS_Water_Quality_Dataset.xls - Aqua_Epoch_BIS_Water_Quality_Dataset.csv" \
  --output reports/phase2_rerun \
  --folds 3 --trees 80 --permutation-repeats 3
python -m ml.report --results-dir reports/phase2_rerun
```

This compares four model families across 12 predefined task/mode/policy/cohort
combinations. It uses two CPU threads, no GPU. Allow several minutes or longer
on a slower laptop/Colab runtime. The small local run is not a latency benchmark.
Reporting regenerates derived summaries/plots; it does not fit new models.

Audit only:

```bash
python -m ml.quality --output reports/data_audit.json
```

The `reports/phase2/` directory is the preserved first exploratory run. Its
original FPR-budget threshold could fall below 0.5; **do not use it for the
deployment decision**. `reports/phase2_validated/` is the corrected run whose
safety-oriented thresholds never make more safe predictions than threshold 0.5.
This protocol change was an engineering guard, not model selection on holdout.

## Promote a demo release (no estimator refitting)

```bash
python -m ml.promote \
  --source reports/phase2_validated \
  --dataset "Aqua_Epoch_BIS_Water_Quality_Dataset.xls - Aqua_Epoch_BIS_Water_Quality_Dataset.csv" \
  --output artifacts/phase3_new
```

Use trusted locally generated source candidates only. The command validates
source/runtime/dataset hashes and CV selection, computes training-only reference
ranges, preserves fitted models and thresholds, and atomically publishes a new
release. It refuses nonempty outputs. No holdout rows are refitted. Use
`--source reports/phase2_rerun` for a run made in your current environment.
Set `AQUA_EPOCH_ARTIFACTS_DIR` to the new release when starting the server.
Generated `promotion.json` records provenance; `requirements-serving.txt`
records exact serialization dependencies. Approval is **demo-only**, not water
safety certification.

## Google Colab / Jupyter

1. Put the **complete project folder** in `MyDrive/Aqua_Epoch`, including `ml/`,
   `backend/`, requirements files, and the CSV. Uploading the notebook alone is
   insufficient because it intentionally reuses the tested Python modules.
2. Open `notebooks/phase3_backend.ipynb` in Colab. The Phase 2 notebook remains
   available for deeper EDA/model comparisons.
3. Approve Drive mounting; adjust `ROOT` if your folder has another name.
4. Run cells top to bottom. Colab defaults to real evaluation and promotion in
   its runtime, saving timestamped outputs on Drive. Local mode reuses
   `artifacts/phase3`; set `REBUILD_RELEASE=True` to rebuild. Existing releases
   are never overwritten.
5. If package installation upgrades already-loaded libraries, restart runtime.

The Phase 3 notebook exercises the real app with TestClient, checks direct-model
parity and validation, and offers an optional temporary loopback HTTP check.
It does not create a public deployment/tunnel. Both notebooks have been locally
executed in reuse mode; real evaluation/promotion was verified through the CLI.
Google-hosted execution itself has not been verified in your account.

### Colab + Cloudflare public demo link

The repository can remain private. In Colab, add a fine-grained GitHub token as
a Colab Secret named `GITHUB_TOKEN` with **Contents: Read-only** access to this
repository. Do not paste the token into a notebook cell or commit it. Then run
these cells from a fresh Colab runtime:

```python
# Clone the private repository without putting the token in the clone URL.
from google.colab import userdata
from pathlib import Path
import os, stat, subprocess

token = userdata.get("GITHUB_TOKEN")
assert token, "Add GITHUB_TOKEN in Colab Secrets first."
os.environ["GITHUB_TOKEN"] = token

askpass = Path("/tmp/git-askpass.sh")
askpass.write_text("""#!/bin/sh
case \"$1\" in
  *Username*) printf '%s\\n' 'x-access-token' ;;
  *) printf '%s\\n' \"$GITHUB_TOKEN\" ;;
esac
""")
askpass.chmod(stat.S_IRWXU)
env = os.environ.copy()
env.update({"GIT_ASKPASS": str(askpass), "GIT_TERMINAL_PROMPT": "0"})

repo = Path("/content/Aqua_Epoch")
if not repo.exists():
    subprocess.run([
        "git", "clone",
        "https://github.com/CodeWithAdityaKumar/Aqua_Epoch-Techfest.git",
        str(repo),
    ], env=env, check=True)
```

```python
# Install the serving dependencies matching the promoted joblib release.
%cd /content/Aqua_Epoch
!python -m pip install -q -r backend/requirements.txt -r artifacts/phase3/requirements-serving.txt
!python -m pip check
```

```python
# Start the loopback-only Aqua Epoch server inside the Colab runtime.
import subprocess, time, requests

server = subprocess.Popen([
    "python", "-m", "backend.demo",
    "--host", "127.0.0.1", "--port", "8000",
], cwd="/content/Aqua_Epoch", stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

for _ in range(30):
    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if health.ok:
            print(health.json())
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise RuntimeError("Aqua Epoch server did not become ready")
```

Install and start a **temporary Cloudflare Quick Tunnel**. It does not require
a Cloudflare account or custom domain, and prints a public HTTPS URL. The URL
changes when the Colab runtime restarts and stops when the runtime/process ends.

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared
```

```python
import re, subprocess, time
from pathlib import Path

log_path = Path("/tmp/cloudflared.log")
log = log_path.open("w")
cloudflared = subprocess.Popen([
    "cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate",
], stdout=log, stderr=subprocess.STDOUT)

public_url = None
for _ in range(30):
    time.sleep(1)
    text = log_path.read_text(errors="ignore")
    match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", text)
    if match:
        public_url = match.group(0)
        break
if not public_url:
    raise RuntimeError(log_path.read_text(errors="ignore"))
print("LIVE URL:", public_url)
```

Open the printed `LIVE URL` in a browser. The frontend is mounted at `/`, so
no extra path is needed. For a stable hostname such as `water.example.com`,
create a named Cloudflare Tunnel in Cloudflare Zero Trust, route its Public
Hostname to `http://127.0.0.1:8000`, save the tunnel token as a Colab Secret
named `CLOUDFLARE_TUNNEL_TOKEN`, and replace the Quick Tunnel cell with:

```python
from google.colab import userdata
import subprocess

cloudflared = subprocess.Popen([
    "cloudflared", "tunnel", "run", "--token",
    userdata.get("CLOUDFLARE_TUNNEL_TOKEN"),
], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
print("Open the Public Hostname configured in Cloudflare Zero Trust.")
```

This service has unauthenticated demo/history routes. Do not enter personal or
real laboratory data, keep **Save history** off for a public Quick Tunnel, and
use Cloudflare Access in front of a named tunnel if the link must be restricted.
The app remains a competition demo, not drinking-water certification.

## Current backend — API v2

```bash
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/docs` after starting the server. Routes:
`GET /health`, `GET /models`, `POST /predict/full`, `POST /predict/quick`,
`POST /classify/full`, `POST /classify/quick`, `POST /ph/full`, `POST /ph/quick`.
The quick request contains numeric `Turbidity`, `Hardness`, `TDS`. Standalone
`/ph/full` accepts 17 inputs without measured pH or Potability. Full combined
inference accepts all 18 keys; explicit null means genuinely unmeasured. Missing
keys/invalid values return 422. Incompatible models return 503. `/models` exposes
schemas, engineering bounds and training reference ranges.

API v2 replaces ambiguous Phase 1 response fields. `probability_safe` means
uncalibrated `P(Potability=1)`; classification uses the stored threshold.
Combined responses have `classification` and `regression` objects;
`regression.estimate` replaces `pH_prediction`. Always display `assessment` and
warnings, not just `model_label`. Direct Phase 1/Phase 2 candidate files are
refused: use `ml.promote` first. Model calls use named DataFrames. There is no
fallback confidence of 1.0, silent pH substitution or output clipping.

Real HTTP verification (temporary localhost server; shuts down afterwards):

```bash
python -m backend.verify_release \
  --dataset "Aqua_Epoch_BIS_Water_Quality_Dataset.xls - Aqua_Epoch_BIS_Water_Quality_Dataset.csv" \
  --output reports/phase3/http_rerun.json --samples 30
```

Choose a new output file each time. Timings are local sequential loopback
latency, not internet latency, concurrent-load performance or an SLA.

## Tests

```bash
python -m pytest -q
```

Tests cover schema/coercion audits, fold-local imputation, target exclusion,
split disjointness, exact duplicate/conflict handling, threshold direction,
portable bundles, immutable promotion, version/hash/schema refusal, safe
422/503 errors, named-frame inference, abstention, and both local notebooks.
Evidence-backed tests skip if generated results/releases are absent; evaluate
and promote first to exercise those. One Starlette TestClient dependency
deprecation warning remains; serving no longer emits positional-feature-name
warnings. No dedicated linter/typechecker/build tool is configured.

## Evidence and limitations

- Never interpret dataset potability labels as complete water certification;
  microbial measurements and source provenance are missing.
- No official feature-unit dictionary/BIS-limit mapping has been supplied.
- Quick mode has a serious information limit; reducing false-safe predictions
  collapses safe recall. Its output needs a limited-screening presentation.
- pH regressors only modestly beat a mean baseline; removing invalid targets
  changes the population and is not proof of dramatic model improvement.
- Exact duplicates are checked. Near-duplicate/source-related samples remain
  an unresolved generalisation risk without provenance/group IDs.
- These are internal, non-nested comparisons after Phase 1 exploration; no
  externally unseen test-set claim is made.
- Only load trusted joblib files. They can execute Python code.

Phase 3 published a local demo release only. No public repository, upload,
public deployment, frontend or PDF was created in this phase.
