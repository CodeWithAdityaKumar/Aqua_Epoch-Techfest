"""Check a local demo using the real app, promoted artifacts and loopback HTTP.

No original dataset is needed. This is contextual smoke/load evidence, not a
production benchmark or verification of internet/offline network isolation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import argparse
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
# Direct script invocation puts scripts/, rather than the repository, on sys.path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.serving_contract import runtime_versions


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _http(base, route, payload=None, *, timeout=15):
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = Request(base + route, data=body, headers={"Content-Type": "application/json"})
    # Loopback must not accidentally go through an inherited corporate proxy.
    opener = build_opener(ProxyHandler({}))
    try:
        response = opener.open(request, timeout=timeout)
    except HTTPError as error:
        response = error
    with response:
        text = response.read().decode("utf-8")
        content = json.loads(text) if response.headers.get_content_type() == "application/json" else text
        return response.code, content


def _require(condition, message):
    # Checks must still run with python -O.
    if not condition:
        raise RuntimeError(message)


def _stop(process):
    try:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        process.kill()
        process.wait()


def _diagnostics(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-2000:]
    except OSError:
        return "(diagnostics unavailable)"


@contextmanager
def _server(release):
    with tempfile.TemporaryDirectory(prefix="aqua-phase6-") as temporary:
        history = Path(temporary) / "history.sqlite3"
        diagnostics = Path(temporary) / "server.stderr.log"
        port = _port()
        env = {**os.environ, "AQUA_EPOCH_ARTIFACTS_DIR": str(release),
               "AQUA_EPOCH_HISTORY_DB": str(history), "PYTHONPATH": str(ROOT)}
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1",
             "--port", str(port), "--workers", "1", "--no-access-log"],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=diagnostics.open("w", encoding="utf-8"),
        )
        try:
            yield process, f"http://127.0.0.1:{port}", history, diagnostics
        finally:
            _stop(process)
            # Popen owns this descriptor; close it after the child has exited.
            if process.stderr:
                process.stderr.close()


def _ready(process, base, diagnostics, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"demo server exited before readiness (exit code {code}): {_diagnostics(diagnostics)}")
        try:
            status, health = _http(base, "/health", timeout=min(2, max(.01, deadline - time.monotonic())))
        except (URLError, TimeoutError, ConnectionError):
            time.sleep(.1)
            continue
        if status == 200:
            _require(isinstance(health, dict) and health.get("ready"), "promoted release is not ready; check artifacts/runtime compatibility")
            return health
        time.sleep(.1)
    raise RuntimeError("demo server did not become ready within the startup timeout")


def _rss(pid):
    """Best-effort current server RSS, not peak usage or system-wide memory."""
    if os.name == "posix":
        try:
            result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                                    capture_output=True, text=True, timeout=2, check=True)
            return {"bytes": int(result.stdout.strip()) * 1024, "source": "ps rss (KiB)"}
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return {"bytes": None, "source": "unavailable on this host"}


def _exercise(process, base, history, diagnostics, samples, concurrency, report):
    started = time.perf_counter()
    health = _ready(process, base, diagnostics)
    report["startup_seconds"] = time.perf_counter() - started
    report["release_id"] = health["release_id"]
    report["routes"]["/health"] = 200
    for route, marker in (("/", "Aqua Epoch"), ("/static/app.js", "createApp")):
        status, content = _http(base, route)
        _require(status == 200 and isinstance(content, str) and marker in content, f"failed page/static check: {route}")
        report["routes"][route] = status
    status, models = _http(base, "/models")
    _require(status == 200 and len(models["models"]) == 4 and
             all(model["available"] for model in models["models"].values()), "four available models required")
    report["routes"]["/models"] = status
    full = {name: bounds["median"] for name, bounds in models["models"]["classification_full"]["reference_ranges"].items()}
    quick = {name: full[name] for name in ("Turbidity", "Hardness", "TDS")}
    for route, payload in (("/predict/full", full), ("/predict/quick", quick)):
        status, result = _http(base, route, payload)
        _require(status == 200 and result["release_id"] == health["release_id"], f"prediction failed: {route}")
        report["routes"][route] = status

    report["resources"]["server_rss_before_load"] = _rss(process.pid)
    def one(_):
        started = time.perf_counter()
        status, result = _http(base, "/classify/quick", quick)
        elapsed = time.perf_counter() - started
        _require(status == 200 and result["release_id"] == health["release_id"], "concurrent classification failed")
        return elapsed * 1000

    workers = min(concurrency, samples)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        latencies = sorted(pool.map(one, range(samples)))
    elapsed = time.perf_counter() - started
    report["concurrency"] = {
        "requests": samples, "workers": workers, "requested_workers": concurrency,
        "all_http_200": True, "elapsed_seconds": elapsed, "requests_per_second": samples / elapsed,
        "latency_ms": {"min": min(latencies), "mean": statistics.mean(latencies),
                       "p50": statistics.median(latencies), "p95": latencies[math.ceil(.95 * samples) - 1],
                       "max": max(latencies)},
        "context": "Warm /classify/quick, repeated median payload, client-observed HTTP latency excluding executor queue time; throughput includes executor overhead. p95 is nearest-rank. Small local sample, not an SLA.",
    }
    report["resources"]["server_rss_after_load"] = _rss(process.pid)

    _require(not history.exists(), "inference unexpectedly created history without opt-in")
    status, result = _http(base, "/predict/quick?save_history=true", quick)
    _require(status == 200 and result.get("history", {}).get("saved") and history.is_file(), "isolated history save failed")
    status, records = _http(base, "/history")
    _require(status == 200 and len(records["items"]) == 1 and
             records["items"][0]["id"] == result["history"]["id"], "isolated history read failed")
    report["history"] = {"storage": "temporary isolated SQLite; removed after run",
                         "opt_in_save_and_read": True, "no_implicit_save": True}
    for path in ("/static/../README.md", "/static/.gitignore", "/.env"):
        code, _ = _http(base, path)
        _require(code in {404, 400, 422}, f"private path check failed: {path}")
    status, _ = _http(base, "/classify/quick", {"Turbidity": True, "Hardness": 150, "TDS": 1200})
    _require(status == 422, "invalid input was not rejected with 422")
    report["security"] = {"private_paths_blocked": True, "invalid_input_422": True,
                          "scope": "Only the listed path probes and boolean-input rejection; not a security audit."}


def _write_report(output, report):
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a competing writer after preflight.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase6/deployment_check.json")
    args = parser.parse_args(argv)
    if not 1 <= args.samples <= 100 or not 1 <= args.concurrency <= 16:
        parser.error("--samples must be 1..100 and --concurrency 1..16")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new evidence path")
    release = ROOT / "artifacts/phase3"
    if not (release / "manifest.json").is_file():
        parser.error(f"Missing promoted release: {release}")
    report = {"status": "running", "scope": {
        "transport": "real uvicorn subprocess over loopback HTTP",
        "dataset_required": False, "network_isolation_verified": False,
        "not_tested": ["public hosting", "TLS", "internet connectivity or latency", "offline/network isolation",
                       "multi-user auth", "production SLA", "browser interaction"],
    }, "runtime": runtime_versions(), "routes": {}, "security": {}, "history": {}, "resources": {
        "platform": platform.platform(), "machine": platform.machine(), "logical_cpu_count": os.cpu_count(),
        "server_workers": 1,
        "context": "RSS snapshots of the server process only, not peak memory. Host CPU count is not a CPU quota. No CPU utilization measurement; timings depend on this host and competing workloads.",
    }}
    started = time.perf_counter()
    actual_history = ROOT / ".local" / "predictions.sqlite3"
    actual_signature = actual_history.stat() if actual_history.exists() else None
    with _server(release) as (process, base, history, diagnostics):
        _exercise(process, base, history, diagnostics, args.samples, args.concurrency, report)
    current_signature = actual_history.stat() if actual_history.exists() else None
    _require(actual_signature == current_signature, "actual local history database changed")
    report["history"]["actual_local_db_unchanged"] = True
    report["elapsed_seconds"] = time.perf_counter() - started
    report["status"] = "passed"
    _write_report(args.output, report)
    print(json.dumps(report, indent=2))
    return report


def _interrupted(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    # SIGTERM from a cancelled CLI run should unwind the child-owning context too.
    signal.signal(signal.SIGTERM, _interrupted)
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Phase 6 check failed: {exc}") from None
