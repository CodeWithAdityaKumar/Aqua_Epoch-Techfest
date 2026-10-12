"""Cross-platform, loopback-only launcher: python -m backend.demo."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_HELP = (
    "Use the Python major/minor and exact package versions recorded in the "
    "promoted manifest's training_versions, or retrain and promote in this environment. "
    "From the repository root, setup commands are: python -m pip install -r requirements.txt; "
    "python -m pip check. Then retry: python -m backend.demo --check-only. "
    "Installing requirements alone does not guarantee compatibility with an older release."
)


class PreflightError(RuntimeError):
    """Fixed public diagnostics, never exception details or configured paths."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default message can echo arbitrary private argument values.
        raise PreflightError("Invalid launcher arguments; run python -m backend.demo --help.")


def _port(value):
    try:
        port = int(value)
    except (ValueError, TypeError):
        raise PreflightError("Port must be an integer from 1 to 65535.") from None
    if not 1 <= port <= 65535:
        raise PreflightError("Port must be an integer from 1 to 65535.")
    return port


def _repo_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def preflight(release):
    """Validate the complete release and public UI; do not bind or write files."""
    from .artifacts import ArtifactStore
    from .frontend import ASSETS, ROOT as frontend_root
    from ml.serving_contract import MODEL_SPECS

    store = ArtifactStore(release)
    readiness = store.readiness()
    if (not readiness.get("ready") or set(readiness.get("models", {})) != set(MODEL_SPECS)
            or not all(item.get("available") for item in readiness["models"].values())):
        raise PreflightError(
            "Model preflight failed: all four promoted models must be available and compatible. "
            "Check --release (or AQUA_EPOCH_ARTIFACTS_DIR) and restore a complete trusted release. "
            + RUNTIME_HELP
        )
    root = frontend_root.resolve()
    for name in ("index.html", *ASSETS):
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise PreflightError("Frontend preflight failed: restore all bundled public frontend files.")
    return store


def main(argv=None):
    """Return a process exit code; check-only never invokes the server."""
    parser = _Parser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check models and UI without starting a server")
    parser.add_argument("--host", default=os.getenv("AQUA_EPOCH_HOST", "127.0.0.1"),
                        help="127.0.0.1, localhost (bound as 127.0.0.1), or ::1 only")
    parser.add_argument("--port", default=os.getenv("AQUA_EPOCH_PORT", "8000"), help="TCP port, 1–65535 (default: 8000)")
    parser.add_argument("--release", "--artifacts-dir", default=os.getenv("AQUA_EPOCH_ARTIFACTS_DIR", "artifacts/phase3"),
                        help="Trusted release directory; relative paths are repository-relative")
    parser.add_argument("--history-db", "--history", dest="history_db",
                        default=os.getenv("AQUA_EPOCH_HISTORY_DB", ".local/predictions.sqlite3"),
                        help="History database; relative paths are repository-relative")
    try:
        args = parser.parse_args(argv)
        if args.host not in {"127.0.0.1", "localhost", "::1"}:
            raise PreflightError("Host must be 127.0.0.1, localhost, or ::1; unauthenticated history requires loopback-only serving.")
        port = _port(args.port)
        release, history = _repo_path(args.release), _repo_path(args.history_db)
        store = preflight(release)
    except PreflightError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ImportError:
        print("Demo runtime dependencies are unavailable. " + RUNTIME_HELP, file=sys.stderr)
        return 2
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception:
        print("Demo preflight could not complete. Check the release, frontend, and path configuration. " + RUNTIME_HELP,
              file=sys.stderr)
        return 2

    if args.check_only:
        print("Demo preflight passed: all four models and public frontend files are ready.")
        return 0
    try:
        import uvicorn
        from .app import create_app
        app = create_app(artifact_store=store, history_db=str(history))
        # Let Uvicorn bind once and report occupied ports; probing first races.
        uvicorn.run(app, host="127.0.0.1" if args.host == "localhost" else args.host,
                    port=port, access_log=False)
    except (OSError, SystemExit):
        print("Demo server could not start; check whether the selected loopback port is already in use.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
