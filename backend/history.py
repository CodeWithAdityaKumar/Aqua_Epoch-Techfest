"""Opt-in bounded local demo history, not a secure multi-user datastore.

No database is created by reads or deletes. Only successful explicit saves create
storage. Callers must expose this only on a trusted local network until auth is
implemented. SQL uses parameters and no request may choose a storage path.
"""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import UUID, uuid4

RETENTION_LIMIT = 200
DEFAULT_DB = Path(__file__).resolve().parents[1] / ".local" / "predictions.sqlite3"


class HistoryUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("history_unavailable")


def valid_id(value):
    try:
        return str(UUID(value)) == value
    except (ValueError, TypeError, AttributeError):
        return False


class HistoryStore:
    def __init__(self, path=None):
        self.path = Path(path if path is not None else DEFAULT_DB).expanduser().absolute()

    def _connect(self, create=False):
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), timeout=10)
        else:
            # mode=rw refuses creation even if a delete/read races with removal.
            conn = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, task, mode, inputs, result):
        try:
            if task not in {"combined", "classification", "regression"} or mode not in {"full", "quick"}:
                raise ValueError("Unsupported history record")
            # JSON response schemas are validated before this method is called.
            if "history" in result:
                raise ValueError("History receipt cannot be stored recursively")
            class_result = result.get("classification") if task == "combined" else result if task == "classification" else None
            ph_result = result.get("regression") if task == "combined" else result if task == "regression" else None
            record_id = str(uuid4())
            created_at = datetime.now(timezone.utc).isoformat()
            values = (record_id, created_at, task, mode, result["release_id"],
                      class_result.get("assessment") if class_result else None,
                      class_result.get("probability_safe") if class_result else None,
                      ph_result.get("estimate") if ph_result else None,
                      json.dumps(inputs, allow_nan=False), json.dumps(result, allow_nan=False))
            with closing(self._connect(create=True)) as conn, conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
                    task TEXT NOT NULL, mode TEXT NOT NULL, release_id TEXT NOT NULL,
                    assessment TEXT, probability_safe REAL, ph_estimate REAL,
                    inputs TEXT NOT NULL, result TEXT NOT NULL)""")
                conn.execute("INSERT INTO predictions (id,created_at,task,mode,release_id,assessment,probability_safe,ph_estimate,inputs,result) VALUES (?,?,?,?,?,?,?,?,?,?)", values)
                conn.execute("DELETE FROM predictions WHERE sequence NOT IN (SELECT sequence FROM predictions ORDER BY sequence DESC LIMIT ?)", (RETENTION_LIMIT,))
            return {"saved": True, "id": record_id, "error": None}
        except (sqlite3.Error, OSError, ValueError, TypeError, KeyError):
            raise HistoryUnavailable() from None

    def recent(self, limit=20):
        if type(limit) is not int or not 1 <= limit <= RETENTION_LIMIT:
            raise ValueError("History limit must be 1–200")
        try:
            if not self.path.exists():
                items = []
            else:
                with closing(self._connect()) as conn:
                    items = [dict(row) for row in conn.execute("SELECT id,created_at,task,mode,release_id,assessment,probability_safe,ph_estimate FROM predictions ORDER BY sequence DESC LIMIT ?", (limit,))]
            return {"items": items, "limit": limit, "retention_limit": RETENTION_LIMIT}
        except (sqlite3.Error, OSError):
            raise HistoryUnavailable() from None

    def get(self, record_id):
        try:
            if not valid_id(record_id) or not self.path.exists():
                return None
            with closing(self._connect()) as conn:
                row = conn.execute("SELECT id,created_at,task,mode,release_id,assessment,probability_safe,ph_estimate,inputs,result FROM predictions WHERE id=?", (record_id,)).fetchone()
            if row is None:
                return None
            record = dict(row)
            for field in ("inputs", "result"):
                record[field] = json.loads(record[field])
            return record
        except (sqlite3.Error, OSError, ValueError, TypeError):
            raise HistoryUnavailable() from None

    def delete(self, record_id=None):
        try:
            if record_id is not None and not valid_id(record_id):
                return 0
            if not self.path.exists():
                return 0
            with closing(self._connect()) as conn, conn:
                conn.execute("PRAGMA secure_delete=ON")
                cursor = conn.execute("DELETE FROM predictions") if record_id is None else conn.execute("DELETE FROM predictions WHERE id=?", (record_id,))
                return cursor.rowcount
        except (sqlite3.Error, OSError):
            raise HistoryUnavailable() from None
