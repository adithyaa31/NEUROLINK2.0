import sqlite3
from contextlib import contextmanager
from .config import DATABASE_PATH

@contextmanager
def connect():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATABASE_PATH)
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    finally:
        db.close()

def initialize():
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT, kind TEXT NOT NULL, label TEXT NOT NULL, confidence REAL,
            latency_ms REAL, metadata TEXT DEFAULT '{}')""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_events_kind_id ON events(kind, id DESC)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")

def log_event(kind: str, label: str, confidence=None, latency_ms=None, session_id=None, metadata="{}"):
    with connect() as db:
        cur = db.execute("INSERT INTO events(session_id,kind,label,confidence,latency_ms,metadata) VALUES(?,?,?,?,?,?)",
                         (session_id, kind, label, confidence, latency_ms, metadata))
        return cur.lastrowid

def fetch_events(limit=100, kind=None):
    with connect() as db:
        if kind:
            rows = db.execute("SELECT * FROM events WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit)).fetchall()
        else:
            rows = db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

def delete_events_before(cutoff_timestamp: str):
    """Delete event rows older than an explicit ISO timestamp cutoff."""
    with connect() as db:
        cur = db.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_timestamp,))
        return cur.rowcount
