"""Mutable mock back-office systems: user directory (identity provider) and ticket database.

The seed data in ``data/`` is copied into ``runtime/`` on first use so demos and
evaluations can change state (unlock accounts, open tickets) without editing the seeds.
``reset_runtime()`` restores a clean state.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_DIR, RUNTIME_DIR

DIRECTORY_PATH = RUNTIME_DIR / "directory.json"
TICKETS_DB = RUNTIME_DIR / "tickets.db"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def reset_runtime() -> None:
    """Restore seed data and clear tickets (keeps the vector store, which is content-hashed)."""
    RUNTIME_DIR.mkdir(exist_ok=True)
    shutil.copy(DATA_DIR / "directory.json", DIRECTORY_PATH)
    # Clear rows instead of deleting the file: Windows refuses to delete an SQLite file another connection holds.
    with _db() as con:
        con.execute("DELETE FROM tickets")
        con.execute("DELETE FROM sqlite_sequence WHERE name = 'tickets'")


# ---------------------------------------------------------------- directory (mock identity provider)
def _load_directory() -> dict:
    if not DIRECTORY_PATH.exists():
        reset_runtime()
    return json.loads(DIRECTORY_PATH.read_text(encoding="utf-8"))


def get_user(username: str) -> dict | None:
    username = (username or "").lower().strip()
    return next((u for u in _load_directory()["users"] if u["username"] == username), None)


def update_user(username: str, **changes) -> dict:
    data = _load_directory()
    for u in data["users"]:
        if u["username"] == username:
            u.update(changes)
            DIRECTORY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return u
    raise KeyError(username)


def list_users() -> list[dict]:
    return _load_directory()["users"]


# ---------------------------------------------------------------- tickets
_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    requester   TEXT NOT NULL,
    category    TEXT NOT NULL,
    priority    TEXT NOT NULL,
    queue       TEXT NOT NULL,
    status      TEXT NOT NULL,
    title       TEXT NOT NULL,
    summary     TEXT NOT NULL,
    steps_tried TEXT NOT NULL,
    transcript  TEXT NOT NULL,
    resolved_by TEXT,
    external_url TEXT
)
"""


def _db() -> sqlite3.Connection:
    RUNTIME_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(TICKETS_DB)
    con.row_factory = sqlite3.Row
    con.execute(_SCHEMA)
    return con


def create_ticket(*, requester: str, category: str, priority: str, queue: str, title: str, summary: str,
                  steps_tried: list[str] | None = None, transcript: str = "", status: str = "open",
                  resolved_by: str | None = None) -> dict:
    with _db() as con:
        cur = con.execute(
            "INSERT INTO tickets (created_at, requester, category, priority, queue, status, title, summary,"
            " steps_tried, transcript, resolved_by) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), requester, category, priority, queue, status, title, summary,
             json.dumps(steps_tried or []), transcript, resolved_by),
        )
        return get_ticket(cur.lastrowid, con)


def get_ticket(ticket_id: int, con: sqlite3.Connection | None = None) -> dict | None:
    con = con or _db()
    row = con.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if not row:
        return None
    t = dict(row)
    t["steps_tried"] = json.loads(t["steps_tried"])
    t["number"] = f"INC{t['id']:06d}"
    return t


def set_external_url(ticket_id: int, url: str) -> None:
    with _db() as con:
        con.execute("UPDATE tickets SET external_url = ? WHERE id = ?", (url, ticket_id))


def list_tickets(status: str | None = None, queue: str | None = None, limit: int = 50) -> list[dict]:
    sql, args = "SELECT id FROM tickets WHERE 1=1", []
    if status:
        sql += " AND status = ?"
        args.append(status)
    if queue:
        sql += " AND queue = ?"
        args.append(queue)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    con = _db()
    return [get_ticket(r["id"], con) for r in con.execute(sql, args).fetchall()]


def update_ticket_status(ticket_id: int, status: str, resolved_by: str | None = None) -> dict | None:
    with _db() as con:
        con.execute("UPDATE tickets SET status = ?, resolved_by = COALESCE(?, resolved_by) WHERE id = ?",
                    (status, resolved_by, ticket_id))
        return get_ticket(ticket_id, con)


def load_services() -> list[dict]:
    return json.loads((DATA_DIR / "services.json").read_text(encoding="utf-8"))["services"]


def read_log(name: str) -> str:
    path: Path = DATA_DIR / "logs" / f"{name}.log"
    if not path.resolve().is_relative_to((DATA_DIR / "logs").resolve()) or not path.exists():
        raise FileNotFoundError(name)
    return path.read_text(encoding="utf-8")
