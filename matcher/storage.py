"""Local SQLite storage: your profile, your saved CV and the application tracker."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from . import config

DEFAULT_PROFILE: dict[str, Any] = {
    "full_name": "", "street": "", "postal_city": "", "phone": "", "email": "",
    "place": "",               # city written before the date, e.g. "Berlin"
    "german_level": "",        # CEFR, e.g. "A2"
    "is_student": True,
    "other_job_hours": 0,      # hours/week you already work elsewhere
    "availability": "",        # e.g. "from 01.12.2026, 15–20 h/week"
    "default_tone": "formal",
    "default_length": "standard",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cvs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL, text TEXT NOT NULL, uploaded_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    company TEXT, role TEXT, location TEXT, job_url TEXT, jd_text TEXT,
    overall_score REAL, analysis_json TEXT, cover_letter_json TEXT,
    status TEXT NOT NULL DEFAULT 'analysed', notes TEXT DEFAULT '');
"""


class Storage:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or config.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    # ---------------- profile
    def get_profile(self) -> dict[str, Any]:
        with self._conn() as c:
            row = c.execute("SELECT value FROM settings WHERE key='profile'").fetchone()
        stored = json.loads(row["value"]) if row else {}
        return {**DEFAULT_PROFILE, **stored}

    def save_profile(self, profile: dict[str, Any]) -> None:
        merged = {**DEFAULT_PROFILE, **profile}
        with self._conn() as c:
            c.execute("INSERT INTO settings(key, value) VALUES('profile', ?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(merged),))

    # ---------------- other settings (model choice, cloud consent …)
    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._conn() as c:
            row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO settings(key, value) VALUES(?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))

    # ---------------- CV
    def save_cv(self, filename: str, text: str) -> int:
        with self._conn() as c:
            c.execute("UPDATE cvs SET is_active=0")
            cur = c.execute("INSERT INTO cvs(filename, text, uploaded_at, is_active) VALUES(?,?,?,1)",
                            (filename, text, self._now()))
            return int(cur.lastrowid)

    def get_active_cv(self) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM cvs WHERE is_active=1 ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def clear_cv(self) -> None:
        with self._conn() as c:
            c.execute("UPDATE cvs SET is_active=0")

    # ---------------- applications
    def add_application(self, analysis: dict[str, Any], jd_text: str, job_url: str = "") -> int:
        job = analysis.get("job", {})
        now = self._now()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO applications(created_at, updated_at, company, role, location, job_url, jd_text, "
                "overall_score, analysis_json, status) VALUES(?,?,?,?,?,?,?,?,?, 'analysed')",
                (now, now, job.get("company", ""), job.get("job_title", ""), job.get("location", ""), job_url,
                 jd_text, analysis.get("overall"), json.dumps(analysis, ensure_ascii=False)))
            return int(cur.lastrowid)

    def update_application(self, app_id: int, **fields: Any) -> None:
        allowed = {"company", "role", "location", "job_url", "status", "notes", "cover_letter_json", "overall_score"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        if not fields:
            return
        if "cover_letter_json" in fields and not isinstance(fields["cover_letter_json"], str):
            fields["cover_letter_json"] = json.dumps(fields["cover_letter_json"], ensure_ascii=False)
        sets = ", ".join(f"{k}=?" for k in fields) + ", updated_at=?"
        with self._conn() as c:
            c.execute(f"UPDATE applications SET {sets} WHERE id=?", (*fields.values(), self._now(), app_id))

    def list_applications(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT id, created_at, updated_at, company, role, location, job_url, overall_score, "
                             "status, notes, cover_letter_json IS NOT NULL AS has_letter "
                             "FROM applications ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]

    def get_application(self, app_id: int) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM applications WHERE id=?", (app_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["analysis"] = json.loads(d.pop("analysis_json") or "{}")
        d["cover_letter"] = json.loads(d.pop("cover_letter_json") or "null")
        return d

    def delete_application(self, app_id: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM applications WHERE id=?", (app_id,))
