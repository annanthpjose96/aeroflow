"""SQLite persistence for zone snapshots and alert events (with retention)."""

from __future__ import annotations

import json
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots(ts REAL, zone_id TEXT, count INTEGER, est_wait_s REAL, density REAL);
CREATE INDEX IF NOT EXISTS ix_snap ON snapshots(zone_id, ts);
CREATE TABLE IF NOT EXISTS events(ts REAL, type TEXT, severity TEXT, zone_id TEXT, summary TEXT, payload TEXT);
CREATE INDEX IF NOT EXISTS ix_evt ON events(ts);
"""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)

    def add_snapshot(self, ts, zone_id, count, est_wait_s, density):
        with self.lock:
            self.db.execute("INSERT INTO snapshots VALUES (?,?,?,?,?)", (ts, zone_id, count, est_wait_s, density))
            self.db.commit()

    def add_event(self, alert: dict):
        with self.lock:
            self.db.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?)",
                (
                    alert["ts"],
                    alert["type"],
                    alert["severity"],
                    alert.get("zone"),
                    alert.get("summary", ""),
                    json.dumps(
                        {"pos": alert.get("pos"), "recommended": alert.get("recommended"), **alert.get("payload", {})}
                    ),
                ),
            )
            self.db.commit()

    def latest_ts(self) -> float | None:
        with self.lock:
            r = self.db.execute("SELECT MAX(ts) m FROM snapshots").fetchone()
        return r["m"]

    def series(self, zone_id: str | None = None, since: float | None = None) -> list[dict]:
        q, args = "SELECT * FROM snapshots WHERE 1=1", []
        if zone_id:
            q += " AND zone_id=?"
            args.append(zone_id)
        if since is not None:
            q += " AND ts>=?"
            args.append(since)
        with self.lock:
            return [dict(r) for r in self.db.execute(q + " ORDER BY ts", args)]

    def events(
        self, since: float | None = None, type_: str | None = None, zone_id: str | None = None, limit: int = 100
    ) -> list[dict]:
        q, args = "SELECT * FROM events WHERE 1=1", []
        for col, val in (("type", type_), ("zone_id", zone_id)):
            if val:
                q += f" AND {col}=?"
                args.append(val)
        if since is not None:
            q += " AND ts>=?"
            args.append(since)
        with self.lock:
            rows = [dict(r) for r in self.db.execute(q + " ORDER BY ts DESC LIMIT ?", args + [limit])]
        for r in rows:
            r["payload"] = json.loads(r["payload"] or "{}")
        return rows

    def purge(self, before_ts: float) -> int:
        """Data-retention enforcement: delete everything older than `before_ts`."""
        with self.lock:
            n = self.db.execute("DELETE FROM snapshots WHERE ts<?", (before_ts,)).rowcount
            n += self.db.execute("DELETE FROM events WHERE ts<?", (before_ts,)).rowcount
            self.db.commit()
        return n
