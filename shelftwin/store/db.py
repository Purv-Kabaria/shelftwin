"""Twin store. SQLite for the prototype (spec Section 4 uses PostgreSQL in production;
the tables mirror the spec so the move is a driver swap). Every decision is logged
with its inputs (design rule 7), so any alert can be replayed."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS observation (id INTEGER PRIMARY KEY AUTOINCREMENT, slot_id TEXT, ts REAL,
  frame_id TEXT, coverage REAL, visibility REAL, quality REAL, sku_pred TEXT, s1 REAL, s2 REAL, crop_uri TEXT);
CREATE TABLE IF NOT EXISTS slot_state (slot_id TEXT PRIMARY KEY, state TEXT, score REAL, decision TEXT,
  last_clean_ts REAL, updated_ts REAL, coverage REAL, visibility REAL);
CREATE TABLE IF NOT EXISTS state_event (id INTEGER PRIMARY KEY AUTOINCREMENT, slot_id TEXT, ts REAL,
  from_state TEXT, to_state TEXT, score REAL, decision TEXT, frame_ids TEXT, engine_version TEXT, inputs TEXT);
CREATE TABLE IF NOT EXISTS task (task_id INTEGER PRIMARY KEY, slot_id TEXT, kind TEXT, priority REAL, status TEXT,
  assignee_team TEXT, created_ts REAL, ack_ts REAL, done_ts REAL, verified_ts REAL, backroom_location TEXT,
  reason_json TEXT);
CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER, user_role TEXT,
  kind TEXT, note TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS obs_slot_ts ON observation(slot_id, ts);
CREATE INDEX IF NOT EXISTS ev_slot_ts ON state_event(slot_id, ts);
"""


class TwinStore:
    def __init__(self, path: str | Path = ":memory:", log_observations: bool = True):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.log_observations = log_observations

    def observation(self, slot_id, ts, coverage, visibility, quality, frame_id="", sku_pred=None, s1=None, s2=None,
                    crop_uri=None):
        if not self.log_observations:
            return
        with self.lock:
            self.conn.execute("INSERT INTO observation (slot_id, ts, frame_id, coverage, visibility, quality, sku_pred,"
                              " s1, s2, crop_uri) VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (slot_id, ts, frame_id, coverage, visibility, quality, sku_pred, s1, s2, crop_uri))

    def slot_state(self, slot_id, state, score, decision, last_clean_ts, ts, coverage, visibility):
        with self.lock:
            self.conn.execute("INSERT INTO slot_state VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(slot_id) DO UPDATE SET "
                              "state=excluded.state, score=excluded.score, decision=excluded.decision, "
                              "last_clean_ts=excluded.last_clean_ts, updated_ts=excluded.updated_ts, "
                              "coverage=excluded.coverage, visibility=excluded.visibility",
                              (slot_id, state, score, decision, last_clean_ts, ts, coverage, visibility))

    def state_event(self, slot_id, ts, from_state, to_state, score, decision, frame_ids=(), inputs=None,
                    engine_version="1.0"):
        with self.lock:
            self.conn.execute("INSERT INTO state_event (slot_id, ts, from_state, to_state, score, decision, frame_ids,"
                              " engine_version, inputs) VALUES (?,?,?,?,?,?,?,?,?)",
                              (slot_id, ts, from_state, to_state, score, decision, json.dumps(list(frame_ids)),
                               engine_version, json.dumps(inputs or {})))

    def upsert_task(self, t: dict):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO task VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                              (t["task_id"], t["slot_id"], t["kind"], t["priority"], t["status"], t["assignee_team"],
                               t["created_ts"], t["ack_ts"], t["done_ts"], t["verified_ts"], t["backroom_location"],
                               json.dumps(t.get("reason", {}), default=str)))

    def feedback(self, task_id, user_role, kind, note, ts):
        with self.lock:
            self.conn.execute("INSERT INTO feedback (task_id, user_role, kind, note, ts) VALUES (?,?,?,?,?)",
                              (task_id, user_role, kind, note, ts))

    def commit(self):
        with self.lock:
            self.conn.commit()

    def query(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def replay(self, slot_id: str, t0: float, t1: float) -> dict:
        """Everything needed to explain an alert: observations and state changes in a window."""
        return dict(
            observations=self.query("SELECT * FROM observation WHERE slot_id=? AND ts BETWEEN ? AND ? ORDER BY ts",
                                    (slot_id, t0, t1)),
            events=self.query("SELECT * FROM state_event WHERE slot_id=? AND ts BETWEEN ? AND ? ORDER BY ts",
                              (slot_id, t0, t1)))
