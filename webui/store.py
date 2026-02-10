"""SQLite-backed task persistence for flux generation jobs."""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB = Path(__file__).parent / "data" / "tasks.db"


def _conn():
    DB.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(str(DB), timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=5000")
    return c


def init():
    c = _conn()
    c.execute("""CREATE TABLE IF NOT EXISTS tasks (
        id            TEXT PRIMARY KEY,
        mode          TEXT NOT NULL,
        prompt        TEXT NOT NULL,
        width         INTEGER NOT NULL,
        height        INTEGER NOT NULL,
        steps         INTEGER NOT NULL,
        seed          INTEGER NOT NULL,
        guidance      REAL NOT NULL,
        schedule      TEXT DEFAULT 'sigmoid',
        power_alpha   REAL DEFAULT 2.0,
        force_base    INTEGER DEFAULT 0,
        no_mmap       INTEGER DEFAULT 0,
        input_paths   TEXT DEFAULT '[]',
        model_dir     TEXT NOT NULL,
        output_path   TEXT,
        status        TEXT DEFAULT 'pending',
        progress      REAL DEFAULT 0,
        phase         TEXT DEFAULT '',
        created_at    TEXT NOT NULL,
        started_at    TEXT,
        completed_at  TEXT,
        elapsed       REAL,
        error         TEXT,
        actual_seed   INTEGER,
        pid           INTEGER
    )""")
    rows = c.execute(
        "SELECT id, pid FROM tasks WHERE status IN ('pending','running')"
    ).fetchall()
    for row in rows:
        pid = row[1]
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except (ProcessLookupError, PermissionError, OSError):
                pass
        if not alive:
            c.execute(
                "UPDATE tasks SET status='failed', "
                "error='Process terminated (server restart or crash)' "
                "WHERE id=?",
                (row[0],),
            )
    c.commit()
    c.close()


def create(mode, prompt, settings, model_dir, input_paths=None):
    task_id = uuid.uuid4().hex[:12]
    out_dir = (Path(__file__).parent / "outputs").resolve()
    out_dir.mkdir(exist_ok=True)
    output_path = str(out_dir / f"{task_id}.png")
    c = _conn()
    c.execute(
        "INSERT INTO tasks "
        "(id,mode,prompt,width,height,steps,seed,guidance,"
        "schedule,power_alpha,force_base,no_mmap,"
        "model_dir,input_paths,output_path,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (task_id, mode, prompt,
         settings["width"], settings["height"],
         settings["steps"], settings["seed"],
         settings["guidance"], settings["schedule"],
         settings["power_alpha"],
         int(settings.get("force_base", False)),
         int(settings.get("no_mmap", False)),
         model_dir, json.dumps(input_paths or []),
         output_path, datetime.now().isoformat()),
    )
    c.commit()
    c.close()
    return task_id


def update_progress(task_id, progress, phase=""):
    c = _conn()
    c.execute(
        "UPDATE tasks SET progress=?, phase=?, status='running', "
        "started_at=COALESCE(started_at,?) WHERE id=?",
        (progress, phase, datetime.now().isoformat(), task_id),
    )
    c.commit()
    c.close()


def complete(task_id, actual_seed=None, elapsed=None):
    c = _conn()
    c.execute(
        "UPDATE tasks SET status='completed', progress=100, phase='Done', "
        "completed_at=?, actual_seed=?, elapsed=? WHERE id=?",
        (datetime.now().isoformat(), actual_seed, elapsed, task_id),
    )
    c.commit()
    c.close()


def fail(task_id, error):
    c = _conn()
    c.execute(
        "UPDATE tasks SET status='failed', phase='', error=?, completed_at=? WHERE id=?",
        (error, datetime.now().isoformat(), task_id),
    )
    c.commit()
    c.close()


def set_pid(task_id, pid):
    c = _conn()
    c.execute("UPDATE tasks SET pid=? WHERE id=?", (pid, task_id))
    c.commit()
    c.close()


def get(task_id):
    c = _conn()
    row = c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    c.close()
    return dict(row) if row else None


def active():
    c = _conn()
    rows = c.execute(
        "SELECT * FROM tasks WHERE status IN ('pending','running') "
        "ORDER BY created_at"
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


def history(limit=50):
    c = _conn()
    rows = c.execute(
        "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


def delete(task_id):
    task = get(task_id)
    if task and task.get("output_path"):
        p = Path(task["output_path"])
        if p.exists():
            p.unlink()
    c = _conn()
    c.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    c.commit()
    c.close()
