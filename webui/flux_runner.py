"""Background execution of the flux binary with real-time progress tracking."""

import fcntl
import json
import os
import re
import select
import subprocess
import threading
from pathlib import Path

import store

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "flux"


def find_models():
    """Scan the project root for directories containing model_index.json."""
    models = []
    for entry in ROOT.iterdir():
        if not entry.is_dir():
            continue
        index = entry / "model_index.json"
        if not index.exists():
            continue
        distilled = True
        try:
            with open(index) as f:
                info = json.load(f)
            distilled = info.get("is_distilled", False)
        except Exception:
            pass
        tag = "distilled, fast" if distilled else "base, high quality"
        models.append({
            "path": str(entry),
            "name": entry.name,
            "distilled": distilled,
            "label": f"{entry.name}  ({tag})",
        })
    return sorted(models, key=lambda m: m["name"])


_lock = threading.Lock()
_workers = {}


def submit(task_id):
    """Launch a background thread to run flux for the given task."""
    t = threading.Thread(target=_run, args=(task_id,), daemon=True)
    with _lock:
        _workers[task_id] = t
    t.start()


def _build_cmd(task):
    cmd = [
        str(BINARY),
        "-d", task["model_dir"],
        "-p", task["prompt"],
        "-o", task["output_path"],
        "-W", str(task["width"]),
        "-H", str(task["height"]),
        "-s", str(task["steps"]),
        "-S", str(task["seed"]),
        "-g", str(task["guidance"]),
    ]
    if task["schedule"] == "linear":
        cmd.append("--linear")
    elif task["schedule"] == "power":
        cmd += ["--power", "--power-alpha", str(task["power_alpha"])]
    if task["force_base"]:
        cmd.append("--base")
    if task["no_mmap"]:
        cmd.append("--no-mmap")
    for p in json.loads(task.get("input_paths", "[]")):
        cmd += ["-i", p]
    return cmd


def _run(task_id):
    task = store.get(task_id)
    if not task:
        return

    store.update_progress(task_id, 1, "Starting")

    try:
        proc = subprocess.Popen(
            _build_cmd(task),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        store.set_pid(task_id, proc.pid)

        mon = _Monitor(task_id)
        fd = proc.stderr.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        while proc.poll() is None:
            if select.select([fd], [], [], 0.5)[0]:
                try:
                    chunk = os.read(fd, 4096)
                    if chunk:
                        mon.feed(chunk.decode("utf-8", errors="replace"))
                except OSError:
                    break

        fcntl.fcntl(fd, fcntl.F_SETFL, flags)
        rest = proc.stderr.read()
        if rest:
            mon.feed(rest.decode("utf-8", errors="replace"))

        if proc.returncode == 0 and Path(task["output_path"]).exists():
            store.complete(task_id, mon.seed, mon.elapsed)
        else:
            store.fail(task_id, mon.error or f"Process exited with code {proc.returncode}")

    except Exception as e:
        store.fail(task_id, str(e))
    finally:
        with _lock:
            _workers.pop(task_id, None)


class _Monitor:
    """Parse flux stderr output to extract progress percentages.

    Progress mapping:
      0-5%    Loading model
      5-20%   Text encoding
      20-90%  Denoising steps (proportional)
      90-97%  VAE decode
      97-100% Saving
    """

    def __init__(self, task_id):
        self.task_id = task_id
        self.buf = ""
        self.seed = None
        self.elapsed = None
        self.error = None
        self._step = 0

    def feed(self, text):
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self._line(line.strip())
        if self.buf:
            self._check_step(self.buf)

    def _line(self, s):
        if not s:
            return

        m = re.match(r"Seed:\s*(\d+)", s)
        if m:
            self.seed = int(m.group(1))
            store.update_progress(self.task_id, 3, "Initialized")
            return

        if "Loading VAE" in s:
            store.update_progress(self.task_id, 5, "Loading model")
            return

        if "Loading" in s and "input" in s:
            store.update_progress(self.task_id, 8, "Loading references")
            return

        if "Encoding text" in s or "encoding text" in s:
            if "done" in s:
                store.update_progress(self.task_id, 20, "Text encoded")
            else:
                store.update_progress(self.task_id, 10, "Encoding text")
            return

        if self._check_step(s):
            return

        if ("Decoding" in s or "decoding" in s) and "Encoding" not in s:
            if "done" in s:
                store.update_progress(self.task_id, 95, "Decoded")
            else:
                store.update_progress(self.task_id, 90, "Decoding image")
            return

        if s.startswith("Saving"):
            store.update_progress(self.task_id, 97, "Saving")
            return

        m = re.search(r"Total generation time:\s*([\d.]+)", s)
        if m:
            self.elapsed = float(m.group(1))
            return

        if re.search(r"[Ee]rror:", s):
            self.error = s

    def _check_step(self, s):
        m = re.search(r"Step\s+(\d+)/(\d+)", s)
        if not m:
            return False
        step, total = int(m.group(1)), int(m.group(2))
        if step != self._step:
            self._step = step
            pct = 20 + (step / total) * 70
            store.update_progress(
                self.task_id, pct, f"Denoising step {step}/{total}"
            )
        return True
