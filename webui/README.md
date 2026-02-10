# FLUX.2 Web UI

A Streamlit interface for the flux image generation binary. Exposes all generation settings in a browser-based UI with persistent task tracking.

![Web UI](https://github.com/user-attachments/assets/placeholder)

## Setup

Requires **Python 3.10+** and the `flux` binary already compiled (see the [main README](../README.md) for build instructions).

```bash
cd webui
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. It auto-detects model directories in the project root.

## Features

- **All generation modes**: text-to-image and image-to-image with up to 16 reference images
- **Every CLI option exposed**: dimensions, steps, seed, guidance, timestep schedule (sigmoid / linear / power curve), force-base, no-mmap
- **Real-time progress**: live progress bar with phase labels (loading → encoding → denoising steps → decode → save)
- **Task persistence**: tasks survive page reloads and server restarts — backed by SQLite with WAL journaling
- **Cancel running tasks**: stop a generation mid-flight from the UI
- **Image gallery**: browse completed generations with prompt, dimensions, timing, and seed metadata
- **Save / Delete**: download images or clean up directly from the gallery
- **Format-agnostic uploads**: any image format (JPEG, PNG, WebP, TIFF, BMP) is converted to PNG before passing to flux

## Architecture

```
app.py          Streamlit UI — sidebar settings, prompt input, gallery, active tasks
store.py        SQLite persistence — task lifecycle (create → running → completed/failed)
flux_runner.py  Background execution — spawns flux in a thread, parses stderr for progress
```

The app launches `flux` as a subprocess per task. A monitor thread reads stderr with non-blocking I/O and maps the output phases to a 0–100% progress range:

| Range   | Phase              |
|---------|--------------------|
| 0–5%    | Loading model      |
| 5–20%   | Text encoding      |
| 20–90%  | Denoising steps    |
| 90–97%  | VAE decode         |
| 97–100% | Saving             |

On startup, `store.init()` checks for orphaned tasks (process no longer alive) and marks them as failed. The init call is cached via `@st.cache_resource` so Streamlit's auto-reloader doesn't re-trigger it.

## File Layout

```
webui/
├── app.py              Main application
├── store.py            Task database
├── flux_runner.py      Binary execution + progress parsing
├── requirements.txt    Python dependencies
├── .gitignore
├── data/               SQLite database (gitignored)
├── outputs/            Generated images (gitignored)
└── uploads/            Uploaded reference images (gitignored)
```
