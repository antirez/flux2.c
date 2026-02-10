"""FLUX.2 Web UI — Streamlit interface for the flux image generation binary."""

import hashlib
import os
import signal
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

import store
import flux_runner

st.set_page_config(page_title="FLUX.2 Generator", layout="wide")


@st.cache_resource
def _init_store():
    store.init()


_init_store()

# ── Prerequisites ─────────────────────────────────────────────────────

if not flux_runner.BINARY.exists():
    st.error(
        f"Binary not found at `{flux_runner.BINARY}`. "
        "Build it first with `make mps` (Apple Silicon) or `make blas` (CPU)."
    )
    st.stop()

models = flux_runner.find_models()
if not models:
    st.error(
        "No model directories found. "
        "Run `./download_model.sh` from the project root to download one."
    )
    st.stop()


# ── Helpers ───────────────────────────────────────────────────────────

def save_upload(f):
    """Save an uploaded file as PNG for guaranteed compatibility with flux."""
    d = (Path(__file__).parent / "uploads").resolve()
    d.mkdir(exist_ok=True)
    h = hashlib.sha256(f.getbuffer()).hexdigest()[:12]
    p = d / f"{h}.png"
    if not p.exists():
        img = Image.open(f)
        img.save(p, format="PNG")
    return str(p)


# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    model = st.selectbox(
        "Model",
        models,
        format_func=lambda m: m["label"],
        help="The FLUX.2 model to use. Distilled is fast (4 steps), "
             "base produces higher quality but is ~25x slower.",
    )
    is_distilled = model["distilled"]

    st.divider()

    mode = st.radio(
        "Mode",
        ["Text to Image", "Image to Image"],
        help="**Text to Image** generates from a prompt alone.\n\n"
             "**Image to Image** uses 1–16 reference images as visual context. "
             "The model attends to every reference during generation via "
             "in-context conditioning. One image works as a classic img2img; "
             "multiple images let the model combine their elements.",
    )

    st.divider()
    st.subheader("Dimensions")
    c1, c2 = st.columns(2)
    width = c1.number_input(
        "Width", 64, 1792, 256, 16,
        help="Output width in pixels. Must be a multiple of 16. Max 1792.",
    )
    height = c2.number_input(
        "Height", 64, 1792, 256, 16,
        help="Output height in pixels. Must be a multiple of 16. Max 1792.",
    )

    st.divider()
    st.subheader("Sampling")

    steps = st.number_input(
        "Steps", 1, 256, 4 if is_distilled else 50,
        help="Number of denoising iterations. "
             "Distilled: 4 is the trained sweet spot. "
             "Base: 50 for max quality, 10 for a quick preview.",
    )
    seed = st.number_input(
        "Seed", -1, 2**31, -1,
        help="Reproducibility seed. Use -1 for a random seed each time. "
             "The same seed with the same settings produces the same image.",
    )
    guidance = st.number_input(
        "Guidance", 0.0, 100.0, 1.0 if is_distilled else 4.0, 0.5,
        help="Classifier-Free Guidance strength. "
             "Higher values follow the prompt more strictly. "
             "Distilled default: 1.0, base default: 4.0.",
    )

    st.divider()
    st.subheader("Schedule")

    SCHEDULES = {
        "Shifted sigmoid (default)": "sigmoid",
        "Linear": "linear",
        "Power curve": "power",
    }
    schedule_label = st.radio(
        "Timestep distribution",
        list(SCHEDULES.keys()),
        help="Controls how denoising effort is distributed across steps.\n\n"
             "**Shifted sigmoid** concentrates work in the high-noise phase "
             "and matches the training distribution.\n\n"
             "**Linear** spreads steps uniformly. Can improve base model results "
             "at reduced step counts (e.g. 10 steps).\n\n"
             "**Power curve** is a tunable middle ground between sigmoid and linear.",
    )
    schedule_val = SCHEDULES[schedule_label]

    power_alpha = 2.0
    if schedule_val == "power":
        power_alpha = st.number_input(
            "Power exponent", 0.1, 10.0, 2.0, 0.1,
            help="Controls front-loading intensity. "
                 "1.0 equals linear, higher values concentrate more steps early.",
        )

    with st.expander("Advanced"):
        force_base = st.checkbox(
            "Force base model mode",
            help="Override autodetection. Enables Classifier-Free Guidance "
                 "with two transformer passes per step.",
        )
        no_mmap = st.checkbox(
            "Disable memory mapping",
            help="Load all weights into RAM upfront. "
                 "Uses ~16 GB but avoids per-step loading overhead on CPU/BLAS.",
        )


# ── Main: Generate ────────────────────────────────────────────────────

st.title("FLUX.2 Image Generator")
st.caption("Generate images with the FLUX.2 klein 4B model — pure C inference.")

prompt = st.text_area(
    "Prompt",
    placeholder="Describe the image you want to generate...",
    help="Be descriptive. For image-to-image, describe the desired output "
         "rather than giving instructions (e.g. 'oil painting of a sunset' "
         "instead of 'make it an oil painting').",
)

input_paths = []

if mode == "Image to Image":
    files = st.file_uploader(
        "Reference images (up to 16)",
        type=["png", "jpg", "jpeg", "ppm", "webp", "tiff", "bmp"],
        accept_multiple_files=True,
        help="Upload one or more reference images. Each is encoded separately and "
             "the model attends to all of them during generation. "
             "One image → classic img2img. Multiple → multi-reference combination.",
    )
    for f in (files or []):
        input_paths.append(save_upload(f))
    if input_paths:
        cols = st.columns(min(len(input_paths), 4))
        for i, p in enumerate(input_paths):
            cols[i % 4].image(p, width=150, caption=f"Ref {i + 1}")

ready = bool(prompt and prompt.strip())
if mode == "Image to Image" and not input_paths:
    ready = False

if st.button("Generate", type="primary", disabled=not ready, use_container_width=True):
    w = (width // 16) * 16
    h = (height // 16) * 16
    settings = {
        "width": w, "height": h, "steps": steps,
        "seed": seed, "guidance": guidance,
        "schedule": schedule_val, "power_alpha": power_alpha,
        "force_base": force_base, "no_mmap": no_mmap,
    }
    mode_key = "img2img" if input_paths else "txt2img"
    task_id = store.create(mode_key, prompt.strip(), settings, model["path"], input_paths)
    flux_runner.submit(task_id)
    st.rerun()


# ── Active Tasks ──────────────────────────────────────────────────────

@st.fragment(run_every=2)
def active_tasks_panel():
    tasks = store.active()

    prev_ids = st.session_state.get("_active_ids", set())
    curr_ids = {t["id"] for t in tasks}
    st.session_state["_active_ids"] = curr_ids

    if prev_ids and not prev_ids.issubset(curr_ids):
        st.rerun()

    if not tasks:
        return

    st.subheader("Active Tasks")
    for task in tasks:
        with st.container(border=True):
            left, right = st.columns([6, 1])
            with left:
                st.markdown(f"**{task['prompt'][:120]}**")
                pct = min(task["progress"] / 100.0, 1.0)
                st.progress(pct, text=task["phase"] or "Queued")
                created = datetime.fromisoformat(task["created_at"])
                elapsed = (datetime.now() - created).total_seconds()
                st.caption(
                    f"{task['width']}\u00d7{task['height']}  \u00b7  "
                    f"{task['mode']}  \u00b7  {elapsed:.0f}s elapsed"
                )
            with right:
                pid = task.get("pid")
                if pid and st.button("Cancel", key=f"stop_{task['id']}"):
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    store.fail(task["id"], "Cancelled by user")
                    st.rerun()

active_tasks_panel()


# ── Gallery ───────────────────────────────────────────────────────────

st.divider()
st.subheader("Generated Images")

all_tasks = store.history(100)
completed = [
    t for t in all_tasks
    if t["status"] == "completed"
    and t.get("output_path")
    and Path(t["output_path"]).exists()
]
failed = [t for t in all_tasks if t["status"] == "failed"]

if completed:
    cols = st.columns(4)
    for i, t in enumerate(completed):
        with cols[i % 4]:
            st.image(t["output_path"])
            st.caption(t["prompt"][:60])
            parts = [f"{t['width']}\u00d7{t['height']}"]
            if t.get("elapsed"):
                parts.append(f"{t['elapsed']:.1f}s")
            if t.get("actual_seed") is not None:
                parts.append(f"seed {t['actual_seed']}")
            st.caption(" \u00b7 ".join(parts))

            dl_col, rm_col = st.columns(2)
            with open(t["output_path"], "rb") as img_file:
                dl_col.download_button(
                    "Save",
                    img_file.read(),
                    file_name=Path(t["output_path"]).name,
                    mime="image/png",
                    key=f"dl_{t['id']}",
                )
            if rm_col.button("Delete", key=f"rm_{t['id']}"):
                store.delete(t["id"])
                st.rerun()

elif not all_tasks:
    st.info("No images generated yet. Enter a prompt and click Generate.")

if failed:
    with st.expander(f"Failed tasks ({len(failed)})"):
        for t in failed:
            col_err, col_dismiss = st.columns([5, 1])
            col_err.error(
                f"**{t['prompt'][:80]}** \u2014 {t.get('error', 'Unknown error')}"
            )
            if col_dismiss.button("Dismiss", key=f"dismiss_{t['id']}"):
                store.delete(t["id"])
                st.rerun()
