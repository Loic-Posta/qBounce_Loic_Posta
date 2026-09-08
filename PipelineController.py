"""
pipeline_controller.py — Orchestrator Dashboard for the CMOS Particle Pipeline
================================================================================
Chains the existing C++ detector and Python ML / labeling tools into one GUI:

        Clean -> Build -> Background Model (dark) -> Detection (signal)
            -> Labeling UI -> ML Classifier -> Backtesting (vs Golden Dataset)

Dependencies:
    pip install pillow pandas numpy matplotlib  # pandas + numpy are required for the
                                              # Labeling UI step's annotation sampling
                                              # (see sample_clusters_for_annotation) and
                                              # for the predict-mode quick-QA summary
                                              # chart. matplotlib powers the Results
                                              # tab's heatmap preview (colorbar + log-
                                              # scale colormap). If matplotlib isn't
                                              # installed, the Results tab falls back
                                              # to plain flat-image previews with no
                                              # colorbar.

Usage:
    python pipeline_controller.py

Layout assumptions — EDIT THE CONSTANTS BELOW IF YOUR LAYOUT DIFFERS:

    <project_root>/
      pipeline_controller.py   <- this file lives at the project root
      pipeline_state.json      <- created automatically, tracks step status
      code/
        src/                   main.cpp, *.h/.cpp, labeling_ui.py,
                    ml_classifier.py, DataExporter.cpp
        build/                 created by the 'Build C++ Pipeline' step
        Data/                  local cache of dark/ and signal/ frame folders
        detections.csv, background_model.yml, ...   <- pipeline outputs

Threading model:
    Every subprocess / long-running action runs on its own background thread.
    Output lines are pushed onto a thread-safe queue.Queue and drained on the
    Tkinter main thread via root.after(...). No Tkinter widget or variable is
    ever touched from a worker thread — all parameter values are read on the
    main thread (inside the button-click handlers) before a step is started.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import queue
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from PIL import Image, ImageDraw, ImageFont, ImageTk
import cv2

# On Windows, sys.stdout defaults to the console's legacy code page (cp1252 on a
# European install), and every print() below that carries an arrow, a degree
# sign or an accent then dies with UnicodeEncodeError. That is not cosmetic: the
# CSV is written before the plots, so the script exits 1 having produced the data
# and none of the figures. Force UTF-8 on the two streams, and fall back to
# replacing the offending character rather than crashing.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.colors import Normalize, LogNorm
    MPL_AVAILABLE = True
except ImportError:
    plt = None
    FigureCanvasTkAgg = None
    Normalize = None
    LogNorm = None
    MPL_AVAILABLE = False

try:
    import pandas as pd
except ImportError:  # pandas is only needed for the predict-mode summary chart
                     # and for building the annotation sample (see _step_label)
    pd = None

try:
    import numpy as np
except ImportError:  # numpy drives the random sampling in
                     # sample_clusters_for_annotation(); previously this was only
                     # imported as a side-effect of matplotlib, which meant it
                     # silently vanished on machines without matplotlib installed
    np = None

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — EDIT THESE TO MATCH YOUR PROJECT LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent

# Prefer the actual repository folder name. macOS/Windows often hide case
# mismatches, but Linux lab machines do not.
CODE_DIR  = PROJECT_ROOT / "Code"
if not CODE_DIR.exists():
    CODE_DIR = PROJECT_ROOT / "code"
SRC_DIR   = CODE_DIR / "src"

# Lets us do a plain `from ml_classifier import bundle_info` later (see
# _refresh_model_quality) regardless of the cwd this GUI happens to be
# launched from.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Assumes CMakeLists.txt sits at code/CMakeLists.txt (sibling of src/), so
# `cmake ..` run from code/build/ finds it. If yours lives inside code/src/,
# change this to SRC_DIR / "build" instead.
BUILD_DIR = CODE_DIR / "build"

# ⚠ Must match the add_executable(...) target name in your CMakeLists.txt
# (".exe" suffix required on Windows — the target Windows 10 offline machine)
DETECTOR_EXE_NAME = "detector.exe" if os.name == "nt" else "detector"


# ══════════════════════════════════════════════════════════════════════════════
#  OFFLINE-WINDOWS TOOLCHAIN BOOTSTRAP
#
#  On the lab machine there is no system-wide CMake and no system-wide OpenCV:
#  WindowsLauncher.py installs cmake as a *pip wheel* into Code\venv\Scripts and
#  extracts OpenCV into cpp_libs\. Neither directory is on PATH when this GUI
#  runs, because launching venv\Scripts\python.exe does NOT activate the venv.
#  That is what made the Build step die with
#      [ERROR] Could not start process: [WinError 2] ...
#  and it would next have stopped detector.exe from loading opencv_world*.dll.
#  So resolve both here, once, and prepend them to this process's PATH — every
#  subprocess started below inherits it.
# ══════════════════════════════════════════════════════════════════════════════

CPP_LIBS_DIR = PROJECT_ROOT / "cpp_libs"


def _find_opencv_dir() -> Optional[Path]:
    """Folder holding OpenCVConfig.cmake — i.e. what -DOpenCV_DIR expects.

    Same search order as WindowsLauncher.py, so the GUI can configure the build
    on its own even when the launcher was not used."""
    env = os.environ.get("OPENCV_DIR", "").strip()
    candidates = [Path(env) if env else None,
                  CPP_LIBS_DIR / "opencv" / "build"]
    if os.name == "nt":
        # Guarded: on POSIX this string is a *relative* path, so an unrelated
        # folder literally named 'C:\opencv\build' in the cwd would match.
        candidates.append(Path(r"C:\opencv\build"))
    for cand in candidates:
        if not cand or not cand.is_dir():
            continue
        # A prebuilt OpenCV ships two configs: one at the root of build/ that
        # derives OpenCV_RUNTIME from the compiler, and one per toolset under
        # build/x64/vcNN/lib. The root one sets OpenCV_FOUND to FALSE as soon as
        # the compiler is newer than any toolset in the pack -- Build Tools 2026
        # against a vc16-only pack fails exactly there, with "no binaries
        # compatible with your configuration". The versioned config skips that
        # check and links fine, so prefer it and keep the root as a fallback.
        for sub in sorted(cand.glob("x64/vc*/lib"), reverse=True):
            if (sub / "OpenCVConfig.cmake").is_file():
                return sub
        if (cand / "OpenCVConfig.cmake").is_file():
            return cand
    return None


def _prepend_to_path(*dirs: Optional[Path]) -> None:
    keep = [str(d) for d in dirs if d is not None and d.is_dir()]
    if keep:
        os.environ["PATH"] = os.pathsep.join(keep + [os.environ.get("PATH", "")])


OPENCV_DIR: Optional[Path] = _find_opencv_dir()

if os.name == "nt":
    # venv\Scripts -> cmake.exe / ninja.exe (installed as pip wheels, no admin)
    _prepend_to_path(Path(sys.executable).parent)
    # OpenCV runtime DLLs, e.g. cpp_libs\opencv\build\x64\vc16\bin
    if OPENCV_DIR is not None:
        _prepend_to_path(*sorted(OPENCV_DIR.glob("x64/vc*/bin")))


def cmake_executable() -> str:
    """Absolute path to cmake. Never trust it to be on the lab machine's PATH."""
    name = "cmake.exe" if os.name == "nt" else "cmake"
    in_venv = Path(sys.executable).parent / name
    if in_venv.is_file():
        return str(in_venv)
    return shutil.which("cmake") or name


def detector_exe_candidates() -> list[Path]:
    """Where detector.exe may legitimately land.

    MSVC's Visual Studio generator is *multi-config*: it writes
    build\\Release\\detector.exe. Single-config generators (Unix Makefiles on the
    Mac, Ninja) write build\\detector directly. Accept both."""
    return [BUILD_DIR / DETECTOR_EXE_NAME,
            BUILD_DIR / "Release" / DETECTOR_EXE_NAME,
            BUILD_DIR / "RelWithDebInfo" / DETECTOR_EXE_NAME,
            BUILD_DIR / "Debug" / DETECTOR_EXE_NAME]


def find_detector_exe() -> Optional[Path]:
    for exe in detector_exe_candidates():
        if exe.is_file():
            return exe
    return None

DATA_DIR = CODE_DIR / "Data"

MODEL_YML                  = CODE_DIR / "background_model.yml"   # rename to
                                                                   # "modele_bruit.yml"
                                                                   # here if that's
                                                                   # your local convention
DARK_PASS_CSV               = CODE_DIR / "dark_pass.csv"
DETECTIONS_CSV              = CODE_DIR / "detections.csv"
# Random, unbiased subset of DETECTIONS_CSV that the Labeling UI actually opens
# (see _step_label / sample_clusters_for_annotation) — this file accumulates
# true_label values as you annotate, so it IS your labeled set once you're done.
DETECTIONS_SAMPLE_CSV        = CODE_DIR / "detections_annotation_sample.csv"
DETECTIONS_LABELED_CSV      = CODE_DIR / "detections_labeled.csv"
DETECTIONS_CLASSIFIED_CSV   = CODE_DIR / "detections_classified.csv"

LABELING_UI_SCRIPT   = SRC_DIR / "labeling_ui.py"
ML_CLASSIFIER_SCRIPT = SRC_DIR / "ml_classifier.py"
STUDY_ENERGY_SCRIPT  = SRC_DIR / "study_energy.py"
BACKTEST_REPORT_SCRIPT = SRC_DIR / "backtest_report.py"
IDS_CAPTURE_SCRIPT   = SRC_DIR / "ids_capture.py"
LIVE_DETECTION_SCRIPT = SRC_DIR / "LiveDetectionEngine.py"
MODEL_BUNDLE = CODE_DIR / "model_bundle.joblib"
# Manually verified CSV (a human-checked true_label column) used only for
# Step 7 / Backtesting — never touched by the Labeling UI or by --train.
GOLDEN_DATASET_CSV = CODE_DIR / "golden_detections.csv"
BACKTEST_REPORT_TXT = CODE_DIR / "backtest_report.txt"
LIVE_WORK_DIR = CODE_DIR / "live_scratch"


def _sample_is_fully_labelled(csv_path: Path) -> bool:
    """True when every row of the annotation sample already carries a verdict.

    Cheap and defensive: any failure to read the file answers False, so the
    normal path is taken and nothing is hidden from the operator.
    """
    try:
        import csv as _csv
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        if not rows or "true_label" not in rows[0]:
            return False
        return all((r.get("true_label") or "").strip() for r in rows)
    except Exception:
        return False


STATE_FILE = PROJECT_ROOT / "pipeline_state.json"

# Same extension set main.cpp uses to collect input frames — keep in sync.
IMAGE_EXTENSIONS = {".bmp", ".tiff", ".tif", ".png"}

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# ── Results tab preview sources ─────────────────────────────────────────────
# Raw, lossless float32 .tiff data written by DataExporter (saveRawTiff) for
# each sensor heatmap — this is what the matplotlib preview plots directly,
# so the colorbar/colormap reflect real ADU/count values, not re-derived PNG
# pixel colors. The 3rd element controls whether the preview may use log scale.
# Keep sigma linear: its absolute spread is easier to read than a compressed
# log map.
RAW_HEATMAP_SOURCES: list[tuple[str, Path, bool]] = [
    # allow_log=True: a handful of exceptional pixels (hot pixels, damage)
    # otherwise crushes the colormap and hides all sensor structure; the
    # preview auto-switches to log only when max/min ratio warrants it.
    ("Background mean",             CODE_DIR / "data_analysed" / "dark"   / "background_mean.tiff",   True),
    ("Background fluctuation (σ)",  CODE_DIR / "data_analysed" / "dark"   / "background_sigma.tiff",  True),
    # Written by the detector server at the end of every live run — shows
    # WHERE captures land on the sensor (e.g. a partially covered region).
    ("Signal arrival (live)",       CODE_DIR / "live_signal_arrival.tiff", True),
]

# Plain, already-rendered images from the ML classifier step — these are
# categorical charts, not sensor heatmaps, so they get no colormap/colorbar.
FLAT_IMAGE_SOURCES: list[tuple[str, Path]] = [
    ("Prediction summary",   CODE_DIR / "prediction_summary.png"),
    ("Detected particles by type",  CODE_DIR / "particle_counts_by_type.png"),
    # Confusion matrix intentionally NOT previewed: it is a model-training
    # diagnostic, not operator information (still written to Code/ by the
    # Train step for whoever retrains the classifier).
    ("Energy spectrum (deposited charge)", CODE_DIR / "energy_histogram.png"),
]

# Variance ratio (max / smallest-positive-value) above which a heatmap auto-
# switches to log scale — mirrors the >= 100.0 threshold in DataExporter.cpp.
LOG_SCALE_VARIANCE_RATIO = 100.0


# ══════════════════════════════════════════════════════════════════════════════
#  STEP REGISTRY  (this is the plug point for future steps, e.g. Backtesting)
# ══════════════════════════════════════════════════════════════════════════════

STEP_ORDER = [
    "clean", "build", "bg_model",
    "detect_signal", "label", "classify", "backtest",
]

STEP_LABELS = {
    "clean":         "1 · Clean Workspace",
    "build":         "2 · Build C++ Pipeline  (cmake configure + build)",
    "bg_model":      "3 · Background Model  —  dark frames",
    "detect_signal": "4 · Cluster Detection  —  signal frames",
    "label":         "5 · Labeling UI",
    "classify":      "6 · ML Classifier  (+ auto QA preview)",
    "backtest":      "7 · Backtesting  (vs Golden Dataset)",
}

STATUS_COLOURS = {
    "pending":     "#adb5bd",
    "running":     "#0071e3",
    "success":     "#1a7f37",
    "failed":      "#b91c1c",
    "interrupted": "#9a6700",
}

# Fixed colors for the Live tab's multi-class graph — matches the default
# class taxonomy LiveDetectionEngine.py always reports (as 0 when a batch
# has none), so particle/cosmic_ray/artifact/hot_pixel always draw in the
# same color run over run. Any other label the classifier returns still
# gets plotted — it's just assigned the next color from the fallback
# palette on first appearance instead of a fixed one.
LIVE_CLASS_COLORS = {
    "particle":   "#0071e3",  # blue
    "cosmic_ray": "#b91c1c",  # red
    "artifact":   "#6e6e73",  # gray
    "hot_pixel":  "#f4a261",  # orange
}
LIVE_EXTRA_COLOR_PALETTE = ["#2a9d8f", "#6f42c1", "#9a6700", "#1a7f37"]


# ══════════════════════════════════════════════════════════════════════════════
#  SMALL PURE-FUNCTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def count_frames_in_folder(folder: Path) -> int:
    """Count image frames using the same extension set as main.cpp's loader,
    so the number matches what the detector will actually ingest."""
    folder = resolve_image_folder(folder)
    if folder is None:
        return 0
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_image_folder(folder: Path) -> Optional[Path]:
    """Return a folder that directly contains image frames.

    If the provided path already has image files, return it unchanged.
    Otherwise, search descendants in natural-sorted order and return the first
    folder that contains images. This lets the UI work when the user picks a
    parent dataset directory instead of the leaf image directory.
    """
    if not folder.is_dir():
        return None

    if any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.iterdir()):
        return folder

    child_dirs = sorted([p for p in folder.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    for child in child_dirs:
        if any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in child.iterdir()):
            return child

    for child in child_dirs:
        resolved = resolve_image_folder(child)
        if resolved is not None:
            return resolved

    return None


def sample_clusters_for_annotation(df: "pd.DataFrame", n_samples: int,
                                    baseline_min_size_px: int,
                                    seed: int) -> "pd.DataFrame":
    """
    Draw a uniform random subset of up to `n_samples` rows from `df` for a
    human to annotate in the Labeling UI.

    Filtering, deliberately minimal:
      Only rows with size_px > baseline_min_size_px survive (the "it's just
      1-2 dead/noise pixels" floor). No filtering on peak_snr, size, or any
      other quality metric is applied here — sorting or thresholding on any
      of those would bias the labeled (and therefore the trained) set toward
      whichever clusters look most "obvious", exactly the failure mode this
      function exists to avoid. See PipelineControllerApp._step_label for how
      the result is then reused (rather than re-drawn) across app restarts so
      an annotation session can be resumed without losing progress.

    Reproducibility:
      `seed` is fed to a dedicated NumPy Generator (not the global RNG), so
      the exact same `df` + `n_samples` + `baseline_min_size_px` + `seed`
      always yields the exact same subset — e.g. if the sample CSV is ever
      lost and has to be rebuilt from detections.csv.

    Efficient by construction: both the boolean mask and the final row
    selection are vectorised (pandas boolean indexing + numpy's Generator.
    choice), so this comfortably handles multi-million-row detections.csv
    files — no Python-level loop touches individual rows.
    """
    baseline = df[df["size_px"] > baseline_min_size_px].reset_index(drop=True)

    if n_samples >= len(baseline):
        return baseline

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(baseline), size=n_samples, replace=False)
    chosen_idx.sort()   # keep rows in their original (~chronological) order
    return baseline.iloc[chosen_idx].reset_index(drop=True)


def render_bar_chart_png(counts: dict, out_path: Path, title: str = "",
                          width: int = 640, height: int = 360) -> None:
    """Minimal PIL-only bar chart. Used as the "quick quality check" visual
    when ml_classifier.py runs in --predict mode (which produces a CSV but
    no PNG) — keeps this dashboard's only hard dependency at Pillow."""
    margin = 50
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if title:
        draw.text((margin, 12), title, fill="#000000", font=font)

    items = list(counts.items())
    if not items:
        draw.text((margin, height // 2), "No data", fill="#6e6e73", font=font)
        img.save(out_path)
        return

    max_val = max(v for _, v in items) or 1
    plot_h = height - 2 * margin
    plot_w = width - 2 * margin
    bar_w = plot_w / len(items)
    palette = ["#457b9d", "#e63946", "#2a9d8f", "#f4a261", "#6f42c1", "#adb5bd"]

    for i, (label, val) in enumerate(items):
        bar_h = int(plot_h * (val / max_val))
        x0 = int(margin + i * bar_w + bar_w * 0.15)
        x1 = int(margin + (i + 1) * bar_w - bar_w * 0.15)
        y1 = height - margin
        y0 = y1 - bar_h
        draw.rectangle([x0, y0, x1, y1], fill=palette[i % len(palette)])
        draw.text((x0, y1 + 4), str(label), fill="#000000", font=font)
        draw.text((x0, y0 - 14), str(val), fill="#000000", font=font)

    img.save(out_path)


def _chain(*callables: Optional[Callable[[], None]]) -> Callable[[], None]:
    """Combine zero-arg callables (skipping Nones) into a single callable
    that runs them in order. Always returns a valid callable."""
    fns = [c for c in callables if c is not None]

    def _run() -> None:
        for fn in fns:
            fn()

    return _run


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STATE  (pipeline_state.json — enables resume after a crash)
# ══════════════════════════════════════════════════════════════════════════════

class PipelineState:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"steps": {}, "last_updated": None}
        self.load()

    def load(self) -> None:
        if self.path.is_file():
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {"steps": {}, "last_updated": None}
        self.data.setdefault("steps", {})
        # A step that was left "running" belonged to a session that is no
        # longer alive (crash / force-quit) — flag it rather than pretend.
        for info in self.data["steps"].values():
            if info.get("status") == "running":
                info["status"] = "interrupted"

    def save(self) -> None:
        self.data["last_updated"] = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_status(self, step_id: str) -> str:
        return self.data["steps"].get(step_id, {}).get("status", "pending")

    def mark_running(self, step_id: str) -> None:
        self.data["steps"][step_id] = {
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.save()

    def mark_success(self, step_id: str) -> None:
        info = self.data["steps"].setdefault(step_id, {})
        info.update({"status": "success",
                      "ended_at": datetime.now().isoformat(timespec="seconds")})
        self.save()

    def mark_failed(self, step_id: str) -> None:
        info = self.data["steps"].setdefault(step_id, {})
        info.update({"status": "failed",
                      "ended_at": datetime.now().isoformat(timespec="seconds")})
        self.save()

    def mark_interrupted(self, step_id: str) -> None:
        info = self.data["steps"].setdefault(step_id, {})
        info.update({"status": "interrupted",
                      "ended_at": datetime.now().isoformat(timespec="seconds")})
        self.save()

    def reset(self) -> None:
        self.data = {"steps": {}, "last_updated": None}
        self.save()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class PipelineControllerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CMOS Particle Pipeline — Controller")
        self.root.geometry("1000x740")

        self.state = PipelineState(STATE_FILE)
        self.q: "queue.Queue" = queue.Queue()

        self.active_proc: Optional[subprocess.Popen] = None
        self.active_proc_step: Optional[str] = None
        self._busy_step: Optional[str] = None
        self._interrupt_requested: bool = False

        self.runners: dict[str, Callable[..., None]] = {
            "clean":         self._step_clean,
            "build":         self._step_build,
            "bg_model":      self._step_bg_model,
            "detect_signal": self._step_detect_signal,
            "label":         self._step_label,
            "classify":      self._step_classify,
            "backtest":      self._step_backtest,
        }

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True)
        self.tab_pipeline = ttk.Frame(self.nb)
        self.tab_params   = ttk.Frame(self.nb)
        self.tab_console  = ttk.Frame(self.nb)
        self.tab_results  = ttk.Frame(self.nb)
        self.tab_live     = ttk.Frame(self.nb)
        self.nb.add(self.tab_pipeline, text="Pipeline")
        self.nb.add(self.tab_params,   text="Parameters")
        self.nb.add(self.tab_live,     text="Live")
        self.nb.add(self.tab_console,  text="Console")
        self.nb.add(self.tab_results,  text="Results")

        self._build_params_tab()
        self._build_pipeline_tab()
        self._build_live_tab()
        self._build_console_tab()
        self._build_results_tab()

        for sid in STEP_ORDER:
            if self.state.get_status(sid) == "interrupted":
                self._append_console(
                    f"[{sid}] was left 'running' in a previous session — "
                    f"marked as interrupted. Re-run it when ready."
                )

        # classify_mode_var is shared between the Parameters tab radio buttons
        # and the Step 6 Train/Predict buttons — whichever sets it, Step 7's
        # availability and the model-quality readout should react.
        self.classify_mode_var.trace_add(
            "write", lambda *a: self._update_backtest_availability())
        self._update_backtest_availability()
        self._refresh_model_quality()

        self.root.after(100, self._poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ────────────────────────────────────────────────────────────────────
    #  UI: Pipeline tab
    # ────────────────────────────────────────────────────────────────────

    def _build_pipeline_tab(self) -> None:
        frame = self.tab_pipeline

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Button(top, text="▶  Run Remaining Steps",
                   command=self._run_remaining).pack(side="left")
        ttk.Button(top, text="⏭  Run Dark → Signal (4a+4b)",
                   command=lambda: self._step_bg_model(
                       on_success=self._step_detect_signal)
                   ).pack(side="left", padx=8)
        ttk.Button(top, text="⏹  Interrupt Running Step",
                   command=self._interrupt_active_step).pack(side="left")
        ttk.Button(top, text="↻  Reset Pipeline State",
                   command=self._reset_state).pack(side="left")

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill="both", expand=True, padx=10, pady=6)

        self.step_rows: dict[str, dict] = {}
        for sid in STEP_ORDER:
            row = ttk.Frame(list_frame, relief="groove", borderwidth=1)
            row.pack(fill="x", pady=3)

            status_var = tk.StringVar(value="●")
            status_lbl = tk.Label(row, textvariable=status_var,
                                   font=("TkDefaultFont", 14), width=2)
            status_lbl.pack(side="left", padx=(6, 4))

            ttk.Label(row, text=STEP_LABELS[sid], width=48,
                      anchor="w").pack(side="left", padx=4)

            note_var = tk.StringVar(value="")
            ttk.Label(row, textvariable=note_var,
                      foreground="#6e6e73").pack(side="left", padx=6)

            run_btn: Optional[ttk.Button] = None
            if sid == "classify":
                # Train vs Predict are different actions on different data
                # (a labeled CSV vs raw detections.csv) with different
                # downstream consequences (Predict disables Backtest — see
                # _update_backtest_availability) — so Step 6 gets two
                # explicit buttons instead of one generic "Run". Both keep
                # classify_mode_var (shared with the Parameters tab radio
                # buttons) in sync before running.
                ttk.Button(row, text="Predict", width=9,
                           command=lambda: self._run_classify_as("predict"),
                           ).pack(side="right", padx=(0, 6))
                ttk.Button(row, text="Train", width=9,
                           command=lambda: self._run_classify_as("train"),
                           ).pack(side="right")
            else:
                run_btn = ttk.Button(row, text="Run", command=self.runners[sid])
                run_btn.pack(side="right", padx=6)

            self.step_rows[sid] = {
                "status_var": status_var,
                "status_lbl": status_lbl,
                "note_var": note_var,
                "run_btn": run_btn,
            }
            self._refresh_step_row(sid)

            if sid == "bg_model":
                self._build_camera_damage_tester(list_frame)

            if sid == "classify":
                # Model-quality readout, right below the Step 6 row (Pipeline
                # tab, not Parameters) — pulled from ml_classifier.bundle_info().
                quality_row = ttk.Frame(list_frame)
                quality_row.pack(fill="x", padx=6, pady=(0, 4))
                self.model_quality_var = tk.StringVar(value="Model quality: —")
                ttk.Label(quality_row, textvariable=self.model_quality_var,
                          foreground="#0071e3").pack(side="left", padx=(28, 0))
                ttk.Button(quality_row, text="↻ Refresh",
                           command=self._refresh_model_quality,
                           width=10).pack(side="left", padx=8)

    def _build_camera_damage_tester(self, parent: tk.Widget) -> None:
        box = ttk.LabelFrame(parent, text="Camera Radiation Damage Tester")
        box.pack(fill="x", padx=28, pady=(0, 6))

        self.damage_before_model_var = tk.StringVar(value=str(MODEL_YML))
        self.damage_after_model_var = tk.StringVar(value="")
        # Models are binary "BGMD" files now (whatever the extension); legacy
        # OpenCV YAML models are still readable via the loader's fallback.
        yml_types = (("Background model", "*.yml *.yaml *.bgm"), ("All files", "*.*"))
        self._file_row(box, "Before background model:", self.damage_before_model_var, yml_types)
        self._file_row(box, "After background model:", self.damage_after_model_var, yml_types)
        ttk.Button(
            box,
            text="Run Degradation Analysis",
            command=self._run_degradation_analysis,
        ).pack(anchor="w", padx=8, pady=(2, 8))

    def _refresh_step_row(self, step_id: str) -> None:
        status = self.state.get_status(step_id)
        row = self.step_rows[step_id]
        row["status_lbl"].configure(fg=STATUS_COLOURS.get(status, "#adb5bd"))
        info = self.state.data["steps"].get(step_id, {})
        note = status
        ts = info.get("ended_at") or info.get("started_at")
        if ts:
            note += f"  ({ts})"
        row["note_var"].set(note)

    def _run_classify_as(self, mode: str) -> None:
        """Used by the Step 6 Train/Predict buttons — sets the shared mode
        variable (so the Parameters tab radio buttons reflect it too) then
        runs Step 6 immediately in that mode."""
        self.classify_mode_var.set(mode)
        self._step_classify()

    def _update_backtest_availability(self, *_args) -> None:
        """Step 7 (Backtest) only makes sense against labeled ground truth.
        Whenever Step 6 is set to Predict — whether via the Pipeline tab's
        Predict button or the Parameters tab radio — grey out Step 7 so it
        can't be launched against unlabeled production predictions."""
        backtest_row = self.step_rows.get("backtest")
        if not backtest_row or backtest_row.get("run_btn") is None:
            return
        disable = (self.classify_mode_var.get() == "predict")
        backtest_row["run_btn"].configure(state="disabled" if disable else "normal")
        if disable:
            backtest_row["note_var"].set(
                "disabled — Step 6 is set to Predict; switch it to Train to re-enable Backtest")
        else:
            self._refresh_step_row("backtest")

    def _refresh_model_quality(self) -> None:
        """Pull ml_classifier.bundle_info() for MODEL_BUNDLE and show it right
        under the Step 6 row in the Pipeline tab."""
        if not hasattr(self, "model_quality_var"):
            return
        if not MODEL_BUNDLE.is_file():
            self.model_quality_var.set("Model quality: No model trained yet")
            return
        try:
            from ml_classifier import bundle_info
            info = bundle_info(str(MODEL_BUNDLE))
        except Exception as exc:
            self.model_quality_var.set(f"Model quality: unavailable ({exc})")
            return

        parts = []
        cv_f1 = info.get("cv_f1_macro_mean")
        if cv_f1 is not None:
            parts.append(f"CV F1 (macro) {cv_f1:.3f} ({cv_f1 * 100:.1f}%)")
        else:
            parts.append("CV F1 unavailable (migrated legacy model)")
        n_samples = info.get("n_samples")
        if n_samples is not None:
            parts.append(f"{n_samples:,} training samples")
        trained_at = info.get("trained_at")
        if trained_at:
            parts.append(f"trained {trained_at}")
        self.model_quality_var.set("Model quality: " + "  ·  ".join(parts))

    def _load_background_full(self, path: Path) -> tuple:
        """Returns (mean, dead_mask, image_count). dead_mask is None if the
        file doesn't carry one (legacy YAML without that node).

        dead_mask matters for degradation analysis specifically because it is
        a MUCH more robust "is this pixel actually damaged" signal than a raw
        mean threshold: BackgroundModel only sets it when a pixel is
        saturated/zero on >=95% of an entire warmup run (DEAD_SAT_FRACTION /
        DEAD_ZERO_FRACTION), so a single noisy frame can't flip it — whereas
        `mean` is a fresh, from-scratch estimate every time bg_model runs
        (Health check's ③ never passes --resume), so with few 'after' dark
        frames it can be noisy enough to make a borderline pixel cross the
        brightness threshold either way by chance alone.
        """
        if np is None:
            raise RuntimeError("numpy is required for degradation analysis.")

        # Binary format written by BackgroundModel::save() — magic "BGMD",
        # then int32 version, int32 image_count, float n_sigma, int32 warmup,
        # int32 rows, int32 cols, then mean/M2/count (f32) and dead_mask (u8).
        with open(path, "rb") as fh:
            magic = fh.read(4)
            if magic == b"BGMD":
                import struct
                header = fh.read(24)  # version,image_count,n_sigma,warmup,rows,cols
                if len(header) != 24:
                    raise RuntimeError(f"Truncated background model header: {path}")
                version, img_count, _n_sigma, _warmup, rows, cols = \
                    struct.unpack("<iifiii", header)
                if rows <= 0 or cols <= 0:
                    raise RuntimeError(f"Corrupted background model header: {path}")
                n = rows * cols
                mean_buf = fh.read(n * 4)
                if len(mean_buf) != n * 4:
                    raise RuntimeError(f"Truncated background model 'mean': {path}")
                mean = np.frombuffer(mean_buf, dtype=np.float32).reshape(rows, cols).copy()
                fh.seek(n * 4 * 2, 1)  # skip M2 and count (not needed here)
                dead_buf = fh.read(n)
                dead_mask = (np.frombuffer(dead_buf, dtype=np.uint8).reshape(rows, cols) != 0
                            if len(dead_buf) == n else None)
                return mean, dead_mask, img_count

        # Legacy OpenCV YAML fallback (models saved before the binary format).
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required to read legacy OpenCV .yml background models."
            ) from exc
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"Could not open background model: {path}")
        try:
            mean = fs.getNode("mean").mat()
            dead_node = fs.getNode("dead_mask")
            dead = dead_node.mat() if not dead_node.empty() else None
            img_count_node = fs.getNode("image_count")
            img_count = int(img_count_node.real()) if not img_count_node.empty() else 0
        finally:
            fs.release()
        if mean is None or mean.size == 0:
            raise RuntimeError(f"Background model has no readable 'mean' matrix: {path}")
        dead_mask = (np.asarray(dead, dtype=np.uint8) != 0) if dead is not None else None
        return np.asarray(mean, dtype=np.float32), dead_mask, img_count

    def _run_degradation_analysis(self) -> None:
        before_path = Path(self.damage_before_model_var.get()).expanduser()
        after_path = Path(self.damage_after_model_var.get()).expanduser()
        if not before_path.is_file():
            messagebox.showerror(
                "Degradation Analysis",
                f"Before-experiment background model not found:\n{before_path}",
            )
            return
        if not after_path.is_file():
            messagebox.showerror(
                "Degradation Analysis",
                f"After-experiment background model not found:\n{after_path}",
            )
            return

        try:
            before, before_dead, before_n = self._load_background_full(before_path)
            after, after_dead, after_n = self._load_background_full(after_path)
        except Exception as exc:
            messagebox.showerror("Degradation Analysis", str(exc))
            return

        if before.shape != after.shape:
            messagebox.showerror(
                "Degradation Analysis",
                f"Model shapes do not match:\n"
                f"Before: {before.shape}\nAfter: {after.shape}",
            )
            return

        # 'after' is always built FRESH (③ never passes --resume — it must
        # start from zero exactly because it's measuring the sensor as it is
        # NOW, not inheriting old state). But a fresh mean is only as good as
        # its sample size: with far fewer 'after' dark frames than 'before',
        # borderline pixels can cross the brightness threshold either way by
        # sampling noise alone — that would look like damage appearing OR
        # disappearing that isn't real. Flag it instead of reporting a silent
        # false signal.
        sample_warning = None
        if before_n > 0 and after_n > 0:
            ratio = after_n / before_n
            if ratio < 0.5 or ratio > 2.0:
                sample_warning = (
                    f"⚠ Sample-size mismatch: 'before' used {before_n} frames, "
                    f"'after' used {after_n} ({ratio:.2f}x). The smaller sample's "
                    f"mean is noisier — treat single-pixel threshold flips near "
                    f"the boundary with caution; the dead-pixel-mask comparison "
                    f"below is more robust to this than the brightness threshold."
                )

        threshold = float(self.live_hot_pixel_brightness_threshold_var.get())
        total_pixels = int(before.size)
        before_mean = float(np.nanmean(before))
        after_mean = float(np.nanmean(after))
        mean_shift = after_mean - before_mean
        mean_shift_pct = (100.0 * mean_shift / before_mean) if before_mean else 0.0

        before_hot = before > threshold
        after_hot = after > threshold
        newly_hot = np.logical_and(~before_hot, after_hot)
        recovered_hot = np.logical_and(before_hot, ~after_hot)

        before_hot_count = int(before_hot.sum())
        after_hot_count = int(after_hot.sum())
        new_hot_count = int(newly_hot.sum())
        recovered_hot_count = int(recovered_hot.sum())
        before_hot_pct = 100.0 * before_hot_count / total_pixels
        after_hot_pct = 100.0 * after_hot_count / total_pixels
        new_hot_pct = 100.0 * new_hot_count / total_pixels

        # ── dead_mask comparison: the robust signal ───────────────────────
        # Each model's dead_mask is built independently from ITS OWN warmup
        # run (>=95% saturated/zero throughout — a whole-run statistic, not a
        # single-frame threshold), so it can't be fooled by one noisy sample
        # the way the brightness cut above can. A pixel truly cannot become
        # LESS damaged between an earlier and a later capture — physically,
        # this count can only credibly go up. If it goes down, that is itself
        # diagnostic: it means the 'before'/'after' comparison is measurement
        # noise, not evidence of an improving sensor — never interpret it as
        # "the camera healed".
        dead_mask_line = None
        if before_dead is not None and after_dead is not None and before_dead.shape == after_dead.shape:
            before_dead_count = int(before_dead.sum())
            after_dead_count = int(after_dead.sum())
            newly_dead = int(np.logical_and(~before_dead, after_dead).sum())
            seemingly_recovered = int(np.logical_and(before_dead, ~after_dead).sum())
            dead_mask_line = (
                f"Dead-pixel mask (robust signal, ≥95% saturated/zero over an entire "
                f"warmup run): {before_dead_count:,} -> {after_dead_count:,} "
                f"({newly_dead:,} newly dead)"
            )
            if seemingly_recovered > 0:
                dead_mask_line += (
                    f"  ⚠ {seemingly_recovered:,} pixel(s) LEFT the dead mask — "
                    f"physically impossible for real damage; this many is your "
                    f"noise floor for this comparison, not evidence of repair."
                )

        if new_hot_pct > 0:
            runs_to_one_pct = max(0.0, (1.0 - after_hot_pct) / new_hot_pct)
            runs_to_five_pct = max(0.0, (5.0 - after_hot_pct) / new_hot_pct)
            lifetime_note = (
                f"At this observed damage rate, reaching 1% hot pixels would take "
                f"~{runs_to_one_pct:.1f} similar runs; 5% would take "
                f"~{runs_to_five_pct:.1f} similar runs."
            )
        else:
            lifetime_note = (
                "No newly born hot pixels were detected at this threshold; "
                "lifetime extrapolation is stable for this run."
            )

        severity = "LOW"
        if new_hot_pct >= 0.1 or mean_shift_pct >= 10.0:
            severity = "MODERATE"
        if new_hot_pct >= 1.0 or mean_shift_pct >= 25.0:
            severity = "HIGH"

        self._append_console("[damage] Camera Radiation Damage Analysis")
        self._append_console(f"[damage] Before model: {before_path} ({before_n} frames)")
        self._append_console(f"[damage] After model : {after_path} ({after_n} frames)")
        if sample_warning:
            self._append_console(f"[damage] {sample_warning}")
        self._append_console(f"[damage] Sensor pixels: {total_pixels:,}")
        self._append_console(
            f"[damage] Mean dark current: {before_mean:.3f} -> {after_mean:.3f} ADU "
            f"(shift {mean_shift:+.3f} ADU, {mean_shift_pct:+.2f}%)"
        )
        if dead_mask_line:
            self._append_console(f"[damage] {dead_mask_line}")
        self._append_console(
            f"[damage] Hot-pixel threshold: > {threshold:.3f} ADU"
        )
        self._append_console(
            f"[damage] Hot pixels before: {before_hot_count:,} / {total_pixels:,} "
            f"({before_hot_pct:.4f}%)"
        )
        self._append_console(
            f"[damage] Hot pixels after : {after_hot_count:,} / {total_pixels:,} "
            f"({after_hot_pct:.4f}%)"
        )
        self._append_console(
            f"[damage] Newly born hot pixels: {new_hot_count:,} "
            f"({new_hot_pct:.4f}% of sensor)"
        )
        self._append_console(
            f"[damage] Recovered/no-longer-hot pixels: {recovered_hot_count:,}"
        )
        self._append_console(f"[damage] Degradation severity: {severity}")
        self._append_console(f"[damage] Lifetime estimate: {lifetime_note}")
        messagebox.showinfo(
            "Degradation Analysis",
            f"Analysis complete.\n\nNew hot pixels: {new_hot_count:,} "
            f"({new_hot_pct:.4f}%)\nMean shift: {mean_shift:+.3f} ADU "
            f"({mean_shift_pct:+.2f}%)\nSeverity: {severity}",
        )

    def _run_remaining(self) -> None:
        pending = [sid for sid in STEP_ORDER if self.state.get_status(sid) != "success"]
        if self.classify_mode_var.get() == "predict" and "backtest" in pending:
            # Same rule as the greyed-out Step 7 button: skip it automatically
            # rather than let the chain stop on it.
            pending = [sid for sid in pending if sid != "backtest"]
        if not pending:
            messagebox.showinfo("Pipeline", "All steps already completed successfully.")
            return
        self._run_chain(pending)

    def _run_chain(self, remaining_ids: list[str]) -> None:
        if not remaining_ids:
            self._append_console("[pipeline] ✓ Run Remaining Steps complete.")
            return
        sid, rest = remaining_ids[0], remaining_ids[1:]
        self.runners[sid](on_success=lambda: self._run_chain(rest))

    def _reset_state(self) -> None:
        if not messagebox.askyesno(
            "Reset Pipeline State",
            "This clears pipeline_state.json (step history) but does NOT "
            "delete any files. Continue?",
        ):
            return
        self.state.reset()
        for sid in STEP_ORDER:
            self._refresh_step_row(sid)
        self._update_backtest_availability()
        self._append_console("[pipeline] State reset.")

    # ────────────────────────────────────────────────────────────────────
    #  UI: Parameters tab
    # ────────────────────────────────────────────────────────────────────

    def _build_params_tab(self) -> None:
        pad = {"padx": 10, "pady": 6}

        # Scrollable container — same pattern as the Live/Results tabs. The
        # parameters list has outgrown a single screen (acquisition presets,
        # cluster filters, auto-tune…), and without a scrollbar everything
        # below the fold was simply unreachable.
        outer = ttk.Frame(self.tab_params)
        outer.pack(fill="both", expand=True)
        self.params_canvas = tk.Canvas(outer, highlightthickness=0)
        params_vscroll = ttk.Scrollbar(outer, orient="vertical",
                                       command=self.params_canvas.yview)
        self.params_canvas.configure(yscrollcommand=params_vscroll.set)
        self.params_canvas.pack(side="left", fill="both", expand=True)
        params_vscroll.pack(side="right", fill="y")

        frame = ttk.Frame(self.params_canvas)
        self._params_inner_window = self.params_canvas.create_window(
            (0, 0), window=frame, anchor="nw")

        def _sync_scrollregion(_event=None) -> None:
            self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))

        def _sync_inner_width(event) -> None:
            self.params_canvas.itemconfigure(self._params_inner_window, width=event.width)

        frame.bind("<Configure>", _sync_scrollregion)
        self.params_canvas.bind("<Configure>", _sync_inner_width)

        def _on_mousewheel(event) -> None:
            if event.num == 4:
                self.params_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.params_canvas.yview_scroll(1, "units")
            else:
                self.params_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        def _bind_mousewheel(_event=None) -> None:
            self.params_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.params_canvas.bind_all("<Button-4>", _on_mousewheel)
            self.params_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mousewheel(_event=None) -> None:
            self.params_canvas.unbind_all("<MouseWheel>")
            self.params_canvas.unbind_all("<Button-4>")
            self.params_canvas.unbind_all("<Button-5>")

        self.params_canvas.bind("<Enter>", _bind_mousewheel)
        self.params_canvas.bind("<Leave>", _unbind_mousewheel)

        folders_box = ttk.LabelFrame(frame, text="Data Folders")
        folders_box.pack(fill="x", **pad)
        self.dark_folder_var = tk.StringVar(value=str(DATA_DIR / "dark"))
        self.signal_folder_var = tk.StringVar(value=str(DATA_DIR / "signal"))
        self._folder_row(folders_box, "Dark frames folder:", self.dark_folder_var)
        self._folder_row(folders_box, "Signal data folder:", self.signal_folder_var)

        ids_box = ttk.LabelFrame(frame, text="IDS Camera Capture  (optional, lab machine only)")
        ids_box.pack(fill="x", **pad)
        self.ids_device_index_var = tk.IntVar(value=0)
        self.ids_frame_count_var = tk.IntVar(value=100)
        self.ids_exposure_us_var = tk.StringVar(value="")
        self.ids_gain_db_var = tk.StringVar(value="")
        self.ids_format_var = tk.StringVar(value="png")
        self.ids_clear_output_var = tk.BooleanVar(value=False)
        self._spin_row(ids_box, "Camera index:", self.ids_device_index_var, 0, 16, 1)
        self._spin_row(ids_box, "Frames to capture:", self.ids_frame_count_var, 1, 100000, 1)

        # ── Acquisition presets ──────────────────────────────────────────
        # "energy": reduced exposure/gain so heavy-ion tracks stay below the
        # 255 saturation level — required for the deposited-energy spectrum
        # (see study_energy.py: tune until its 'Saturated' fraction ≈ 0%).
        # "sensitivity": full exposure/gain (the historical behaviour) for
        # when faint events (gammas) matter more than energy resolution.
        self.acq_mode_var = tk.StringVar(value="energy")
        self.ids_energy_exposure_us_var = tk.StringVar(value="50000")
        self.ids_energy_gain_db_var = tk.StringVar(value="0")

        mode_row = ttk.Frame(ids_box)
        mode_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(mode_row, text="Acquisition mode:", width=22, anchor="w").pack(side="left")
        ttk.Radiobutton(mode_row, text="Energy resolution (reduced gain)",
                        value="energy", variable=self.acq_mode_var).pack(side="left")
        ttk.Radiobutton(mode_row, text="High sensitivity (high gain)",
                        value="sensitivity", variable=self.acq_mode_var).pack(side="left", padx=12)

        self._entry_row(ids_box, "Exposure us (energy):", self.ids_energy_exposure_us_var,
                        "exposure sets duty cycle; gain (below) sets saturation")
        # Not labelled "dB": on the U3-380xACP-M the Gain node is a
        # multiplicative factor whose minimum is 1.0, so "0" is not "no gain"
        # -- the camera rejects it outright. Anything below the minimum is
        # clamped up to it (ids_capture._set_value says so in the console),
        # which is the lowest gain available and exactly what energy mode wants.
        self._entry_row(ids_box, "Gain (energy):", self.ids_energy_gain_db_var,
                        "lowest gain; below the camera's minimum is clamped up to it")

        # First-proposal search for the energy-mode exposure: the experimenter
        # only chooses how much saturation is acceptable; the camera search
        # does the rest (see ids_capture.py auto_tune()).
        self.autotune_target_sat_var = tk.DoubleVar(value=50.0)
        tune_row = ttk.Frame(ids_box)
        tune_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(tune_row, text="Target saturated px/frame:", width=22,
                  anchor="w").pack(side="left")
        ttk.Spinbox(tune_row, from_=0.0, to=100000.0, increment=10.0,
                    textvariable=self.autotune_target_sat_var, width=8).pack(side="left")
        ttk.Button(tune_row, text="Auto-tune gain (beam ON)",
                   command=self._run_exposure_autotune).pack(side="left", padx=10)
        self._entry_row(ids_box, "Exposure us (sensitivity):", self.ids_exposure_us_var,
                        "blank = keep Cockpit/camera setting")
        self._entry_row(ids_box, "Gain (sensitivity):", self.ids_gain_db_var,
                        "blank = keep Cockpit/camera setting")
        fmt_row = ttk.Frame(ids_box)
        fmt_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(fmt_row, text="Save format:", width=22, anchor="w").pack(side="left")
        ttk.Combobox(fmt_row, textvariable=self.ids_format_var,
                     values=("png", "tiff", "bmp"), state="readonly", width=8).pack(side="left")
        ttk.Checkbutton(ids_box, text="Clear existing images in target folder before capture",
                        variable=self.ids_clear_output_var).pack(anchor="w", padx=8, pady=(0, 4))
        ids_buttons = ttk.Frame(ids_box)
        ids_buttons.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Button(ids_buttons, text="List IDS Cameras",
                   command=self._step_ids_list_devices).pack(side="left")
        ttk.Button(ids_buttons, text="Capture Dark Folder",
                   command=lambda: self._step_ids_capture("dark")).pack(side="left", padx=8)
        ttk.Button(ids_buttons, text="Capture Signal Folder",
                   command=lambda: self._step_ids_capture("signal")).pack(side="left")

        det_box = ttk.LabelFrame(frame, text="Detector Parameters (main.cpp)")
        det_box.pack(fill="x", **pad)
        self.nsigma_var = tk.DoubleVar(value=5.0)
        self._spin_row(det_box, "n_sigma (threshold):", self.nsigma_var, 0.5, 20.0, 0.5)

        self.auto_warmup_var = tk.BooleanVar(value=True)
        self.warmup_var = tk.IntVar(value=18)
        warm_row = ttk.Frame(det_box)
        warm_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(warm_row, text="warmup_frames:", width=22,
                  anchor="w").pack(side="left")
        ttk.Spinbox(warm_row, from_=1, to=5000,
                    textvariable=self.warmup_var, width=8).pack(side="left")
        ttk.Checkbutton(warm_row, text="Auto (count dark folder frames)",
                         variable=self.auto_warmup_var).pack(side="left", padx=10)

        # Morphological-closing radius (px) applied to the event mask before
        # connected-components labeling — bridges small gaps in split tracks
        # (e.g. a muon track broken by a couple of under-threshold pixels).
        # 0 = disabled, matching the detector's own default.
        # Cluster quality floors — a THROUGHPUT GUARDRAIL only (the ML
        # classifier does the physics selection), so these are set loose
        # enough never to touch anything the classifier might call signal.
        #   • Min size (px): kills the flood of 1-2 px noise cheaply, at the
        #     connected-components stage. 3 leaves margin below the smallest
        #     real lithium track (5 px in the labelled data).
        #   • Min peak SNR: gain-INVARIANT floor (signal/noise in σ units
        #     cancels the gain, unlike absolute ADU) — so it stays valid when
        #     you lower the gain to de-saturate for the energy spectrum,
        #     instead of silently eating dimmed tracks. Requires rebuilding
        #     the background model at the operating gain.
        #   • Min charge (ADU): legacy absolute floor, default OFF because it
        #     is NOT gain-invariant. Use only at a fixed operating point.
        # Set all to 0 to record everything (e.g. to label gammas later).
        self.min_cluster_size_var = tk.IntVar(value=3)
        self._spin_row(det_box, "Min cluster size (px):", self.min_cluster_size_var, 0, 100, 1)
        self.min_peak_snr_var = tk.DoubleVar(value=8.0)
        self._spin_row(det_box, "Min peak SNR (gain-invariant):", self.min_peak_snr_var, 0.0, 100.0, 1.0)
        self.min_cluster_charge_var = tk.DoubleVar(value=0.0)
        self._spin_row(det_box, "Min cluster charge (ADU, legacy):", self.min_cluster_charge_var, 0.0, 100000.0, 50.0)

        self.cluster_gap_var = tk.IntVar(value=0)
        self._spin_row(det_box, "Cluster gap tolerance (px):", self.cluster_gap_var, 0, 20, 1)
        # Default 1: include the sub-threshold halo pixels around each track in
        # its footprint — recovers halo charge for the energy proxy and gives
        # arrival heatmaps their track outline. NOTE: changing this changes the
        # total_charge scale, so recalibrate the energy axis after changing it.
        self.cluster_grow_var = tk.IntVar(value=1)
        self._spin_row(det_box, "Cluster grow radius (px):", self.cluster_grow_var, 0, 10, 1)
        self.cluster_grow_min_snr_var = tk.DoubleVar(value=1.0)
        self._spin_row(det_box, "Grow min SNR:", self.cluster_grow_min_snr_var, 0.0, 20.0, 0.5)
        ttk.Label(
            det_box,
            text="Bridges small gaps in broken/split particle tracks via morphological "
                 "closing before clustering. Grow radius then adds weaker neighbouring "
                 "pixels to each cluster footprint for size/charge/bbox measurement "
                 "(0 = off — identical to previous behaviour).",
            foreground="#6e6e73", wraplength=560, justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 6))

        sample_box = ttk.LabelFrame(frame, text="Annotation Sampling  (Labeling UI)")
        sample_box.pack(fill="x", **pad)

        self.num_clusters_var = tk.IntVar(value=200)
        self._spin_row(sample_box, "Clusters to annotate:", self.num_clusters_var, 10, 5000, 10)

        self.baseline_min_size_var = tk.IntVar(value=2)
        self._spin_row(sample_box, "Noise floor \u2014 exclude size_px \u2264:",
                        self.baseline_min_size_var, 0, 50, 1)

        self.sample_seed_var = tk.IntVar(value=42)
        self._spin_row(sample_box, "Random seed:", self.sample_seed_var, 0, 999_999, 1)

        self.force_resample_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            sample_box,
            text="Force a new sample on next Run  (discards the existing "
                 "sample file and any labels already made on it)",
            variable=self.force_resample_var,
        ).pack(anchor="w", padx=8, pady=(0, 4))

        ttk.Label(
            sample_box,
            text="The Labeling UI opens a random subset drawn from ALL detections "
                 "above the noise floor (no size/SNR filtering beyond that) \u2014 "
                 "keeps the labeled set representative instead of biased toward "
                 "the most obvious clusters. The sample is generated once and "
                 "reused on every restart, so an annotation session survives a "
                 "crash or app close without losing progress or reshuffling.",
            foreground="#6e6e73", wraplength=560, justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 6))

        clf_box = ttk.LabelFrame(frame, text="ML Classifier (ml_classifier.py)")
        clf_box.pack(fill="x", **pad)
        self.classify_mode_var = tk.StringVar(value="train")
        ttk.Radiobutton(clf_box, text="Train  (needs a labeled CSV)", value="train",
                         variable=self.classify_mode_var).pack(anchor="w", padx=8, pady=2)
        ttk.Radiobutton(clf_box, text="Predict  (score detections.csv)", value="predict",
                         variable=self.classify_mode_var).pack(anchor="w", padx=8, pady=2)

        live_hotpix_box = ttk.LabelFrame(frame, text="Live Detection — Dynamic Hot/Dead Pixel Masking")
        live_hotpix_box.pack(fill="x", **pad)
        # A pixel that reads above `brightness_threshold` for `consecutive_frames`
        # frames in a row (while the sensor should be in a quiet background
        # state) is assumed stuck/hot rather than a genuine particle event —
        # a real event is transient and won't re-fire the same pixel every
        # single frame. Both values are read fresh by _step_live_detection()
        # each time "Start Live" is pressed and forwarded to
        # LiveDetectionEngine.py as --hot-pixel-consecutive-frames /
        # --hot-pixel-brightness-threshold.
        self.live_hot_pixel_consecutive_frames_var = tk.IntVar(value=5)
        self._spin_row(live_hotpix_box, "Consecutive frames (N):",
                        self.live_hot_pixel_consecutive_frames_var, 1, 500, 1)
        self.live_hot_pixel_brightness_threshold_var = tk.DoubleVar(value=50.0)
        self._spin_row(live_hotpix_box, "Brightness threshold (ADU):",
                        self.live_hot_pixel_brightness_threshold_var, 0.0, 65535.0, 1.0)
        ttk.Label(
            live_hotpix_box,
            text="A pixel that stays above the brightness threshold for N consecutive "
                 "frames is flagged Dead/Hot and added to the runtime mask — masked "
                 "pixels are excluded from the live heatmap's color scale (shown as "
                 "black) so rare, high-intensity particle events keep the full plasma "
                 "dynamic range instead of being washed out by a stuck pixel.",
            foreground="#6e6e73", wraplength=560, justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 6))

        backtest_box = ttk.LabelFrame(frame, text="Backtesting (backtest_report.py)")
        backtest_box.pack(fill="x", **pad)
        self.golden_dataset_var = tk.StringVar(value=str(GOLDEN_DATASET_CSV))
        self._file_row(backtest_box, "Golden dataset (true_label):", self.golden_dataset_var)
        ttk.Label(
            backtest_box,
            text="A manually verified CSV — a true_label column you're 100% sure of — "
                 "used only for Step 7. Backtest runs the current model_bundle.joblib "
                 "directly on this file's features and scores predicted_label against "
                 "true_label (accuracy, per-class precision/recall/F1, confusion matrix).",
            foreground="#6e6e73", wraplength=560, justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 6))

        # ── Energy analysis (office use, NOT the live lab loop) ──────────
        # The deposited-energy spectrum needs statistics: a normal experiment
        # (100–500 frames) has too few heavy tracks for meaningful peaks, so
        # ⑤'s end-of-experiment report does NOT run it. Point this field at a
        # LARGE classified dataset (the rare 15k-frame runs) and press the
        # button — done back at the office, on accumulated data.
        energy_box = ttk.LabelFrame(frame, text="Energy analysis (office — large datasets)")
        energy_box.pack(fill="x", **pad)
        ttk.Button(energy_box, text="⚡ Energy Spectrum (runs on the file below)",
                   command=lambda: self._run_energy_spectrum(
                       Path(self.energy_dataset_var.get()).expanduser(),
                       source_label="energy dataset")).pack(anchor="w", padx=8, pady=(6, 2))
        self.energy_dataset_var = tk.StringVar(value=str(DETECTIONS_CLASSIFIED_CSV))
        self._file_row(energy_box, "Energy dataset (classified CSV):", self.energy_dataset_var)
        ttk.Label(energy_box,
                  text="Needs a classified CSV with enough statistics (≥ a few thousand "
                       "heavy tracks — e.g. a 15 000-frame run). Result appears in Results.",
                  foreground="#6e6e73", wraplength=560, justify="left",
                  ).pack(anchor="w", padx=8, pady=(0, 6))

        export_box = ttk.LabelFrame(frame, text="Data Export (DataExporter)")
        export_box.pack(fill="x", **pad)
        self.dark_export_dir_var = tk.StringVar(value=str(CODE_DIR / "data_analysed" / "dark"))
        self.signal_export_dir_var = tk.StringVar(value=str(CODE_DIR / "data_analysed" / "signal"))
        self._folder_row(export_box, "Dark analyzed folder:", self.dark_export_dir_var)
        self._folder_row(export_box, "Signal analyzed folder:", self.signal_export_dir_var)

        paths_box = ttk.LabelFrame(
            frame, text="Resolved Paths — edit the constants at the top of the file to change")
        paths_box.pack(fill="both", expand=True, **pad)
        for label, value in [
            ("Project root",   PROJECT_ROOT),
            ("Build dir",      BUILD_DIR),
            ("Detector exe",   find_detector_exe()
                               or f"{BUILD_DIR / DETECTOR_EXE_NAME}   (not built yet)"),
            ("CMake",          cmake_executable()),
            ("OpenCV C++",     OPENCV_DIR or "not found — Build will fail"),
            ("Model file",     MODEL_YML),
            ("Detections CSV", DETECTIONS_CSV),
            ("Annotation sample", DETECTIONS_SAMPLE_CSV),
            ("Model bundle",   MODEL_BUNDLE),
            ("Golden dataset", GOLDEN_DATASET_CSV),
            ("Backtest report", BACKTEST_REPORT_TXT),
            ("Pipeline state", STATE_FILE),
        ]:
            row = ttk.Frame(paths_box)
            row.pack(fill="x", padx=8, pady=1)
            ttk.Label(row, text=f"{label}:", width=16, anchor="w",
                      foreground="#6e6e73").pack(side="left")
            ttk.Label(row, text=str(value), anchor="w").pack(side="left")

    def _folder_row(self, parent: tk.Widget, label: str, var: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=label, width=22, anchor="w").pack(side="left")
        ttk.Entry(row, textvariable=var, width=46).pack(side="left", padx=4)
        ttk.Button(row, text="Browse…",
                   command=lambda: self._browse_folder(var)).pack(side="left")

    def _browse_folder(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(initialdir=var.get() or str(PROJECT_ROOT))
        if path:
            var.set(path)

    def _file_row(self, parent: tk.Widget, label: str, var: tk.StringVar,
                   filetypes: tuple = (("CSV files", "*.csv"), ("All files", "*.*"))) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=label, width=22, anchor="w").pack(side="left")
        ttk.Entry(row, textvariable=var, width=46).pack(side="left", padx=4)
        ttk.Button(row, text="Browse…",
                   command=lambda: self._browse_file(var, filetypes)).pack(side="left")

    def _browse_file(self, var: tk.StringVar, filetypes: tuple) -> None:
        current = Path(var.get()) if var.get() else PROJECT_ROOT
        initialdir = str(current.parent) if current.suffix else str(current)
        path = filedialog.askopenfilename(
            initialdir=initialdir or str(PROJECT_ROOT), filetypes=filetypes)
        if path:
            var.set(path)

    def _spin_row(self, parent: tk.Widget, label: str, var: tk.Variable,
                  frm: float, to: float, inc: float) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=label, width=22, anchor="w").pack(side="left")
        ttk.Spinbox(row, from_=frm, to=to, increment=inc,
                    textvariable=var, width=8).pack(side="left")

    def _entry_row(self, parent: tk.Widget, label: str, var: tk.StringVar,
                   hint: str = "") -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=label, width=22, anchor="w").pack(side="left")
        ttk.Entry(row, textvariable=var, width=12).pack(side="left")
        if hint:
            ttk.Label(row, text=hint, foreground="#6e6e73").pack(side="left", padx=8)

    # ────────────────────────────────────────────────────────────────────
    #  UI: Live tab
    # ────────────────────────────────────────────────────────────────────

    def _build_live_tab(self) -> None:
        pad = {"padx": 10, "pady": 6}

        # Scrollable container — same pattern as the Results tab. Without
        # this, the matplotlib graph (which wants a decent chunk of
        # vertical space) pushes the "Live Statistics" box below the
        # visible area on anything but a tall window, with no way to
        # reach it.
        outer = ttk.Frame(self.tab_live)
        outer.pack(fill="both", expand=True)

        self.live_canvas = tk.Canvas(outer, highlightthickness=0)
        live_vscroll = ttk.Scrollbar(outer, orient="vertical", command=self.live_canvas.yview)
        self.live_canvas.configure(yscrollcommand=live_vscroll.set)
        self.live_canvas.pack(side="left", fill="both", expand=True)
        live_vscroll.pack(side="right", fill="y")

        frame = ttk.Frame(self.live_canvas)
        self._live_inner_window = self.live_canvas.create_window(
            (0, 0), window=frame, anchor="nw")

        def _sync_scrollregion(_event=None) -> None:
            self.live_canvas.configure(scrollregion=self.live_canvas.bbox("all"))

        def _sync_inner_width(event) -> None:
            self.live_canvas.itemconfigure(self._live_inner_window, width=event.width)

        frame.bind("<Configure>", _sync_scrollregion)
        self.live_canvas.bind("<Configure>", _sync_inner_width)

        def _on_mousewheel(event) -> None:
            if event.num == 4:            # Linux scroll up
                self.live_canvas.yview_scroll(-1, "units")
            elif event.num == 5:          # Linux scroll down
                self.live_canvas.yview_scroll(1, "units")
            else:                         # Windows / macOS
                self.live_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        def _bind_mousewheel(_event=None) -> None:
            self.live_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.live_canvas.bind_all("<Button-4>", _on_mousewheel)
            self.live_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mousewheel(_event=None) -> None:
            self.live_canvas.unbind_all("<MouseWheel>")
            self.live_canvas.unbind_all("<Button-4>")
            self.live_canvas.unbind_all("<Button-5>")

        # Only capture the wheel while the pointer is actually over the Live
        # tab, so scrolling elsewhere in the app is unaffected — mirrors the
        # Results tab's own _bind_mousewheel / _unbind_mousewheel.
        self.live_canvas.bind("<Enter>", _bind_mousewheel)
        self.live_canvas.bind("<Leave>", _unbind_mousewheel)

        source_box = ttk.LabelFrame(frame, text="Live Source")
        source_box.pack(fill="x", **pad)

        self.live_source_var = tk.StringVar(value="simulated")
        src_row = ttk.Frame(source_box)
        src_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(src_row, text="Source:", width=22, anchor="w").pack(side="left")
        ttk.Radiobutton(src_row, text="Simulated folder", value="simulated",
                        variable=self.live_source_var).pack(side="left")
        ttk.Radiobutton(src_row, text="IDS camera (lab)", value="ids",
                        variable=self.live_source_var).pack(side="left", padx=12)

        # Simulated-only controls live in their own sub-frame that is shown
        # only while "Simulated folder" is selected — at the lab you only
        # ever use "IDS camera", and these two fields (replay folder / fake
        # fps) are meaningless there, so hide them instead of leaving them
        # to clutter the screen next to the real controls.
        self._live_sim_box = ttk.Frame(source_box)
        self.live_folder_var = tk.StringVar(value=str(DATA_DIR / "signal"))
        self._folder_row(self._live_sim_box, "Simulated folder:", self.live_folder_var)
        self.live_fps_var = tk.DoubleVar(value=10.0)
        self._spin_row(self._live_sim_box, "Simulated FPS:", self.live_fps_var, 0.1, 200.0, 0.5)

        camera_row = ttk.Frame(source_box)
        ttk.Label(camera_row, text="IDS settings:", width=22, anchor="w").pack(side="left")
        ttk.Label(camera_row, text="Camera index / Exposure / Gain — set in the Parameters tab",
                  foreground="#6e6e73").pack(side="left")
        self._live_ids_box = camera_row

        def _sync_source_visibility(*_a) -> None:
            if self.live_source_var.get() == "simulated":
                self._live_ids_box.pack_forget()
                self._live_sim_box.pack(fill="x")
            else:
                self._live_sim_box.pack_forget()
                self._live_ids_box.pack(fill="x", padx=8, pady=4)
        self.live_source_var.trace_add("write", _sync_source_visibility)
        _sync_source_visibility()

        # ── Temporal coverage (duty cycle) ────────────────────────────────
        # Neutrons arriving between two exposures are never seen: the
        # counting efficiency is exposure × fps. This label makes that loss
        # explicit next to where fps is chosen. For the IDS camera, "Query
        # camera timing" reads the REAL ExposureTime/ResultingFrameRate from
        # the device (the camera may clamp exposure below 1/fps for readout).
        coverage_row = ttk.Frame(source_box)
        coverage_row.pack(fill="x", padx=8, pady=4)
        self.live_coverage_var = tk.StringVar()
        self.live_coverage_label = ttk.Label(coverage_row, textvariable=self.live_coverage_var)
        self.live_coverage_label.pack(side="left")
        ttk.Button(coverage_row, text="Query camera timing",
                   command=self._query_camera_timing).pack(side="left", padx=10)
        for var in (self.live_fps_var, self.live_source_var, self.acq_mode_var,
                    self.ids_energy_exposure_us_var, self.ids_exposure_us_var):
            var.trace_add("write", lambda *_: self._update_coverage_label())
        self._update_coverage_label()

        # ── Guided lab session: the four phases of a beam-time run ───────
        wf_box = ttk.LabelFrame(frame, text="Live Session Workflow (lab)")
        wf_box.pack(fill="x", **pad)
        wf_row = ttk.Frame(wf_box)
        wf_row.pack(fill="x", padx=8, pady=6)
        ttk.Button(wf_row, text="① Background model (beam OFF)",
                   command=self._workflow_background_before).pack(side="left")
        ttk.Button(wf_row, text="② Start measuring (beam ON)",
                   command=lambda: self._step_live_detection(self.live_source_var.get())
                   ).pack(side="left", padx=8)
        ttk.Button(wf_row, text="③ Camera health check (beam OFF)",
                   command=self._workflow_health_check).pack(side="left")
        ttk.Button(wf_row, text="④ Background sanity (false positives)",
                   command=self._run_background_sanity).pack(side="left", padx=8)
        ttk.Button(wf_row, text="⑤ End-of-experiment report",
                   command=self._run_end_of_experiment_report).pack(side="left")
        ttk.Label(wf_box, foreground="#6e6e73",
                  text="① capture darks + build noise model (keeps a dated 'before' copy)   "
                       "② live neutron counting   ③ fresh darks → 'after' model → damage report   "
                       "④ classify the dark-run clusters → false-positive floor of the ML model   "
                       "⑤ end-of-experiment report: 2 noise maps (mean/σ) + 2 arrival maps "
                       "(all particles / neutrons) + separability verdict for the run that "
                       "just finished (needs 'Save during live' = data_analysed) — in Results. "
                       "Energy spectrum: Parameters → Energy analysis (office, large datasets)."
                  ).pack(anchor="w", padx=8, pady=(0, 6))

        # These settings apply to BOTH sources — "Duration" is the total
        # session length before the engine auto-stops regardless of whether
        # frames come from the simulated replay or the real IDS camera, and
        # ② Start measuring above triggers this exact run for either one.
        run_box = ttk.LabelFrame(frame, text="Live Run Settings (both sources)")
        run_box.pack(fill="x", **pad)

        self.live_duration_var = tk.DoubleVar(value=30.0)
        self.live_batch_seconds_var = tk.DoubleVar(value=0.2)
        self.live_batch_max_frames_var = tk.IntVar(value=1)
        self.live_bundle_var = tk.StringVar(value=str(MODEL_BUNDLE))
        # Rolling-average window, in seconds of acquisition time rather than a
        # fixed batch count — see _append_live_counts_line for why: a batch
        # count has no fixed physical meaning (it depends on batch_seconds),
        # while a time window maps directly onto Poisson counting statistics
        # (relative error on the windowed neutron count ~= 1/sqrt(N)).
        self.live_rolling_window_s_var = tk.DoubleVar(value=10.0)

        # ── What gets written to disk during live ────────────────────────
        # "raw frames": bit-identical .bmp archive of every camera frame
        # (IDS-Cockpit replacement, ~20 MB/frame). "data_analysed": compact
        # products only (masked frames ~50 KB, noise model, arrival map) —
        # ~400x less disk. Both writers run on their own threads/queues, so
        # neither throttles detection.
        self.live_save_mode_var = tk.StringVar(value="raw + data_analysed")
        save_row = ttk.Frame(run_box)
        save_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(save_row, text="Save during live:", width=22, anchor="w").pack(side="left")
        ttk.Combobox(save_row, textvariable=self.live_save_mode_var, state="readonly",
                     width=24, values=("nothing", "data_analysed only",
                                       "raw frames only", "raw + data_analysed")
                     ).pack(side="left")
        self.live_record_dir_var = tk.StringVar(value=str(CODE_DIR / "live_recording"))
        self._folder_row(run_box, "Raw recording folder:", self.live_record_dir_var)

        self._spin_row(run_box, "Duration (s):", self.live_duration_var, 1.0, 3600.0, 1.0)
        self._spin_row(run_box, "Batch seconds:", self.live_batch_seconds_var, 0.1, 60.0, 0.1)
        self._spin_row(run_box, "Batch max frames:", self.live_batch_max_frames_var, 1, 1000, 1)
        self._spin_row(run_box, "Rolling window (s):", self.live_rolling_window_s_var, 1.0, 3600.0, 1.0)
        bundle_row = ttk.Frame(run_box)
        bundle_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(bundle_row, text="Model bundle:", width=22, anchor="w").pack(side="left")
        ttk.Entry(bundle_row, textvariable=self.live_bundle_var, width=54).pack(side="left", padx=4)
        ttk.Button(bundle_row, text="Browse…",
                   command=self._browse_live_bundle).pack(side="left")

        # No "Start Live" here on purpose: ② Start measuring (beam ON) above,
        # in the workflow row, IS the start button — one control per action,
        # not two paths to the same thing.
        btn_row = ttk.Frame(run_box)
        btn_row.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Button(btn_row, text="Stop Live",
                   command=self._interrupt_active_step).pack(side="left")
        ttk.Button(btn_row, text="Clear Live Counts",
                   command=self._clear_live_graph).pack(side="left", padx=8)

        status_box = ttk.LabelFrame(frame, text="Live Status")
        status_box.pack(fill="x", **pad)
        self.live_status_var = tk.StringVar(value="Idle")
        ttk.Label(status_box, textvariable=self.live_status_var,
                  foreground="#6e6e73").pack(anchor="w", padx=8, pady=4)

        stats_box = ttk.LabelFrame(frame, text="Live Statistics")
        stats_box.pack(fill="x", **pad)
        self.live_stats_var = tk.StringVar()
        ttk.Label(stats_box, textvariable=self.live_stats_var,
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=8, pady=6)

        # Throughput line: camera fps vs what the pipeline actually analyses.
        # Turns red as soon as frames are being dropped, because every rate
        # above (neutrons/s etc.) silently under-counts by exactly that loss.
        self.live_throughput_var = tk.StringVar()
        self.live_throughput_label = ttk.Label(
            stats_box, textvariable=self.live_throughput_var,
            font=("TkDefaultFont", 10, "bold"))
        self.live_throughput_label.pack(anchor="w", padx=8, pady=(0, 6))

        counts_box = ttk.LabelFrame(frame, text="Operational Particle Counts")
        counts_box.pack(fill="both", expand=True, **pad)
        self.live_counts_text = tk.Text(
            counts_box, height=12, wrap="none", font=("TkFixedFont", 12),
            bg="#0d1117", fg="#c9d1d9", insertbackground="#c9d1d9")
        self.live_counts_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.live_counts_text.configure(state="disabled")

        self._reset_live_plot()
        self._update_live_stats_label()

    def _browse_live_bundle(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(CODE_DIR),
            title="Select model bundle",
            filetypes=[("Joblib model bundle", "*.joblib"), ("All files", "*.*")]
        )
        if path:
            self.live_bundle_var.set(path)

    def _reset_live_plot(self) -> None:
        """Reset live telemetry counters without touching expensive plots."""
        self.live_last_elapsed = 0.0
        self.live_total_alpha = 0
        self.live_total_lithium = 0
        self.live_total_neutrons = 0.0   # alpha + lithium (one daughter per capture)
        self.live_total_frames = 0
        self.live_recent_batches: list[dict] = []

        # Dead/hot pixel mask stats (from LIVE_BATCH's "dead_pixels" block —
        # see _update_live_from_line). live_dead_total is None until the
        # first batch reports it, since it depends on the sensor frame shape
        # LiveDetectionEngine.py is actually running with.
        self.live_dead_count: int = 0
        self.live_dead_total: Optional[int] = None
        self.live_dead_pct: float = 0.0

        if hasattr(self, "live_counts_text"):
            self.live_counts_text.configure(state="normal")
            self.live_counts_text.delete("1.0", "end")
            self.live_counts_text.insert(
                "end",
                "Waiting for live counts...\n"
                "Frame mode: set Batch max frames = 1 for one line per camera frame.\n",
            )
            self.live_counts_text.configure(state="disabled")

        if hasattr(self, "live_throughput_var"):
            self.live_throughput_var.set("Throughput: waiting for first batch…")
            self.live_throughput_label.configure(foreground="#6e6e73")

        if hasattr(self, "live_stats_var"):
            self._update_live_stats_label()

    def _update_live_stats_label(self) -> None:
        neutron_rate = (self.live_total_neutrons / self.live_last_elapsed
                        if self.live_last_elapsed > 0 else 0.0)
        neutron_per_frame = (self.live_total_neutrons / self.live_total_frames
                             if self.live_total_frames > 0 else 0.0)
        if self.live_dead_total:
            dead_line = (
                f"    |    Dead Cells: {self.live_dead_count:,} / "
                f"{self.live_dead_total:,} ({self.live_dead_pct:.2f}%)"
            )
        else:
            dead_line = "    |    Dead Cells: —"
        # Alpha and lithium are shown right next to the neutron count
        # (= alpha + lithium, one daughter per capture) on purpose: their
        # deposited energies differ, so a drift in their ratio is a live
        # diagnostic of the detection/classification response.
        alpha_rate = (self.live_total_alpha / self.live_last_elapsed
                      if self.live_last_elapsed > 0 else 0.0)
        lithium_rate = (self.live_total_lithium / self.live_last_elapsed
                        if self.live_last_elapsed > 0 else 0.0)
        self.live_stats_var.set(
            f"Neutrons: {self.live_total_neutrons:.1f}    |    "
            f"Alpha: {self.live_total_alpha} ({alpha_rate:.2f}/s)    |    "
            f"Lithium: {self.live_total_lithium} ({lithium_rate:.2f}/s)    |    "
            f"Frames: {self.live_total_frames}    |    "
            f"Elapsed Time: {self.live_last_elapsed:.1f}s    |    "
            f"Neutron Rate: {neutron_rate:.2f}/s    |    "
            f"Neutrons/Frame: {neutron_per_frame:.3f}"
            + dead_line
        )

    def _clear_live_graph(self) -> None:
        self.live_status_var.set("Idle")
        self._reset_live_plot()

    def _redraw_live_graph(self) -> None:
        # Kept as a no-op for older callbacks; live plotting is intentionally
        # disabled to keep acquisition responsive.
        return

    def _append_live_counts_line(self, elapsed: float, summary: dict, counts: dict) -> None:
        if not hasattr(self, "live_counts_text"):
            return
        frames = max(1, int(summary.get("frames", 1) or 1))
        alpha = int(summary.get("alpha", 0) or 0)
        lithium = int(summary.get("lithium", 0) or 0)
        neutrons = float(summary.get("neutrons", 0) or 0)

        self.live_recent_batches.append({
            "elapsed": elapsed,
            "frames": frames,
            "alpha": alpha,
            "lithium": lithium,
            "neutrons": neutrons,
        })
        # Keep only batches within the configured time window, not a fixed
        # batch count — a batch count has no fixed physical meaning since it
        # depends on batch_seconds/batch_max_frames, while a time window
        # relates directly to acquisition time and to Poisson counting
        # statistics (see the relative-error estimate below).
        window_s = max(0.1, float(self.live_rolling_window_s_var.get()))
        cutoff = elapsed - window_s
        self.live_recent_batches = [
            item for item in self.live_recent_batches if item["elapsed"] > cutoff
        ]
        recent_frames = sum(item["frames"] for item in self.live_recent_batches) or 1
        recent_neutrons = sum(item["neutrons"] for item in self.live_recent_batches)
        recent_alpha = sum(item["alpha"] for item in self.live_recent_batches)
        recent_lithium = sum(item["lithium"] for item in self.live_recent_batches)
        # Actual covered span (can be shorter than window_s right after
        # start/reset) — use it instead of window_s so the rate is not
        # under-estimated while the window is still filling up.
        recent_span = max(elapsed - self.live_recent_batches[0]["elapsed"], 1e-6) \
            if self.live_recent_batches else 1e-6
        recent_neutron_rate = recent_neutrons / recent_span

        # Poisson counting statistics: relative error on a count N is
        # ~1/sqrt(N). This tells you whether the window holds enough
        # neutrons for the displayed rate to mean anything (rule of thumb:
        # need N>=100 for ~10% precision, N>=10000 for ~1%).
        rel_err_pct = (100.0 / (recent_neutrons ** 0.5)) if recent_neutrons >= 1 else float("nan")

        label_bits = [
            f"{label}={count}" for label, count in sorted(counts.items())
            if int(count) != 0
        ]
        label_text = ", ".join(label_bits) if label_bits else "no detections"
        line = (
            f"{elapsed:8.2f}s | frames={frames:3d} | "
            f"n={neutrons:5.1f} a={alpha:3d} li={lithium:3d} | "
            f"n/frame={neutrons / frames:5.2f} | "
            f"[{window_s:g}s window] N={recent_neutrons:6.1f} "
            f"rate={recent_neutron_rate:6.2f}/s (±{rel_err_pct:4.1f}%) "
            f"a={recent_alpha:4d} li={recent_lithium:4d} | "
            f"{label_text}\n"
        )

        self.live_counts_text.configure(state="normal")
        if self.live_counts_text.get("1.0", "1.end").startswith("Waiting for"):
            self.live_counts_text.delete("1.0", "end")
            self.live_counts_text.insert(
                "end",
                " elapsed | batch      | instant counts       | rolling window (time-based)                          | labels\n",
            )
        self.live_counts_text.insert("end", line)
        self.live_counts_text.see("end")
        self.live_counts_text.configure(state="disabled")

    def _update_live_from_line(self, text: str) -> None:
        # One-time modal for the operating-point mismatch: the engine emits it
        # at most once per run, so surfacing it as a dialog (rather than a
        # console line the batch stream immediately scrolls past) can't spam —
        # and this is important enough that the operator must not miss it.
        if "OPERATING-POINT MISMATCH" in text:
            messagebox.showwarning("Operating-point mismatch", text)
            return

        if text.startswith("LIVE_BATCH:"):
            payload_text = text[len("LIVE_BATCH:"):].strip()
            try:
                payload = json.loads(payload_text)
            except (json.JSONDecodeError, ValueError):
                self._append_console(f"[live] Could not parse LIVE_BATCH payload: {payload_text}")
                return

            counts = payload.get("counts", {})
            if not isinstance(counts, dict):
                return
            elapsed = float(payload.get("elapsed_seconds", self.live_last_elapsed))
            summary = payload.get("particle_summary", {})
            if not isinstance(summary, dict):
                summary = {}
            dead_pixels = payload.get("dead_pixels")
            if isinstance(dead_pixels, dict):
                self.live_dead_count = int(dead_pixels.get("count", 0) or 0)
                total = dead_pixels.get("total")
                self.live_dead_total = int(total) if total else None
                self.live_dead_pct = float(dead_pixels.get("pct", 0.0) or 0.0)

            self.live_last_elapsed = elapsed
            self.live_total_frames += int(summary.get("frames", 1) or 1)
            self.live_total_alpha += int(summary.get("alpha", 0) or 0)
            self.live_total_lithium += int(summary.get("lithium", 0) or 0)
            self.live_total_neutrons += float(summary.get("neutrons", 0) or 0)

            throughput = payload.get("throughput")
            if isinstance(throughput, dict):
                self._update_live_throughput_label(throughput)

            self._append_live_counts_line(elapsed, summary, counts)
            self._update_live_stats_label()
            self.live_status_var.set(
                f"Batch @ {elapsed:.1f}s — "
                f"neutrons={float(summary.get('neutrons', 0) or 0):.1f}, "
                f"alpha={int(summary.get('alpha', 0) or 0)}, "
                f"lithium={int(summary.get('lithium', 0) or 0)}"
            )
            return

        # Anything else (the initial "[live] started ...", "[live] status
        # ...", "[live] stopped", etc.) just goes to the status line as-is.
        self.live_status_var.set(text)

    def _update_live_throughput_label(self, thr: dict) -> None:
        """Camera fps vs analysed fps, with explicit dropped-frame count.

        The red state is the point of this label: as soon as drop_pct > 0
        every displayed rate under-counts reality by that fraction, so the
        operator must either lower the camera fps or accept biased rates."""
        if not hasattr(self, "live_throughput_var"):
            return
        camera_fps = float(thr.get("camera_fps", 0) or 0)
        processing_fps = float(thr.get("processing_fps", 0) or 0)
        dropped = int(thr.get("dropped", 0) or 0)
        drop_pct = float(thr.get("drop_pct", 0) or 0)
        captured = int(thr.get("captured", 0) or 0)
        processed = int(thr.get("processed", 0) or 0)
        server_ms = float(thr.get("server_ms_per_frame", 0) or 0)
        classify_ms = float(thr.get("classify_ms", 0) or 0)

        # The requested rate is shown ALWAYS, next to the achieved one, however
        # small the gap: the operator should never have to wonder whether the
        # camera is doing what was asked. Colour is a separate decision, below.
        requested_fps = float(thr.get("requested_fps") or 0)
        shortfall = (requested_fps - camera_fps) / requested_fps if requested_fps else 0.0
        asked = f"/{requested_fps:.2f} asked" if requested_fps else ""

        text = (
            f"Camera: {camera_fps:.2f}{asked} fps ({captured} frames)    |    "
            f"Analysed: {processing_fps:.2f} fps ({processed} frames)    |    "
            f"Dropped: {dropped} ({drop_pct:.0f}%)    |    "
            f"Detector: {server_ms:.0f} ms/frame    |    "
            f"Classify: {classify_ms:.0f} ms/batch"
        )
        # Three states: red = frames already lost; orange = none lost yet but
        # the queue is filling (drops are only minutes away at this pace);
        # green = pipeline keeps up with the camera.
        # Is the CAMERA itself keeping its promise? The checks below compare the
        # camera with the analysis; this one compares the camera with what was
        # asked of it. A requested rate the sensor cannot sustain (typically an
        # exposure longer than the frame period) is capped silently: you plan a
        # run on N frames and come back with fewer, and the discrepancy only
        # surfaces later, if ever. The measured rates stay correct -- everything
        # here divides by real elapsed time -- but the run PLAN does not.
        #
        # The 20-frame gate is not a significance threshold, it is a guard
        # against an artefact: camera_fps is cumulative since run start, so it
        # still carries the camera's initialisation and first exposure. One
        # frame after 2 s of start-up reads as 0.5 fps -- a fake 75% shortfall.
        # The gap itself is displayed from the very first frame regardless; only
        # the WARNING waits until the average means something.
        backlog = max(0, captured - processed - dropped)
        if dropped > 0:
            text += "    ⚠ camera faster than analysis — rates under-count!"
            self.live_throughput_label.configure(foreground="#c62828")
        elif requested_fps and shortfall > 0.05 and captured > 20:
            text += (f"    ⚠ camera delivering {camera_fps:.2f} fps, "
                     f"{requested_fps:.2f} requested ({100*shortfall:.0f}% short) "
                     f"— exposure too long for this rate?")
            self.live_throughput_label.configure(foreground="#e65100")
        elif backlog > 10 and camera_fps > processing_fps * 1.1:
            text += f"    ⚠ queue filling ({backlog} frames waiting) — drops imminent"
            self.live_throughput_label.configure(foreground="#e65100")
        else:
            self.live_throughput_label.configure(foreground="#2e7d32")
        self.live_throughput_var.set(text)

    # ────────────────────────────────────────────────────────────────────
    #  UI: Console tab
    # ────────────────────────────────────────────────────────────────────

    def _build_console_tab(self) -> None:
        frame = self.tab_console
        self.console_text = tk.Text(
            frame, wrap="word", bg="#0d1117", fg="#c9d1d9",
            insertbackground="#c9d1d9", font=("TkFixedFont", 10),
        )
        self.console_text.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.console_text.configure(state="disabled")

        bottom = ttk.Frame(frame)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        self.stdin_var = tk.StringVar()
        entry = ttk.Entry(bottom, textvariable=self.stdin_var)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        entry.bind("<Return>", lambda e: self._send_stdin())
        ttk.Button(bottom, text="Send to process",
                   command=self._send_stdin).pack(side="left")
        ttk.Button(bottom, text="Clear",
                   command=self._clear_console).pack(side="left", padx=6)

        tk.Label(
            frame,
              text="Tip: use the Console tab only for interactive subprocesses like the labeling UI. "
                  "Type your reply above and hit Enter or 'Send to process' while it runs.",
            fg="#6e6e73", font=("TkDefaultFont", 9),
        ).pack(fill="x", padx=8, pady=(0, 6))

    def _append_console(self, text: str) -> None:
        self.console_text.configure(state="normal")
        self.console_text.insert("end", text + "\n")
        self.console_text.see("end")
        self.console_text.configure(state="disabled")

    def _format_console_line(self, step_id: str, text: str) -> str:
        clean_line = ANSI_ESCAPE_RE.sub("", text).replace("\r", "").strip()
        clean_line = re.sub(rf"^\s*\[{re.escape(step_id)}\]\s*", "", clean_line)
        return clean_line.strip()

    def _clear_console(self) -> None:
        self.console_text.configure(state="normal")
        self.console_text.delete("1.0", "end")
        self.console_text.configure(state="disabled")

    def _interrupt_active_step(self) -> None:
        proc = self.active_proc
        if proc is None:
            self._append_console("[INFO] No active process to interrupt.")
            return
        self._interrupt_requested = True
        step_id = self.active_proc_step or "process"
        try:
            proc.terminate()
            self._append_console(f"[{step_id}] Interrupt requested.")
        except Exception as exc:
            self._append_console(f"[{step_id}] Could not interrupt process: {exc}")

    def _send_stdin(self) -> None:
        text = self.stdin_var.get()
        if self.active_proc is not None and self.active_proc.stdin:
            try:
                self.active_proc.stdin.write(text + "\n")
                self.active_proc.stdin.flush()
                self._append_console(f"[{self.active_proc_step}] > {text}")
            except Exception as exc:
                self._append_console(f"[ERROR] Could not send input: {exc}")
        else:
            self._append_console("[INFO] No active process waiting for input.")
        self.stdin_var.set("")

    # ────────────────────────────────────────────────────────────────────
    #  UI: Results tab
    # ────────────────────────────────────────────────────────────────────

    def _build_results_tab(self) -> None:
        frame = self.tab_results
        top = ttk.Frame(frame)
        top.pack(fill="x", padx=8, pady=8)
        # No Energy Spectrum button here: it lives in Parameters → "Energy
        # analysis (office)", tied to its large-dataset path field — a
        # 100–500-frame lab run has too few tracks for a meaningful spectrum.
        ttk.Button(top, text="↻  Refresh Preview",
                   command=self._load_results_preview).pack(side="left")
        ttk.Button(top, text="🔬 Check separability",
                   command=self._run_separability_check).pack(side="left", padx=(8, 0))
        self.result_caption_var = tk.StringVar(
            value="No preview yet — run Background Model or Cluster Detection.")
        ttk.Label(top, textvariable=self.result_caption_var,
                  foreground="#6e6e73").pack(side="left", padx=10)
        if not MPL_AVAILABLE:
            ttk.Label(top, text="matplotlib not installed — showing flat previews, no colorbar "
                                 "(pip install matplotlib)",
                      foreground="#b91c1c").pack(side="left", padx=10)

        # Scrollable single-column viewer: every available preview is stacked
        # vertically inside `results_inner`, which sits inside a Canvas that
        # scrolls. The inner frame's width is always kept in sync with the
        # canvas's visible width (see _sync_inner_width), so panels lay out
        # in one column and never grow wider than the window — only taller,
        # which the scrollbar/mouse-wheel handle.
        outer = ttk.Frame(frame)
        outer.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.results_canvas = tk.Canvas(outer, bg="#e0e0e0", highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=self.results_canvas.yview)
        self.results_canvas.configure(yscrollcommand=vscroll.set)
        self.results_canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        self.results_inner = ttk.Frame(self.results_canvas)
        self._results_inner_window = self.results_canvas.create_window(
            (0, 0), window=self.results_inner, anchor="nw")

        def _sync_scrollregion(_event=None) -> None:
            self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

        def _sync_inner_width(event) -> None:
            self.results_canvas.itemconfigure(self._results_inner_window, width=event.width)

        self.results_inner.bind("<Configure>", _sync_scrollregion)
        self.results_canvas.bind("<Configure>", _sync_inner_width)

        def _on_mousewheel(event) -> None:
            if event.num == 4:            # Linux scroll up
                self.results_canvas.yview_scroll(-1, "units")
            elif event.num == 5:          # Linux scroll down
                self.results_canvas.yview_scroll(1, "units")
            else:                         # Windows / macOS
                self.results_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        def _bind_mousewheel(_event=None) -> None:
            self.results_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.results_canvas.bind_all("<Button-4>", _on_mousewheel)
            self.results_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mousewheel(_event=None) -> None:
            self.results_canvas.unbind_all("<MouseWheel>")
            self.results_canvas.unbind_all("<Button-4>")
            self.results_canvas.unbind_all("<Button-5>")

        # Only capture the wheel while the pointer is actually over the
        # preview panel, so scrolling elsewhere in the app is unaffected.
        self.results_canvas.bind("<Enter>", _bind_mousewheel)
        self.results_canvas.bind("<Leave>", _unbind_mousewheel)

        self._result_photos: list = []     # keep PhotoImage refs alive (Tk drops unreferenced ones)
        self._results_canvases: list = []  # keep FigureCanvasTkAgg refs alive
        # Per-report overrides shown ahead of the generic sources; set by the
        # ⑤ End-of-experiment report, empty otherwise.
        self._report_heatmaps: list = []
        self._report_flats: list = []

    # ── discovering what's available to preview ──────────────────────────

    def _collect_heatmap_previews(self) -> list[tuple[str, Path, bool]]:
        """Raw float32 .tiff sources with a matching file on disk — these get
        the full matplotlib treatment (colorbar + log/linear colormap)."""
        return [(caption, path, allow_log) for caption, path, allow_log in RAW_HEATMAP_SOURCES
                if path.is_file()]

    def _collect_fallback_heatmap_pngs(
        self, have_raw: list[tuple[str, Path, bool]]
    ) -> list[tuple[str, Path]]:
        """Pre-rendered PNG heatmaps (already contain DataExporter's own
        baked-in colorbar) — shown only when the matching raw .tiff isn't
        available yet, e.g. before rebuilding the C++ pipeline with the new
        exporter, or when matplotlib isn't installed."""
        legacy = [
            ("Background mean",            CODE_DIR / "data_analysed" / "dark"   / "background_mean_heatmap.png"),
            ("Background fluctuation (σ)", CODE_DIR / "data_analysed" / "dark"   / "background_sigma_heatmap.png"),
            ("Signal arrival (live)",      CODE_DIR / "live_signal_arrival_heatmap.png"),
        ]
        have_captions = {caption for caption, _, _ in have_raw}
        return [(caption, path) for caption, path in legacy
                if caption not in have_captions and path.is_file()]

    def _collect_flat_previews(self) -> list[tuple[str, Path]]:
        return [(caption, path) for caption, path in FLAT_IMAGE_SOURCES if path.is_file()]

    def _run_energy_spectrum(self, csv_path: Optional[Path] = None,
                             source_label: str = "offline pipeline",
                             extra_flat: Optional[tuple] = None) -> None:
        """Deposited-energy histogram of the heavy (neutron-daughter) tracks.

        Runs src/study_energy.py on a classified-clusters CSV: by default the
        OFFLINE end-of-pipeline product (detect + Predict). Pass csv_path to
        point it at a different one instead — e.g. a day's accumulated live
        CSV — since both files share the same column layout (total_charge,
        peak_value, predicted_label...). Always writes to the single canonical
        energy_histogram.png so the Results tab has one "latest spectrum you
        asked for" slot. extra_flat, when given, is appended to the report's
        flat panels once the PNG exists (used by the end-of-experiment report)."""
        csv_path = csv_path or DETECTIONS_CLASSIFIED_CSV
        if not csv_path.is_file():
            messagebox.showerror(
                "Energy Spectrum",
                f"No classified detections found:\n{csv_path}\n\n"
                "Run Cluster Detection then Predict (step 6) first.",
            )
            return
        out_png = CODE_DIR / "energy_histogram.png"
        cmd = [sys.executable, str(STUDY_ENERGY_SCRIPT),
               "--csv", str(csv_path), "--out", str(out_png)]
        self._append_console(f"[energy] Computing deposited-energy spectrum ({source_label})…")

        def worker() -> None:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(CODE_DIR))
            lines = [f"[energy] {ln}" for ln in
                     (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines()]
            if proc.returncode != 0:
                lines.append(f"[energy] FAILED (exit {proc.returncode})")

            def apply() -> None:
                for ln in lines:
                    self._append_console(ln)
                if proc.returncode == 0:
                    if extra_flat is not None and Path(extra_flat[1]).is_file():
                        self._report_flats = getattr(self, "_report_flats", []) + [extra_flat]
                    self._load_results_preview()
                else:
                    # A silent failure looks like "the button does nothing" —
                    # surface the reason in a dialog, not just the console.
                    detail = "\n".join(lines[-6:]) or "(no output)"
                    messagebox.showerror(
                        "Energy Spectrum",
                        f"study_energy.py failed (exit {proc.returncode}):\n\n{detail}")
            # Tk widgets are only touched from the main thread.
            self.root.after(0, apply)

        threading.Thread(target=self._guarded("computing the energy spectrum", worker),
                         daemon=True, name="energy-spectrum").start()

    def _run_separability_check(self, csv_path: Optional[Path] = None,
                                silent_ok: bool = False) -> None:
        """Run study_energy's signal-vs-noise separability gate. Defaults to
        the freshest classified CSV (last live experiment, else offline). A
        MERGED verdict (exit 3) always pops a red dialog — the neutron count
        can no longer be trusted; silent_ok suppresses only the reassuring
        "still separable" popup (used inside the end-of-experiment report)."""
        if csv_path is None:
            csv_path = self._live_session_csv()
            if csv_path is None or not csv_path.is_file():
                csv_path = DETECTIONS_CLASSIFIED_CSV
        if not csv_path.is_file():
            messagebox.showerror(
                "Separability", f"No classified detections found:\n{csv_path}\n\n"
                "Run detection + Predict, or a live session with data_analysed on.")
            return
        cmd = [sys.executable, str(STUDY_ENERGY_SCRIPT),
               "--csv", str(csv_path), "--check-separability"]
        self._append_console(f"[separability] Checking {csv_path.name}…")

        def worker() -> None:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(CODE_DIR))
            out = (proc.stdout or "") + (proc.stderr or "")

            def apply() -> None:
                for ln in out.splitlines():
                    self._append_console(f"[separability] {ln}")
                if proc.returncode == 3:
                    messagebox.showerror(
                        "Separability — MERGED",
                        "Signal and noise can no longer be separated at this "
                        "operating point.\nThe neutron count is NOT trustworthy — "
                        "raise the gain back up, or re-label & retrain here.\n\n"
                        "See the Console for the per-feature breakdown.")
                elif proc.returncode == 0 and not silent_ok:
                    messagebox.showinfo(
                        "Separability", "Signal and noise are still separable "
                        "(see Console for the gap on each feature).")
                elif proc.returncode not in (0, 3):
                    messagebox.showerror(
                        "Separability", f"Check failed (exit {proc.returncode}):\n"
                        + "\n".join(out.splitlines()[-6:]))
            self.root.after(0, apply)

        threading.Thread(target=self._guarded("computing the separability verdict", worker),
                         daemon=True, name="separability").start()

    def _render_neutron_arrival_png(self, csv_path: Path) -> Optional[Path]:
        """Where did the NEUTRONS land? A 2D histogram of the sensor
        positions of clusters the classifier called lithium/alpha (i.e.
        neutron-capture daughters), unlike the C++ arrival map which counts
        every detected particle. White = no neutron; colour = neutron density."""
        if pd is None or np is None or not MPL_AVAILABLE:
            return None
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            self._append_console(f"[report] Could not read {csv_path.name}: {exc}")
            return None
        lab = "predicted_label" if "predicted_label" in df.columns else "label"
        if lab not in df.columns or not {"center_x", "center_y"}.issubset(df.columns):
            return None
        neut = df[df[lab].astype(str).str.lower().isin(["lithium", "alpha"])]
        x = pd.to_numeric(neut["center_x"], errors="coerce")
        y = pd.to_numeric(neut["center_y"], errors="coerce")
        ok = x.notna() & y.notna()
        x, y = x[ok].to_numpy(), y[ok].to_numpy()

        fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=100, constrained_layout=True)
        ax.set_facecolor("white")
        if len(x) == 0:
            ax.text(0.5, 0.5, "No neutrons classified", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
        else:
            w = int(np.ceil(max(x.max(), 1))); h = int(np.ceil(max(y.max(), 1)))
            bins_x = min(400, max(20, w // 16))
            bins_y = max(1, int(round(bins_x * h / w)))
            hist, xe, ye = np.histogram2d(x, y, bins=[bins_x, bins_y],
                                          range=[[0, w], [0, h]])
            cmap = plt.cm.YlOrRd.copy(); cmap.set_bad("white")
            masked = np.ma.masked_where(hist.T <= 0, hist.T)
            im = ax.imshow(masked, origin="upper", extent=[0, w, h, 0],
                           cmap=cmap, aspect="auto", interpolation="nearest")
            ax.set_xlabel("sensor x (px)"); ax.set_ylabel("sensor y (px)")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("neutrons per bin", fontsize=8)
        ax.set_title(f"Neutron arrival map ({len(x):,} neutrons)", fontsize=9)
        out = csv_path.parent / "neutron_arrival_map.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        return out

    def _run_end_of_experiment_report(self) -> None:
        """⑤ End-of-experiment report — one click after a live run shows the
        whole picture for the EXPERIMENT that just finished: the two noise
        maps (mean, σ), the two arrival maps (all particles / neutrons only),
        and an automatic signal-vs-noise separability verdict.

        Deliberately NO energy spectrum here: a normal experiment (100–500
        frames) has far too few heavy tracks for meaningful peaks — the
        spectrum lives in Parameters → 'Energy analysis (office)', to be run
        on large accumulated datasets."""
        sess_dir = self._live_session_dir()
        csv_path = self._live_session_csv()
        if sess_dir is None or csv_path is None or not csv_path.is_file():
            messagebox.showerror(
                "End-of-experiment report",
                "No recorded live experiment found.\n\n"
                "The report reads the session recorded with 'Save during live' = "
                "data_analysed. The last run used 'nothing' or 'raw frames only', "
                "so there is nothing to report.")
            return

        self._append_console(f"[report] End-of-experiment report for {sess_dir.name}…")
        self._report_heatmaps = [
            (cap, sess_dir / fn, True) for cap, fn in (
                ("Noise — background mean (this experiment)", "background_mean.tiff"),
                ("Noise — background σ (this experiment)", "background_sigma.tiff"),
                ("Arrival — all detected particles (this experiment)", "signal_arrival.tiff"),
            ) if (sess_dir / fn).is_file()
        ]
        self._report_flats = []
        arrival = self._render_neutron_arrival_png(csv_path)
        if arrival is not None:
            self._report_flats.append(("Neutron arrival map (this experiment)", arrival))

        # Separability runs as a subprocess (dialog only if MERGED) and the
        # preview below immediately shows the four maps.
        self._run_separability_check(csv_path=csv_path, silent_ok=True)
        self.nb.select(self.tab_results)
        self._load_results_preview()

    def _load_results_preview(self) -> None:
        # Per-report overrides, set by ⑤ End-of-experiment report so the run's
        # live-session maps are shown first, ahead of the generic offline ones.
        heatmaps = list(getattr(self, "_report_heatmaps", []))
        flats = list(getattr(self, "_report_flats", []))
        heatmaps += self._collect_heatmap_previews() if MPL_AVAILABLE else []
        legacy_heatmaps = self._collect_fallback_heatmap_pngs(heatmaps)
        flats += self._collect_flat_previews()

        particle_counts_path = CODE_DIR / "particle_counts_by_type.png"
        if DETECTIONS_CLASSIFIED_CSV.is_file() and not particle_counts_path.is_file():
            particle_counts = self._render_particle_counts_png()
            if particle_counts is not None:
                flats.append(("Detected particles by type", particle_counts))

        if not (heatmaps or legacy_heatmaps or flats) and DETECTIONS_CLASSIFIED_CSV.is_file():
            summary = self._render_prediction_summary_png()
            if summary is not None:
                flats = [("Prediction summary", summary)]

        for child in list(self.results_inner.winfo_children()):
            child.destroy()
        self._result_photos = []
        self._results_canvases = []

        total = len(heatmaps) + len(legacy_heatmaps) + len(flats)
        if total == 0:
            self._append_console("[results] No preview image available yet.")
            self.result_caption_var.set("No preview yet — run Background Model or Cluster Detection.")
            ttk.Label(self.results_inner, text="No preview yet",
                      foreground="#6e6e73").pack(pady=40)
            return

        self.result_caption_var.set(f"Showing {total} preview{'s' if total != 1 else ''}")
        # Each panel is independent: one bad/corrupt file must not abort the
        # whole preview and leave the tab looking empty while the caption
        # still claims "Showing N previews" — catch, log, keep going.
        shown = 0
        for caption, path, allow_log in heatmaps:
            try:
                self._add_heatmap_panel(caption, path, allow_log)
                shown += 1
            except Exception as exc:
                self._append_console(f"[results] Failed to render {caption}: {exc}")
        for caption, path in legacy_heatmaps:
            try:
                self._add_flat_image_panel(
                    caption, path,
                    note="(pre-rendered by DataExporter — rebuild the C++ pipeline for a live colorbar)")
                shown += 1
            except Exception as exc:
                self._append_console(f"[results] Failed to render {caption}: {exc}")
        for caption, path in flats:
            try:
                self._add_flat_image_panel(caption, path)
                shown += 1
            except Exception as exc:
                self._append_console(f"[results] Failed to render {caption}: {exc}")

        if shown < total:
            self.result_caption_var.set(f"Showing {shown}/{total} previews — see Console for errors")

        self.nb.select(self.tab_results)
        self.results_inner.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _render_prediction_summary_png(self) -> Optional[Path]:
        if pd is None:
            self._append_console(
                "[classify] pandas not installed — skipping auto-summary chart "
                "(pip install pandas to enable it).")
            return None
        try:
            df = pd.read_csv(DETECTIONS_CLASSIFIED_CSV)
        except Exception as exc:
            self._append_console(f"[classify] Could not read classified CSV: {exc}")
            return None
        if "predicted_label" not in df.columns:
            return None
        counts = df["predicted_label"].value_counts()
        out_path = CODE_DIR / "prediction_summary.png"
        render_bar_chart_png(counts.to_dict(), out_path, title="Predicted label counts")
        return out_path

    def _render_particle_counts_png(self) -> Optional[Path]:
        if pd is None:
            return None
        try:
            df = pd.read_csv(DETECTIONS_CLASSIFIED_CSV)
        except Exception as exc:
            self._append_console(f"[classify] Could not read classified CSV: {exc}")
            return None
        if "predicted_label" not in df.columns:
            return None
        counts = df["predicted_label"].value_counts().sort_index()
        out_path = CODE_DIR / "particle_counts_by_type.png"
        render_bar_chart_png(counts.to_dict(), out_path, title="Detected particles by predicted type")
        return out_path


    # ── rendering individual panels into the scrollable column ───────────

    def _add_heatmap_panel(self, caption: str, tiff_path: Path, allow_log: bool) -> None:
        """One matplotlib figure with colorbar, rendered to PNG then shown as
        a lightweight Tk image so scrolling stays responsive.

        These sensor maps are typically ~98% exactly zero (a clean CMOS reads
        0 ADU on almost every dark pixel), so the useful content is the sparse
        few percent of hot/noisy pixels. Two choices make that visible:
          • MAX-pooling when downscaling, not stride sampling — a lone hot
            pixel sits on one row/column and stride sampling would step right
            over it, so it would vanish from the preview entirely.
          • A robust color scale clipped at the 99.9th percentile, not the
            global max — otherwise one 255-ADU hot pixel stretches the whole
            colormap and washes everything else out (and log scale renders the
            98% zeros as blank white, which is what made these look empty)."""
        try:
            with Image.open(tiff_path) as im:
                data = np.asarray(im, dtype=np.float32)
        except Exception as exc:
            self._append_console(f"[results] Could not read {tiff_path.name}: {exc}")
            return

        vmax_full = float(data.max())
        nonzero_frac = float(np.mean(data > 0)) if data.size else 0.0

        # Down-scale for DISPLAY only (imshow on 20 MP freezes the Tk thread),
        # by MAX-pooling each block so sparse hot pixels survive.
        step = max(1, data.shape[1] // 1400)
        if step > 1:
            h = (data.shape[0] // step) * step
            w = (data.shape[1] // step) * step
            data = data[:h, :w].reshape(h // step, step, w // step, step).max(axis=(1, 3))

        # Robust upper bound: p99.9 of the ORIGINAL data (not of the pooled
        # copy) so the scale is stable; fall back to max for near-empty maps.
        hi = float(np.percentile(data, 99.9)) if data.size else vmax_full
        if hi <= 0:
            hi = vmax_full if vmax_full > 0 else 1.0
        norm = Normalize(vmin=0.0, vmax=hi)

        # WHITE background: the ~98% exactly-zero pixels are masked to white so
        # the eye reads the sparse coloured specks (hot/noisy pixels) against a
        # clean white field, which stands out far better than on black. Only
        # the non-zero pixels get a colour (pale yellow -> deep red = hotter).
        masked = np.ma.masked_where(data <= 0, data)
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad("white")

        fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=100, constrained_layout=True)
        ax.set_facecolor("white")
        im_artist = ax.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"{caption}   (white = 0; scale 0–{hi:.3g} ADU, max {vmax_full:.3g}; "
                     f"{100*nonzero_frac:.1f}% non-zero)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im_artist, ax=ax, fraction=0.046, pad=0.03, extend="max")
        cbar.set_label("ADU", fontsize=8)

        preview_path = CODE_DIR / f"results_preview_{tiff_path.stem}.png"
        fig.savefig(preview_path, dpi=130)
        plt.close(fig)
        self._add_flat_image_panel(caption, preview_path)

    def _add_flat_image_panel(self, caption: str, path: Path, note: str = "") -> None:
        """Plain image panel (no colormap/colorbar) — used for classifier
        charts and for legacy pre-rendered heatmap PNGs."""
        try:
            img = Image.open(path)
        except Exception as exc:
            self._append_console(f"[results] Could not open {path.name}: {exc}")
            return
        available = self.results_canvas.winfo_width()
        max_w = available - 40 if available > 100 else 760
        scale = min(max_w / img.width, 1.0)
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._result_photos.append(photo)  # keep alive — Tk drops unreferenced PhotoImages

        panel = ttk.Frame(self.results_inner, relief="groove", borderwidth=1)
        panel.pack(fill="x", padx=4, pady=6)
        label_text = f"{caption}  {note}" if note else caption
        ttk.Label(panel, text=label_text, foreground="#6e6e73").pack(anchor="w", padx=6, pady=(4, 0))
        tk.Label(panel, image=photo).pack(padx=6, pady=6)

    # ────────────────────────────────────────────────────────────────────
    #  GENERIC ASYNC STEP RUNNER
    # ────────────────────────────────────────────────────────────────────

    def _detector_exe(self) -> Path:
        exe = find_detector_exe()
        if exe is None:
            searched = "\n".join(f"  {p}" for p in detector_exe_candidates())
            raise FileNotFoundError(
                f"Detector executable not found. Searched:\n{searched}\n"
                f"Run the 'Build C++ Pipeline' step first, or fix DETECTOR_EXE_NAME "
                f"at the top of this file to match your CMakeLists.txt target name."
            )
        return exe

    def _start_step(self, step_id: str, commands: list,
                     cwd: Optional[Path] = None,
                     on_success: Optional[Callable[[], None]] = None) -> None:
        if self._busy_step is not None:
            messagebox.showwarning(
                "Pipeline busy",
                f"Step '{self._busy_step}' is still running — wait for it to finish.")
            return
        self._busy_step = step_id
        self._interrupt_requested = False
        self.state.mark_running(step_id)
        if step_id in self.step_rows:
            self._refresh_step_row(step_id)
        self._append_console(f"[{step_id}] ▶ starting…")
        threading.Thread(
            target=self._run_sequence,
            args=(step_id, commands, cwd, on_success),
            daemon=True,
        ).start()

    def _run_sequence(self, step_id: str, steps: list,
                       cwd: Optional[Path],
                       on_success: Optional[Callable[[], None]]) -> None:
        ok = True
        for step in steps:
            if callable(step):
                # Pure-python action (file I/O only — never touches Tkinter).
                try:
                    step()
                except Exception as exc:
                    self.q.put(("line", step_id, f"[ERROR] {exc}"))
                    ok = False
                    break
                continue

            cmd = step  # argv list
            self.q.put(("line", step_id, "$ " + " ".join(str(c) for c in cmd)))
            try:
                proc = subprocess.Popen(
                    [str(c) for c in cmd],
                    cwd=str(cwd) if cwd else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except Exception as exc:
                self.q.put(("line", step_id, f"[ERROR] Could not start process: {exc}"))
                ok = False
                break

            self.active_proc = proc
            self.active_proc_step = step_id
            try:
                for line in proc.stdout:
                    self.q.put(("line", step_id, line.rstrip("\n")))
            finally:
                proc.wait()
                self.active_proc = None
                self.active_proc_step = None

            if proc.returncode != 0:
                self.q.put(("line", step_id, f"[FAILED] exit code {proc.returncode}"))
                ok = False
                break

        self.q.put(("finished", step_id, ok, on_success))

    def _report_internal_error(self, what: str, exc: BaseException) -> None:
        """Put an unexpected failure on the console instead of losing it.

        Goes through the queue rather than _append_console because callers
        include worker threads, and Tkinter widgets must only be touched from
        the main thread."""
        detail = "".join(
            traceback.format_exception_only(type(exc), exc)).strip()
        text = f"[ERROR] while {what}: {detail}"
        try:
            self.q.put(("line", "app", text))
        except Exception:                      # queue itself is gone
            print(text, file=sys.stderr, flush=True)

    def _guarded(self, what: str, func: Callable[[], None]) -> Callable[[], None]:
        """Wrap a worker-thread body so a crash is reported, not swallowed.

        A daemon thread that raises dies silently: Python prints the traceback
        to stderr, which nobody reads in a GUI started from RunLab.bat, and the
        feature simply never finishes -- the console keeps its "working..."
        line forever and the operator has no way to tell a slow step from a
        dead one. That is the worst thing that can happen during beam time."""
        def runner() -> None:
            try:
                func()
            except Exception as exc:
                self._report_internal_error(what, exc)
        return runner

    def _poll_queue(self) -> None:
        # Nothing in here may raise. This callback reschedules itself, so an
        # escaping exception does not show up as an error the operator can see
        # -- it stops the pump for good: the window stays responsive, but the
        # console never prints again, steps never report finishing, and chained
        # steps never start. Hence the per-message guard and the finally.
        try:
            while True:
                msg = self.q.get_nowait()
                try:
                    self._handle_queue_message(msg)
                except Exception as exc:
                    self._report_internal_error("processing a pipeline message", exc)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_queue)

    def _handle_queue_message(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "line":
            _, step_id, text = msg
            clean_text = self._format_console_line(step_id, text)
            if clean_text:
                self._append_console(f"[{step_id}] {clean_text}")
                if step_id == "live":
                    self._update_live_from_line(clean_text)
        elif kind == "ui_update":
            _, func = msg
            try:
                func()
            except Exception as exc:
                self._report_internal_error("updating the display", exc)
        elif kind == "finished":
            _, step_id, ok, on_success = msg
            self._busy_step = None
            interrupted = self._interrupt_requested and not ok
            self._interrupt_requested = False
            if ok:
                self.state.mark_success(step_id)
                self._append_console(f"[{step_id}] ✓ done")
            elif interrupted:
                self.state.mark_interrupted(step_id)
                self._append_console(f"[{step_id}] ⏹ interrupted")
            else:
                self.state.mark_failed(step_id)
                self._append_console(f"[{step_id}] ✗ failed")
            if step_id in self.step_rows:
                self._refresh_step_row(step_id)
            if ok and on_success:
                # A follow-up step that raises must not take the pump with it:
                # this used to run unguarded inside _poll_queue, so a failing
                # chain (① -> build model, ③ -> degradation analysis) froze all
                # further console output for the rest of the session.
                on_success()

    def _on_close(self) -> None:
        if self.active_proc is not None:
            if not messagebox.askyesno("Quit", "A process is still running. Quit anyway?"):
                return
            try:
                self.active_proc.terminate()
            except Exception:
                pass
        self.root.destroy()

    # ────────────────────────────────────────────────────────────────────
    #  STEP IMPLEMENTATIONS
    #  Every _step_* method:
    #    1. Reads whatever it needs from Tkinter variables (main thread — safe)
    #    2. Validates inputs, shows a messagebox and returns early on problems
    #    3. Builds a list of argv-lists / pure-python callables
    #    4. Hands them to self._start_step(...) which runs them on a worker thread
    #  All accept on_success=None so they can be chained generically by
    #  _run_chain() / "Run Remaining Steps".
    # ────────────────────────────────────────────────────────────────────

    def _step_clean(self, on_success: Optional[Callable[[], None]] = None) -> None:
        targets = [
            MODEL_YML, DETECTIONS_CSV, DARK_PASS_CSV,
            CODE_DIR / "feature_importances.png",
            CODE_DIR / "confusion_matrix.png",
            CODE_DIR / "prediction_summary.png",
            CODE_DIR / "particle_counts_by_type.png",
            CODE_DIR / "ml_signal_arrival_counts.png",
            CODE_DIR / "results_preview_background_mean.png",
            CODE_DIR / "results_preview_background_sigma.png",
            CODE_DIR / "results_preview_signal_arrival.png",
        ]
        existing = [p for p in targets if p.exists()]
        if not existing:
            messagebox.showinfo("Clean Workspace", "Nothing to clean.")
            return
        names = "\n".join(f" • {p.name}" for p in existing)
        if not messagebox.askyesno(
            "Clean Workspace",
            f"This will permanently delete:\n{names}\n\n"
            f"Note: detections_labeled.csv is NEVER touched — your manual "
            f"labeling work is safe.\n\nContinue?",
        ):
            return

        def _do_clean() -> None:
            for p in existing:
                p.unlink()

        self._start_step("clean", [_do_clean], on_success=on_success)

    def _step_build(self, on_success: Optional[Callable[[], None]] = None) -> None:
        def _mkdir() -> None:
            BUILD_DIR.mkdir(parents=True, exist_ok=True)

        cmake = cmake_executable()
        configure = [cmake, ".."]
        if os.name == "nt":
            # The bundled OpenCV is an MSVC vc16 build with no registry entry, so
            # find_package(OpenCV) cannot locate it unaided on the lab machine.
            configure.append("-DOpenCV_RUNTIME=vc16")
            if OPENCV_DIR is not None:
                configure.append(f"-DOpenCV_DIR={OPENCV_DIR}")
        # `cmake --build` rather than `make`: there is no make on Windows, and
        # MSVC's generator is multi-config so the config must be named here.
        compile_cmd = [cmake, "--build", ".", "--config", "Release", "--parallel", "4"]
        self._start_step("build", [_mkdir, configure, compile_cmd],
                         cwd=BUILD_DIR, on_success=on_success)

    def _run_exposure_autotune(self) -> None:
        """Run ids_capture --auto-tune and fill the energy-mode GAIN field
        with the AUTO_TUNE_RESULT it reports. It searches GAIN, not exposure:
        a prompt particle deposit's peak amplitude is set by gain, not
        integration time. Requires the IDS camera and the beam ON."""
        if not IDS_CAPTURE_SCRIPT.is_file():
            messagebox.showerror("Auto-tune", f"Script not found:\n{IDS_CAPTURE_SCRIPT}")
            return
        cmd = [
            sys.executable, str(IDS_CAPTURE_SCRIPT), "--auto-tune",
            "--device-index", str(int(self.ids_device_index_var.get())),
            "--target-sat-pixels", str(float(self.autotune_target_sat_var.get())),
        ]
        # Keep exposure fixed during the search at whatever the energy preset
        # says (blank = leave the camera's current value).
        expo_raw = self.ids_energy_exposure_us_var.get().strip()
        if expo_raw:
            cmd += ["--exposure-us", expo_raw]
        self._append_console("[auto_tune] Searching energy-mode GAIN (beam must be ON)…")

        def worker() -> None:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            out_lines = (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines()
            gain_db: Optional[str] = None
            for ln in out_lines:
                if ln.startswith("AUTO_TUNE_RESULT:"):
                    for part in ln.split():
                        if part.startswith("gain_db="):
                            gain_db = part.split("=", 1)[1]

            def apply() -> None:
                for ln in out_lines:
                    self._append_console(f"[auto_tune] {ln}")
                if proc.returncode != 0:
                    self._append_console(f"[auto_tune] FAILED (exit {proc.returncode})")
                elif gain_db is not None:
                    self.ids_energy_gain_db_var.set(gain_db)
                    self.acq_mode_var.set("energy")
                    self._append_console(
                        f"[auto_tune] Energy-mode gain set to {gain_db} dB "
                        "(acquisition mode switched to 'energy'). Rebuild the "
                        "background model at this gain before measuring.")
            self.root.after(0, apply)

        threading.Thread(target=self._guarded("auto-tuning the gain", worker),
                         daemon=True, name="gain-autotune").start()

    def _live_session_dir(self) -> Optional[Path]:
        """Folder of the CURRENT/most recent live experiment.

        One folder per experiment (timestamped at ② start), not per day: an
        'end-of-experiment' report must cover exactly the run that just
        finished, even when several experiments happen the same day. Falls
        back to the newest live_* folder on disk (e.g. after an app restart)."""
        current = getattr(self, "_live_session_path", None)
        if current is not None and Path(current).is_dir():
            return Path(current)
        candidates = sorted((CODE_DIR / "data_analysed").glob("live_*"))
        return candidates[-1] if candidates else None

    def _live_session_csv(self) -> Optional[Path]:
        d = self._live_session_dir()
        return (d / "detections_classified.csv") if d else None

    def _live_save_args(self) -> list[str]:
        """Translate the 'Save during live' choice into engine flags.
        Mints a NEW session folder for this experiment. The running
        classified-clusters CSV is written whenever data_analysed is on —
        a few hundred bytes/cluster, negligible next to the masked PNGs."""
        mode = self.live_save_mode_var.get()
        args: list[str] = []
        if "data_analysed" in mode:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._live_session_path = CODE_DIR / "data_analysed" / f"live_{stamp}"
            args += ["--export-dir", str(self._live_session_path)]
            args += ["--live-csv", str(self._live_session_path / "detections_classified.csv")]
        if "raw" in mode:
            args += ["--record-raw-dir", self.live_record_dir_var.get()]
        return args

    # ── Live session workflow (lab) ──────────────────────────────────────

    def _workflow_background_before(self) -> None:
        """① Beam OFF: capture darks (IDS source), build the background
        model, and keep a 'before' copy for the end-of-run health check.
        With the simulated source, builds from the existing dark folder."""
        def snapshot() -> None:
            try:
                dst = CODE_DIR / "background_model_before.bgm"
                shutil.copyfile(MODEL_YML, dst)
                self.damage_before_model_var.set(str(dst))
                self._append_console(f"[workflow] 'Before' model snapshot -> {dst}")
            except Exception as exc:
                self._append_console(f"[workflow] snapshot failed: {exc}")

        build = lambda: self._step_bg_model(on_success=snapshot)
        if self.live_source_var.get() == "ids":
            self._step_ids_capture("dark", on_success=build)
        else:
            build()

    def _workflow_health_check(self) -> None:
        """③ Beam OFF again: capture FRESH darks into their own folder,
        build an 'after' model from them (separate file — never overwrites
        the live model), then run the degradation analysis against the
        'before' snapshot from step ①."""
        after_dir = DATA_DIR / "dark_after"
        after_model = CODE_DIR / "background_model_after.bgm"

        def analyse() -> None:
            before = Path(self.damage_before_model_var.get())
            if not before.is_file():
                self.damage_before_model_var.set(str(MODEL_YML))
            self.damage_after_model_var.set(str(after_model))
            self._run_degradation_analysis()

        def build() -> None:
            try:
                exe = self._detector_exe()
            except FileNotFoundError as exc:
                messagebox.showerror("Health check", str(exc))
                return
            folder = resolve_image_folder(after_dir) or after_dir
            n = count_frames_in_folder(folder)
            if n < 1:
                messagebox.showerror("Health check", f"No dark frames in:\n{folder}")
                return
            cmd = [str(exe), "--folder", str(folder), "--model", str(after_model),
                   "--csv", str(CODE_DIR / "dark_pass_after.csv"),
                   "--nsigma", str(self.nsigma_var.get()), "--warmup", str(n),
                   "--min-size", str(int(self.min_cluster_size_var.get())),
                   "--min-charge", str(float(self.min_cluster_charge_var.get()))]
            self._start_step("bg_model", [cmd], cwd=BUILD_DIR, on_success=analyse)

        if self.live_source_var.get() == "ids":
            self._step_ids_capture("dark", on_success=build, output_folder=after_dir)
        elif count_frames_in_folder(resolve_image_folder(after_dir) or after_dir) > 0:
            build()
        else:
            messagebox.showinfo(
                "Health check",
                f"Needs fresh beam-OFF dark frames from the camera.\n"
                f"With the simulated source, place them in:\n{after_dir}\nthen press again.")

    def _run_background_sanity(self) -> None:
        """④ Supervisor question: how many events / 'neutrons' does the
        trained model report on pure beam-OFF frames? Classifies
        dark_pass.csv (clusters found while building the background model)
        and prints the false-positive floor per frame. To measure the
        AMBIENT room rate (neighbouring experiments), run ② with the beam
        OFF instead — that includes real ambient particles."""
        if not DARK_PASS_CSV.is_file():
            messagebox.showerror("Background sanity",
                                 f"No dark-pass clusters found:\n{DARK_PASS_CSV}\n\nRun ① first.")
            return
        if not MODEL_BUNDLE.is_file():
            messagebox.showerror("Background sanity",
                                 f"No trained model:\n{MODEL_BUNDLE}\n\nTrain the classifier first.")
            return
        out_csv = CODE_DIR / "dark_pass_classified.csv"
        cmd = [sys.executable, str(ML_CLASSIFIER_SCRIPT), "--predict",
               "--input", str(DARK_PASS_CSV), "--output", str(out_csv),
               "--bundle", str(MODEL_BUNDLE)]
        n_frames = max(1, count_frames_in_folder(
            resolve_image_folder(Path(self.dark_folder_var.get()))
            or Path(self.dark_folder_var.get())))
        self._append_console("[sanity] Classifying beam-OFF clusters…")

        def worker() -> None:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(CODE_DIR))
            lines: list[str] = []
            if proc.returncode != 0:
                lines.append(f"[sanity] FAILED (exit {proc.returncode}): "
                             + (proc.stderr or proc.stdout or "").strip()[-400:])
            else:
                import pandas as pd
                df = pd.read_csv(out_csv)
                counts = df["predicted_label"].value_counts().to_dict()
                fake_neutrons = sum(v for k, v in counts.items()
                                    if k in ("alpha", "lithium"))
                lines.append("[sanity] ── ML false-positive floor on beam-OFF frames ──")
                lines.append(f"[sanity] dark frames: {n_frames}   clusters: {len(df)} "
                             f"({len(df) / n_frames:.2f}/frame)")
                for k, v in sorted(counts.items()):
                    lines.append(f"[sanity]   {k:12s} {v:6d}  ({v / n_frames:.3f}/frame)")
                lines.append(f"[sanity] fake 'neutrons': {fake_neutrons} "
                             f"= {fake_neutrons / n_frames:.3f}/frame — subtract this floor "
                             f"from any beam-ON rate.")
            self.root.after(0, lambda: [self._append_console(ln) for ln in lines])

        threading.Thread(target=self._guarded("running the background sanity check", worker),
                         daemon=True, name="bg-sanity").start()

    # ── Temporal coverage (duty cycle) ───────────────────────────────────

    def _update_coverage_label(self, exposure_us: Optional[float] = None,
                               fps: Optional[float] = None) -> None:
        """Coverage = exposure × fps: the fraction of wall-clock time the
        sensor is actually integrating. Neutrons arriving outside it are
        never counted, so any rate must be divided by this coverage."""
        if not hasattr(self, "live_coverage_var"):
            return
        if self.live_source_var.get() == "simulated" and exposure_us is None:
            self.live_coverage_var.set(
                "Coverage: 100% (simulated replay — every stored frame was a full exposure)")
            self.live_coverage_label.configure(foreground="#2e7d32")
            return
        if exposure_us is None:
            raw = self._active_exposure_gain()[0].strip()
            if not raw:
                self.live_coverage_var.set(
                    "Coverage: unknown (camera-side exposure) — press Query with the camera connected")
                self.live_coverage_label.configure(foreground="#6e6e73")
                return
            try:
                exposure_us = float(raw)
            except ValueError:
                return
        if fps is None:
            self.live_coverage_var.set(
                f"Exposure {exposure_us / 1000:.1f} ms → coverage = exposure × real fps; "
                "press Query for the camera's actual frame rate")
            self.live_coverage_label.configure(foreground="#6e6e73")
            return
        coverage = min(100.0, exposure_us * 1e-6 * fps * 100.0)
        missed = 100.0 - coverage
        txt = (f"Camera integrates {exposure_us / 1000:.1f} ms of every "
               f"{1000.0 / fps:.1f} ms → coverage {coverage:.0f}%")
        if missed >= 1.0:
            txt += f"  ⚠ {missed:.0f}% of neutrons are NOT captured"
            self.live_coverage_label.configure(
                foreground="#c62828" if missed > 20 else "#e65100")
        else:
            self.live_coverage_label.configure(foreground="#2e7d32")
        self.live_coverage_var.set(txt)

    def _query_camera_timing(self) -> None:
        """Read the REAL ExposureTime / resulting frame rate from the IDS
        camera (it may clamp exposure below 1/fps for sensor readout) and
        feed them into the coverage label."""
        cmd = [sys.executable, str(IDS_CAPTURE_SCRIPT), "--query",
               "--device-index", str(int(self.ids_device_index_var.get()))]

        def worker() -> None:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            exposure_us = fps = None
            for ln in (proc.stdout or "").splitlines():
                if ln.startswith("CAMERA_TIMING:"):
                    for part in ln.split():
                        if part.startswith("exposure_us="):
                            exposure_us = float(part.split("=", 1)[1])
                        elif part.startswith("frame_rate="):
                            fps = float(part.split("=", 1)[1])

            def apply() -> None:
                if exposure_us is not None and fps is not None:
                    self._append_console(
                        f"[camera] exposure={exposure_us:.0f} µs  frame_rate={fps:.3f} fps")
                    self._update_coverage_label(exposure_us, fps)
                else:
                    detail = (proc.stderr or proc.stdout or "").strip()[-300:]
                    messagebox.showerror("Camera timing",
                                         f"Could not query the camera:\n{detail}")
            self.root.after(0, apply)

        threading.Thread(target=self._guarded("querying the camera timing", worker),
                         daemon=True, name="camera-query").start()

    def _active_exposure_gain(self) -> tuple[str, str]:
        """Exposure/gain strings of the selected acquisition preset.

        "energy" keeps heavy tracks below saturation (deposited-energy
        spectrum usable); "sensitivity" is the full-exposure behaviour."""
        if self.acq_mode_var.get() == "energy":
            return (self.ids_energy_exposure_us_var.get(),
                    self.ids_energy_gain_db_var.get())
        return self.ids_exposure_us_var.get(), self.ids_gain_db_var.get()

    def _ids_optional_float(self, label: str, raw_value: str) -> Optional[float]:
        raw_value = raw_value.strip()
        if not raw_value:
            return None
        try:
            return float(raw_value)
        except ValueError:
            messagebox.showerror("IDS Camera Capture", f"{label} must be a number or blank.")
            return None

    def _step_ids_list_devices(self) -> None:
        if not IDS_CAPTURE_SCRIPT.is_file():
            messagebox.showerror("IDS Camera Capture", f"Script not found:\n{IDS_CAPTURE_SCRIPT}")
            return
        cmd = [sys.executable, str(IDS_CAPTURE_SCRIPT), "--list-devices"]
        self._start_step("ids_list", [cmd], cwd=SRC_DIR)

    def _step_ids_capture(self, target: str,
                          on_success: Optional[Callable[[], None]] = None,
                          output_folder: Optional[Path] = None) -> None:
        if not IDS_CAPTURE_SCRIPT.is_file():
            messagebox.showerror("IDS Camera Capture", f"Script not found:\n{IDS_CAPTURE_SCRIPT}")
            return

        if output_folder is None:
            target_folder_var = self.dark_folder_var if target == "dark" else self.signal_folder_var
            output_folder = Path(target_folder_var.get()).expanduser()
        if not output_folder:
            messagebox.showerror("IDS Camera Capture", "Choose an output folder first.")
            return

        exposure_raw, gain_raw = self._active_exposure_gain()
        exposure_us = self._ids_optional_float("Exposure us", exposure_raw)
        if exposure_raw.strip() and exposure_us is None:
            return
        gain_db = self._ids_optional_float("Gain dB", gain_raw)
        if gain_raw.strip() and gain_db is None:
            return

        count = int(self.ids_frame_count_var.get())
        if count < 1:
            messagebox.showerror("IDS Camera Capture", "Frames to capture must be at least 1.")
            return

        cmd = [
            sys.executable, str(IDS_CAPTURE_SCRIPT),
            "--output", str(output_folder),
            "--count", str(count),
            "--device-index", str(int(self.ids_device_index_var.get())),
            "--format", self.ids_format_var.get(),
        ]
        if exposure_us is not None:
            cmd += ["--exposure-us", str(exposure_us)]
        if gain_db is not None:
            cmd += ["--gain-db", str(gain_db)]
        if self.ids_clear_output_var.get():
            cmd.append("--clear-output")

        self._start_step(f"ids_capture_{target}", [cmd], cwd=SRC_DIR, on_success=on_success)

    def _step_live_detection(self, source: str) -> None:
        if not LIVE_DETECTION_SCRIPT.is_file():
            messagebox.showerror("Live Detection", f"Script not found:\n{LIVE_DETECTION_SCRIPT}")
            return
        try:
            exe = self._detector_exe()
        except FileNotFoundError as exc:
            messagebox.showerror("Live Detection", str(exc))
            return
        if not MODEL_YML.is_file():
            # Pressing ② first is what a new operator does on a fresh kit: the
            # model is BUILT, never shipped, so it cannot exist yet. The old
            # message said "Run Background Model first", which names a step
            # appearing twice in this GUI -- here as ①, and as "3 · Background
            # Model" in the pipeline tab -- without saying which button to
            # press. Offer to run it instead of leaving the operator to guess.
            #
            # Deliberately NOT chained into ② afterwards: ① needs the beam OFF
            # and ② needs it ON, so a human has to act in between.
            how = ("The camera will record fresh dark frames, then build the "
                   "model from them."
                   if source == "ids" else
                   "The model will be built from the dark frames already in the "
                   "dark folder.")
            if messagebox.askyesno(
                "Live Detection — no background model yet",
                f"There is no background model:\n{MODEL_YML}\n\n"
                "It is built on this machine, never shipped, so a fresh kit "
                "never has one. It measures the sensor's own noise, which "
                "every later step subtracts.\n\n"
                f"Run ① Background model (beam OFF) now?\n{how}\n\n"
                "MAKE SURE THE BEAM IS OFF first. When it finishes, turn the "
                "beam back on and press ② again.",
            ):
                self._workflow_background_before()
            return

        # The live engine classifies with the ONNX export (+ its .meta.json
        # sidecar), not the .joblib bundle. The Classify step writes
        # model_bundle.onnx / model_bundle.meta.json next to the .joblib, so
        # derive those from whatever bundle the user selected.
        bundle_path = Path(self.live_bundle_var.get()).expanduser()
        onnx_path = bundle_path.with_suffix(".onnx")
        meta_path = bundle_path.with_suffix(".meta.json")
        if not onnx_path.is_file() or not meta_path.is_file():
            messagebox.showerror(
                "Live Detection",
                f"No ONNX model found next to the bundle:\n{onnx_path}\n"
                f"(expected sidecar {meta_path.name})\n\n"
                "Run the 'Classify' step first — it exports model_bundle.onnx "
                "and model_bundle.meta.json alongside model_bundle.joblib.",
            )
            return

        cmd = [
            sys.executable, str(LIVE_DETECTION_SCRIPT),
            "--camera", source,
            "--detector", str(exe),
            "--model", str(MODEL_YML),
            "--onnx", str(onnx_path),
            "--work-dir", str(LIVE_WORK_DIR),
            "--duration-s", str(float(self.live_duration_var.get())),
            "--batch-seconds", str(float(self.live_batch_seconds_var.get())),
            "--batch-max-frames", str(int(self.live_batch_max_frames_var.get())),
            "--nsigma", str(float(self.nsigma_var.get())),
            "--cluster-gap", str(int(self.cluster_gap_var.get())),
            "--cluster-grow", str(int(self.cluster_grow_var.get())),
            "--cluster-grow-min-snr", str(float(self.cluster_grow_min_snr_var.get())),
            "--min-size", str(int(self.min_cluster_size_var.get())),
            "--min-charge", str(float(self.min_cluster_charge_var.get())),
            "--min-peak-snr", str(float(self.min_peak_snr_var.get())),
            *self._live_save_args(),
            "--hot-pixel-consecutive-frames",
            str(int(self.live_hot_pixel_consecutive_frames_var.get())),
            "--hot-pixel-brightness-threshold",
            str(float(self.live_hot_pixel_brightness_threshold_var.get())),
        ]
        if source == "simulated":
            sim_folder = resolve_image_folder(Path(self.live_folder_var.get()))
            if sim_folder is None:
                messagebox.showerror(
                    "Live Detection",
                    f"No image folder found under:\n{self.live_folder_var.get()}",
                )
                return
            cmd += ["--folder", str(sim_folder), "--fps", str(float(self.live_fps_var.get()))]
        elif source == "ids":
            cmd += ["--device-index", str(int(self.ids_device_index_var.get()))]
            exposure_raw, gain_raw = self._active_exposure_gain()
            exposure_us = self._ids_optional_float("Exposure us", exposure_raw)
            if exposure_raw.strip() and exposure_us is None:
                return
            gain_db = self._ids_optional_float("Gain dB", gain_raw)
            if gain_raw.strip() and gain_db is None:
                return
            if exposure_us is not None:
                cmd += ["--exposure-us", str(exposure_us)]
            if gain_db is not None:
                cmd += ["--gain-db", str(gain_db)]
        else:
            messagebox.showerror("Live Detection", f"Unknown live source: {source}")
            return

        self._clear_live_graph()
        self.nb.select(self.tab_live)
        self._start_step("live", [cmd], cwd=CODE_DIR)

    def _step_bg_model(self, on_success: Optional[Callable[[], None]] = None) -> None:
        dark_folder = Path(self.dark_folder_var.get())
        dark_folder = resolve_image_folder(dark_folder) or dark_folder
        if not dark_folder.is_dir():
            # A fresh kit ships with no frames, so this is the first thing a new
            # user hits -- and "not found" alone reads like a broken install
            # rather than "you have not recorded anything yet". Say what to do.
            messagebox.showerror(
                "Background Model",
                f"Dark folder not found:\n{dark_folder}\n\n"
                "This folder holds the DARK frames: images recorded with the "
                "camera running but the neutron beam OFF. They measure the "
                "sensor's own noise, which every later step subtracts.\n\n"
                "You have two ways to fill it:\n"
                "  - record them now: Live tab -> set 'Record raw frames to' to "
                "this folder, run with the beam off;\n"
                "  - or point the 'Dark frames folder' field above at a folder "
                "of frames you already have.\n\n"
                "Accepted formats: .tiff, .tif, .bmp, .png.")
            return
        try:
            exe = self._detector_exe()
        except FileNotFoundError as exc:
            messagebox.showerror("Background Model", str(exc))
            return

        # An EXISTING but EMPTY dark folder used to sail straight through: the
        # detector was handed warmup=0, processed nothing, exited 0, and the
        # step reported "done" -- leaving a background model built from no data
        # at all, which every later step then subtracts. Refuse instead.
        n_frames = count_frames_in_folder(dark_folder)
        if n_frames == 0:
            messagebox.showerror(
                "Background Model",
                f"No image frames in the dark folder:\n{dark_folder}\n\n"
                "The folder exists but holds no .tiff/.tif/.bmp/.png files, so "
                "there is nothing to build a noise model from. Building one "
                "anyway would produce a model of nothing, which every later "
                "step would then subtract.\n\n"
                "Record dark frames first (Live tab, ① Background model, beam "
                "OFF), or point the 'Dark frames folder' field at a folder that "
                "already holds some.")
            return

        if self.auto_warmup_var.get():
            self.warmup_var.set(n_frames)  # main thread — safe
            self._append_console(
                f"[bg_model] Auto-detected {n_frames} frames in dark folder "
                f"-> warmup={n_frames}")
            warmup = n_frames
        else:
            warmup = int(self.warmup_var.get())

        n_sigma = self.nsigma_var.get()
        cluster_gap = int(self.cluster_gap_var.get())
        cluster_grow = int(self.cluster_grow_var.get())
        cluster_grow_min_snr = float(self.cluster_grow_min_snr_var.get())
        cmd = [
            str(exe), "--folder", str(dark_folder), "--model", str(MODEL_YML),
            "--csv", str(DARK_PASS_CSV), "--nsigma", str(n_sigma), "--warmup", str(warmup),
            "--cluster-gap", str(cluster_gap),
            "--cluster-grow", str(cluster_grow),
            "--cluster-grow-min-snr", str(cluster_grow_min_snr),
            "--min-size", str(int(self.min_cluster_size_var.get())),
            "--min-charge", str(float(self.min_cluster_charge_var.get())),
        ]
        dark_export_dir = self.dark_export_dir_var.get().strip()
        if dark_export_dir:
            cmd += ["--export-dir", dark_export_dir]
        self._start_step(
            "bg_model", [cmd], cwd=BUILD_DIR,
            on_success=_chain(self._load_results_preview, on_success),
        )

    def _step_detect_signal(self, on_success: Optional[Callable[[], None]] = None) -> None:
        signal_folder = Path(self.signal_folder_var.get())
        signal_folder = resolve_image_folder(signal_folder) or signal_folder
        if not signal_folder.is_dir():
            messagebox.showerror("Detector", f"Signal folder not found:\n{signal_folder}")
            return
        if not MODEL_YML.is_file():
            if not messagebox.askyesno(
                "Detector",
                "No background_model.yml found — run 'Background Model (dark)' "
                "first.\nContinue anyway?",
            ):
                return
        try:
            exe = self._detector_exe()
        except FileNotFoundError as exc:
            messagebox.showerror("Detector", str(exc))
            return

        n_sigma = self.nsigma_var.get()
        cluster_gap = int(self.cluster_gap_var.get())
        cluster_grow = int(self.cluster_grow_var.get())
        cluster_grow_min_snr = float(self.cluster_grow_min_snr_var.get())
        cmd = [
            str(exe), "--folder", str(signal_folder), "--model", str(MODEL_YML),
            "--csv", str(DETECTIONS_CSV), "--resume", "--nsigma", str(n_sigma),
            "--cluster-gap", str(cluster_gap),
            "--cluster-grow", str(cluster_grow),
            "--cluster-grow-min-snr", str(cluster_grow_min_snr),
            "--min-size", str(int(self.min_cluster_size_var.get())),
            "--min-charge", str(float(self.min_cluster_charge_var.get())),
        ]
        signal_export_dir = self.signal_export_dir_var.get().strip()
        if signal_export_dir:
            cmd += ["--export-dir", signal_export_dir]
        self._start_step(
            "detect_signal", [cmd], cwd=BUILD_DIR,
            on_success=_chain(self._load_results_preview, on_success),
        )


    def _step_label(self, on_success: Optional[Callable[[], None]] = None) -> None:
        if not DETECTIONS_CSV.is_file():
            messagebox.showerror("Labeling UI", f"No detections CSV found:\n{DETECTIONS_CSV}")
            return
        signal_folder = resolve_image_folder(Path(self.signal_folder_var.get()))
        if signal_folder is None:
            messagebox.showerror(
                "Labeling UI",
                f"No image folder found under:\n{self.signal_folder_var.get()}",
            )
            return
        if pd is None or np is None:
            messagebox.showerror(
                "Labeling UI",
                "pandas and numpy are required to build the annotation sample.\n"
                "Install them with:  pip install pandas numpy",
            )
            return

        # ── Read sampling parameters on the MAIN thread (Tkinter vars) ────
        # (same rule every other step follows — see the threading-model note
        #  at the top of this file — the worker thread below only ever sees
        #  plain Python values captured in this closure.)
        n_samples = int(self.num_clusters_var.get())
        min_size  = int(self.baseline_min_size_var.get())
        seed      = int(self.sample_seed_var.get())
        force_new = bool(self.force_resample_var.get())

        if n_samples < 1:
            messagebox.showerror("Labeling UI", "Clusters to annotate must be at least 1.")
            return

        sample_exists = DETECTIONS_SAMPLE_CSV.is_file()

        # ── Force-resample: confirm before nuking any in-progress labels ──
        if sample_exists and force_new:
            n_labeled = 0
            try:
                existing = pd.read_csv(DETECTIONS_SAMPLE_CSV)
                if "true_label" in existing.columns:
                    n_labeled = int((existing["true_label"].fillna("") != "").sum())
            except Exception:
                pass
            progress_note = (
                f"({n_labeled} clusters already labeled in it.)\n\n" if n_labeled
                else "\n"
            )
            if not messagebox.askyesno(
                "Force New Sample",
                f"A sample already exists:\n  {DETECTIONS_SAMPLE_CSV.name}\n"
                f"{progress_note}"
                f"Force-resampling draws a brand-new random subset and discards "
                f"that progress. Continue?",
            ):
                return
            sample_exists = False   # fall through to (re)generation below

        def _build_sample() -> None:
            df = pd.read_csv(DETECTIONS_CSV)
            if "size_px" not in df.columns:
                raise ValueError("detections.csv has no 'size_px' column — cannot sample.")
            sample = sample_clusters_for_annotation(df, n_samples, min_size, seed)
            sample.to_csv(DETECTIONS_SAMPLE_CSV, index=False)
            self.q.put(("line", "label",
                        f"Sampled {len(sample):,} / {len(df):,} clusters "
                        f"(size_px > {min_size}, seed={seed}) -> "
                        f"{DETECTIONS_SAMPLE_CSV.name}"))

        commands: list = []
        if sample_exists:
            # Reuse as-is: this is what makes a restarted annotation session
            # resume the *same* clusters (including any labels already made)
            # instead of silently drawing a different subset.
            self._append_console(
                f"[label] Reusing existing sample: {DETECTIONS_SAMPLE_CSV.name}  "
                f"(tick 'Force a new sample' in the Parameters tab to redraw it)"
            )
        else:
            commands.append(_build_sample)

        label_cmd = [
            sys.executable, str(LABELING_UI_SCRIPT),
            "--csv", str(DETECTIONS_SAMPLE_CSV), "--folder", str(signal_folder),
            # The sample above is already the unbiased set (random draw above
            # the noise floor) — disable the UI's own size/SNR filters so it
            # doesn't re-introduce the exact bias this sampling step exists
            # to avoid by silently hiding some of the sampled clusters.
            "--min-size", "0", "--min-snr", "0",
        ]
        # A finished sample has no unlabelled rows, so the tool would open on an
        # empty queue and say "all 0 qualifying clusters have been labelled" with
        # no way in. Open it in review mode instead, which shows every cluster and
        # changes nothing until something is clicked.
        if _sample_is_fully_labelled(DETECTIONS_SAMPLE_CSV):
            label_cmd.append("--review")
        commands.append(label_cmd)

        self._start_step("label", commands, cwd=SRC_DIR, on_success=on_success)

    def _step_classify(self, on_success: Optional[Callable[[], None]] = None) -> None:
        mode = self.classify_mode_var.get()
        if mode == "train":
            if DETECTIONS_LABELED_CSV.is_file():
                labeled = DETECTIONS_LABELED_CSV
            elif DETECTIONS_SAMPLE_CSV.is_file():
                # The annotation sample IS the labeled set once you've been
                # through it in the Labeling UI (it carries a true_label column).
                labeled = DETECTIONS_SAMPLE_CSV
            else:
                labeled = DETECTIONS_CSV
            if not labeled.is_file():
                messagebox.showerror("ML Classifier", f"No labeled CSV found:\n{labeled}")
                return
            cmd = [sys.executable, str(ML_CLASSIFIER_SCRIPT),
                   "--train", "--labeled", str(labeled),
                   "--bundle", str(MODEL_BUNDLE)]
        else:
            if not DETECTIONS_CSV.is_file():
                messagebox.showerror("ML Classifier", f"No detections CSV found:\n{DETECTIONS_CSV}")
                return
            cmd = [sys.executable, str(ML_CLASSIFIER_SCRIPT),
                   "--predict", "--input", str(DETECTIONS_CSV),
                   "--output", str(DETECTIONS_CLASSIFIED_CSV),
                   "--bundle", str(MODEL_BUNDLE)]

        self._start_step(
            "classify", [cmd], cwd=CODE_DIR,
            on_success=_chain(self._refresh_model_quality,
                               self._load_results_preview, on_success),
        )

    def _step_backtest(self, on_success: Optional[Callable[[], None]] = None) -> None:
        if self.classify_mode_var.get() == "predict":
            # Defense in depth: the Step 7 button is greyed out whenever Step 6
            # is set to Predict (see _update_backtest_availability), but guard
            # here too in case this is ever invoked programmatically (e.g. a
            # stale "Run Remaining Steps" chain built before the mode changed).
            messagebox.showwarning(
                "Backtest",
                "Step 6 is currently set to Predict (scoring unlabeled production "
                "data), so a backtest can't be run against it.\n\n"
                "Switch Step 6 to Train in the Parameters tab, or use the Step 6 "
                "'Train' button, then try Backtest again.",
            )
            return

        golden_path = Path(self.golden_dataset_var.get().strip())
        if not golden_path.is_file():
            messagebox.showerror(
                "Backtest",
                f"No golden dataset found:\n{golden_path}\n\n"
                "Set 'Golden dataset (true_label)' in the Parameters tab, "
                "under Backtesting.",
            )
            return
        if not MODEL_BUNDLE.is_file():
            messagebox.showerror(
                "Backtest",
                f"No trained model bundle found:\n{MODEL_BUNDLE}\n\n"
                "Train the ML Classifier (Step 6 · Train) first.",
            )
            return

        cmd = [
            sys.executable, str(BACKTEST_REPORT_SCRIPT),
            "--golden", str(golden_path),
            "--bundle", str(MODEL_BUNDLE),
            "--output", str(BACKTEST_REPORT_TXT),
        ]
        self._start_step("backtest", [cmd], cwd=SRC_DIR, on_success=on_success)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _kit_venv_python() -> Optional[Path]:
    """The kit's own interpreter, if this checkout has a provisioned venv."""
    for rel in (("venv", "Scripts", "python.exe"), ("venv", "bin", "python")):
        p = CODE_DIR.joinpath(*rel)
        if p.is_file():
            return p
    return None


def _refuse_wrong_interpreter() -> None:
    """Stop if the kit has a venv and we are not the interpreter inside it.

    Every child process this GUI spawns is launched with sys.executable, so the
    interpreter that starts the GUI is the one that runs ids_capture.py,
    ml_classifier.py and the rest. Start it with the system Python -- which is
    what happens when VS Code's interpreter selector points there, or when
    someone runs `python PipelineController.py` by hand instead of RunLab.bat --
    and every camera step fails with "IDS peak Python bindings are not
    installed", because the bindings live in the kit's venv and nowhere else.
    The message blamed the bindings; the cause was the launch. Refuse early and
    say which interpreter to use.

    Silent when no kit venv exists: that is a developer checkout, where running
    against any environment with the dependencies is legitimate.
    """
    venv_py = _kit_venv_python()
    if venv_py is None:
        return
    here = Path(sys.executable).resolve()
    if here == venv_py.resolve() or venv_py.parent.resolve() == here.parent:
        return

    launcher = "RunLab.bat" if os.name == "nt" else "RunLab.sh"
    message = (
        "This is not the interpreter the lab kit was provisioned with.\n\n"
        f"Running:   {here}\n"
        f"Expected:  {venv_py}\n\n"
        "Every step this window launches -- camera capture, live acquisition, "
        "classification -- runs with the SAME interpreter that started it. The "
        "IDS camera bindings are installed only in the kit's environment, so "
        "starting from anywhere else makes every camera step fail with "
        "\"IDS peak Python bindings are not installed\", which names the "
        "bindings but not the cause.\n\n"
        f"Start it with:\n    {venv_py} PipelineController.py\n"
        f"or simply double-click {launcher}.\n\n"
        "If you are using VS Code, its interpreter selector is most likely "
        "pointing at the system Python -- change it to the path above."
    )
    print("[PipelineController] " + message.replace("\n\n", "\n"),
          file=sys.stderr, flush=True)
    try:
        warn = tk.Tk()
        warn.withdraw()
        messagebox.showerror("Wrong Python environment", message)
        warn.destroy()
    except Exception:
        pass          # no display: the stderr message above is the whole story
    raise SystemExit(2)


def main() -> None:
    _refuse_wrong_interpreter()
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 2.0)  # HiDPI / Retina on macOS
    except Exception:
        pass
    PipelineControllerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
