"""
labeling_ui.py  –  Cluster labeling tool (Anki-style)  v6
==========================================================
New in v6:
  - Live class-distribution panel (sidebar bottom): count + % + proportional
    bar per label class; updates instantly on every label action.
  - LRU image cache: bounded to MAX_IMG_CACHE frames (default 20) so RAM
    stays flat even on 10 000-frame runs with 20 MP uint16 images.

Fixes vs v4:
  - SyntaxError: global declaration moved to top of main()
  - Raw intensity panel shows original ADU values (anti-CLAHE confusion)
  - Interactive filters (Min Size + Min SNR) with sliders + entry boxes
  - Grayscale colorbar calibrated to 0-255 display range
  - Correct 20-column CSV mapping (filter_status column included)
  - CLI args passed directly to LabelingApp (no global mutation)

Dependencies : pip install pandas pillow numpy
Usage        : python labeling_ui.py [--csv PATH] [--folder PATH] [--zoom N] [--context N]
               python labeling_ui.py --min-size 3 --min-snr 8.0
               python labeling_ui.py --cache-size 30
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
import re
import sys
import tkinter as tk
from collections import OrderedDict
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageTk

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


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT PATHS  (edit these two lines to match your project layout)
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR    = Path(__file__).resolve().parent
DEFAULT_CSV    = _SCRIPT_DIR.parent / "detections.csv"   # <project_root>/detections.csv
DEFAULT_FOLDER = _SCRIPT_DIR.parent / "data"             # <project_root>/data/

# ══════════════════════════════════════════════════════════════════════════════
#  TUNABLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS  = (".png", ".bmp", ".tiff", ".tif")
TRUE_LABEL_COL    = "true_label"

# 20-column CSV layout produced by the C++ detector (must match csvHeader())
# Columns 18-20 are strings; everything else is numeric.
CSV_STRING_COLS   = {"label", "confidence", "filter_status", TRUE_LABEL_COL}

DEFAULT_ZOOM      = 10      # integer pixel zoom factor (NEAREST)
DEFAULT_CONTEXT   = 25      # border around bbox in source pixels
MAX_PREVIEW_PX    = 520     # max screen size of the zoomed crop

DEFAULT_MIN_SIZE  = 2       # px  (filters ~15 000 single-pixel noise hits)
DEFAULT_MIN_SNR   = 5.0     # SNR units
MAX_IMG_CACHE     = 20      # max frames kept in RAM (LRU eviction)

# ── Colour theme: true black on white, Retina-friendly ───────────────────────
T: dict[str, str] = {
    "bg":              "#ffffff",
    "bg_panel":        "#f5f5f7",
    "bg_img":          "#e0e0e0",
    "text":            "#000000",   # true black
    "text_muted":      "#6e6e73",
    "text_value":      "#000000",   # true black
    "accent":          "#0071e3",
    "ok":              "#1a7f37",   # green  – good saturation
    "warn":            "#9a6700",   # amber  – medium saturation
    "danger":          "#b91c1c",   # red    – near-saturated
    "border":          "#d2d2d7",
    "status_bg":       "#f0f0f0",
    "status_fg":       "#6e6e73",
    "prog_trough":     "#d2d2d7",
    # macOS-native clickable label colours (tk.Button ignores bg on macOS Aqua)
    "btn_alpha":     "#e63946",   # rouge vif   – Alpha (He²⁺, fort dépôt)
    "btn_lithium":   "#f4a261",   # orange      – Li (recul noyau lourd)
    "btn_beta":      "#2a9d8f",   # vert-sarcelle – Beta (électron rapide)
    "btn_gamma":     "#457b9d",   # bleu ardoise  – Gamma / X (photon)
    "btn_muon":      "#6f42c1",   # violet      – Muon (rayon cosmique)
    "btn_artifact":  "#adb5bd",   # gris clair  – Artefact électronique
    "btn_neutral":   "#555555",
}

# ── Label definitions: (csv_value, display_text, hotkey, T-colour-key) ───────
# Hotkeys 1-6 couvrent les 6 catégories physiques.
LABELS: list[tuple[str, str, str, str]] = [
    ("alpha",    "α  Alpha",   "1", "btn_alpha"),
    ("lithium",  "Li Lithium", "2", "btn_lithium"),
    ("beta",     "β  Beta",    "3", "btn_beta"),
    ("gamma",    "γ  Gamma",   "4", "btn_gamma"),
    ("muon",     "μ  Muon",    "5", "btn_muon"),
    ("artifact", "✕  Artefact","6", "btn_artifact"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  PURE UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def natural_sort_key(name: str) -> list:
    """'img_9' < 'img_10' ordering."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load detections.csv.  Handles the 20-column layout from the C++ pipeline:
      cols 1-17  : numeric
      col  18    : label   (string)
      col  19    : confidence (float)
      col  20    : filter_status (string: 'valid' | 'rejected')
      col  21    : true_label   (string, added by this tool)
    Forces numeric columns to float so stray strings never silently corrupt data.
    """
    df = pd.read_csv(csv_path)

    # Add true_label column if first run
    if TRUE_LABEL_COL not in df.columns:
        df[TRUE_LABEL_COL] = ""
    df[TRUE_LABEL_COL] = df[TRUE_LABEL_COL].fillna("").astype(str)

    # Force numeric columns to float (guards against column-shift bugs)
    numeric_cols = [c for c in df.columns if c not in CSV_STRING_COLS]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def list_images(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: natural_sort_key(p.name),
    )


def parse_frame_id_from_filename(path: Path) -> Optional[int]:
    """
    Extract the experiment frame number encoded in an image filename.

    Mirrors main.cpp's frameIdFromFilenameOrFallback(): use the first block of
    digits in the filename stem. Examples:
      images_numero_0539.tiff -> 539
      img_539.bmp             -> 539
      image_539_2026-...bmp   -> 539

    Return None when the filename has no digits; build_frame_image_index()
    then applies the same sequential fallback index that the C++ backend uses.
    """
    match = re.search(r"\d+", path.stem)
    return int(match.group(0)) if match else None


def build_frame_image_index(candidates: list[Path]) -> tuple[dict[int, Path], dict[int, list[Path]], list[Path]]:
    """
    Build a direct lookup from parsed frame number to image path.

    Returns:
      - frame_index: unique frame_id -> image path
      - duplicates: frame_id -> all paths with that same parsed id
      - fallback_paths: files where no integer was found, so the natural-sort
        position was used as the frame_id for backward compatibility
    """
    grouped: dict[int, list[Path]] = {}
    fallback_paths: list[Path] = []

    for sequence_idx, path in enumerate(candidates):
        frame_id = parse_frame_id_from_filename(path)
        if frame_id is None:
            frame_id = sequence_idx
            fallback_paths.append(path)
        grouped.setdefault(frame_id, []).append(path)

    duplicates = {
        frame_id: paths
        for frame_id, paths in grouped.items()
        if len(paths) > 1
    }
    frame_index = {
        frame_id: paths[0]
        for frame_id, paths in grouped.items()
        if len(paths) == 1
    }
    return frame_index, duplicates, fallback_paths


def find_image_for_frame(frame_index: dict[int, Path], frame_id: int) -> tuple[Optional[Path], bool]:
    """
    Resolve the source image for `frame_id`.

    `frame_id` is the real experiment frame number stored in detections.csv.
    It may be 50-149 even when the folder contains only 100 files, so lookup
    must use the parsed filename number, not the image's position in a list.

    Returns (path_or_None, missing). `missing` is True when the CSV references
    a frame number that is not present in the parsed filename index.
    """
    path = frame_index.get(frame_id)
    return path, path is None


def preflight_check_sample_images(csv_path: Path, img_folder: Path) -> None:
    """
    Verify — BEFORE the Tk UI ever opens — that every frame referenced in
    the annotation sample CSV resolves to an image whose filename encodes
    that same frame number. This supports folders whose 100 files may span
    real experiment frames 50-149 rather than positional indices 0-99.

    Exits the process with a non-zero code and a diagnostic message if any
    referenced frame is missing, rather than letting the UI open and fail
    one row at a time.
    """
    if not csv_path.is_file():
        print(f"[ERROR] Annotation sample CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[ERROR] Could not read {csv_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    if "image_id" not in df.columns:
        print(f"[ERROR] '{csv_path.name}' has no 'image_id' column — "
              f"cannot preflight-check image availability.", file=sys.stderr)
        sys.exit(1)

    candidates = list_images(img_folder)
    if not candidates:
        print(f"[ERROR] No image files found in {img_folder}", file=sys.stderr)
        sys.exit(1)

    frame_index, duplicates, fallback_paths = build_frame_image_index(candidates)
    if duplicates:
        preview = [
            f"{frame_id}: {', '.join(p.name for p in paths[:3])}"
            for frame_id, paths in sorted(duplicates.items())[:10]
        ]
        print(
            f"[ERROR] {len(duplicates)} duplicate parsed frame id(s) found in {img_folder}:\n"
            f"        " + "\n        ".join(preview) +
            ("\n        ..." if len(duplicates) > 10 else "") +
            "\n        Rename or separate duplicate frames before labeling so "
            "each CSV image_id maps to exactly one file.",
            file=sys.stderr,
        )
        sys.exit(1)

    if fallback_paths:
        preview = [p.name for p in fallback_paths[:10]]
        print(
            f"[WARN] No digits found in {len(fallback_paths)} image filename(s); "
            f"using natural-sort fallback indices for those files: "
            f"{preview}{' ...' if len(fallback_paths) > 10 else ''}",
            file=sys.stderr,
        )

    # One conversion, used for both the list of frames wanted and the count of
    # those reachable, so the two cannot disagree about a type. A CSV whose
    # image_id is text -- Excel on the laboratory machine writes such files --
    # used to raise ValueError here with a traceback instead of a message.
    ids = pd.to_numeric(df["image_id"], errors="coerce").dropna().astype(int)
    if ids.empty:
        print(f"[ERROR] No usable frame number in the 'image_id' column of "
              f"{csv_path.name}.", file=sys.stderr)
        sys.exit(1)
    needed_frames = sorted(set(ids.tolist()))
    available_frames = set(frame_index)
    missing_frames = [f for f in needed_frames if f not in available_frames]

    print(f"[preflight] {len(needed_frames)} distinct frame(s) referenced in "
          f"{csv_path.name} (range {needed_frames[0]}\u2013{needed_frames[-1]}); "
          f"{len(candidates)} image file(s), {len(available_frames)} parsed frame id(s) "
          f"found in {img_folder}.")

    if missing_frames:
        preview = missing_frames[:15]
        available_min = min(available_frames) if available_frames else "none"
        available_max = max(available_frames) if available_frames else "none"
        print(
            f"[ERROR] {len(missing_frames)} / {len(needed_frames)} referenced frame(s) "
            f"are not present as parsed filename frame ids:\n"
            f"        {preview}{' ...' if len(missing_frames) > 15 else ''}\n"
            f"        Available parsed frame range: {available_min}\u2013{available_max}.\n"
            f"        Point --folder at the image folder used for detection, or rename/copy "
            f"the files so their filenames contain the real frame numbers from image_id.",
            file=sys.stderr,
        )
        # A partial folder is the shipped kit's normal state, not an error: it
        # carries twenty frames of a sample drawn from fifteen hundred, and the
        # README says so. Refusing to start on that made the labelling step
        # unreachable for anyone working from the kit. Only a folder that
        # resolves nothing at all is useless.
        reachable = int(ids.isin(available_frames).sum())
        if reachable == 0:
            sys.exit(1)
        print(f"[WARN] continuing: {reachable} of {len(df)} clusters are reachable "
              f"in this folder; the rest sit in frames it does not carry.",
              file=sys.stderr)
        return

    print(f"[preflight] \u2713 All {len(needed_frames)} referenced frame(s) resolve by "
          f"filename frame id. Safe to open the Labeling UI.")


def open_image_raw(path: Path) -> Optional[np.ndarray]:
    """
    Opens an image and returns a 2-D NumPy array with the ORIGINAL pixel values.
    - 8-bit  → uint8   (0-255)
    - 16-bit → uint16  (0-65535)
    - Colour → luminance average across channels
    No normalisation is applied here; that happens only in to_uint8_display().
    """
    try:
        pil = Image.open(path)
    except Exception as exc:
        print(f"[WARN] Cannot open {path}: {exc}", file=sys.stderr)
        return None

    arr = np.array(pil)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)   # colour → grayscale luminance
    return arr   # dtype preserved: uint8 or uint16


def to_uint8_display(arr: np.ndarray) -> np.ndarray:
    """Global min-max stretch to uint8 for display purposes only."""
    if arr.dtype == np.uint8:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        stretched = (arr.astype(np.float32) - lo) / (hi - lo) * 255.0
        return np.clip(stretched, 0, 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def apply_clahe(arr_u8: np.ndarray) -> np.ndarray:
    """
    CLAHE for local contrast enhancement.
    Falls back to global min-max stretch if OpenCV is unavailable.
    """
    try:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        return clahe.apply(arr_u8)
    except ImportError:
        lo, hi = int(arr_u8.min()), int(arr_u8.max())
        if hi > lo:
            return ((arr_u8.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        return arr_u8


def dtype_max(arr: np.ndarray) -> float:
    """Returns 65535 for uint16, 255 otherwise."""
    return 65535.0 if arr.dtype == np.uint16 else 255.0


def compute_bbox_raw_stats(full_raw: np.ndarray,
                            bx: int, by: int, bw: int, bh: int) -> dict:
    """Extract raw ADU statistics strictly within the cluster bounding box."""
    H, W = full_raw.shape
    y0, y1 = max(0, by),      min(H, by + bh)
    x0, x1 = max(0, bx),      min(W, bx + bw)
    patch   = full_raw[y0:y1, x0:x1]
    dmax    = dtype_max(full_raw)
    if patch.size == 0:
        return {"raw_min": 0, "raw_max": 0, "raw_mean": 0.0,
                "dtype_max": dmax, "sat_pct": 0.0}
    rmin  = float(patch.min())
    rmax  = float(patch.max())
    rmean = float(patch.mean())
    return {
        "raw_min":   rmin,
        "raw_max":   rmax,
        "raw_mean":  rmean,
        "dtype_max": dmax,
        "sat_pct":   rmax / dmax * 100.0,
    }


def crop_and_enhance(full_raw: np.ndarray,
                     row: pd.Series,
                     context: int,
                     zoom: int) -> tuple[Optional[Image.Image], dict]:
    """
    Returns (zoomed_PIL_RGB_image, raw_stats_dict).
    The image has CLAHE applied for visibility.
    raw_stats contains the un-normalised ADU values inside the bbox.
    """
    H, W = full_raw.shape

    bx = int(row["bbox_x"])
    by = int(row["bbox_y"])
    bw = max(1, int(row["bbox_w"]))
    bh = max(1, int(row["bbox_h"]))

    # Stats on the raw bbox BEFORE any processing
    raw_stats = compute_bbox_raw_stats(full_raw, bx, by, bw, bh)

    # Crop with context border, clamped to image edges
    cx  = max(0, bx - context)
    cy  = max(0, by - context)
    cx2 = min(W, bx + bw + context)
    cy2 = min(H, by + bh + context)

    if cx2 <= cx or cy2 <= cy:
        return None, raw_stats

    patch     = full_raw[cy:cy2, cx:cx2]
    patch_u8  = to_uint8_display(patch)
    enhanced  = apply_clahe(patch_u8)

    # Zoom with NEAREST to preserve single-pixel cluster structure
    pil  = Image.fromarray(enhanced, mode="L")
    pil  = pil.resize((pil.width * zoom, pil.height * zoom), Image.NEAREST)
    pil  = pil.convert("RGB")
    draw = ImageDraw.Draw(pil)

    # Bounding-box overlay. Drawn as a crisp, single-width outline pinned to
    # exact pixel-grid boundaries (no crosshair — it added clutter without
    # information the box itself doesn't already convey, and it obscured
    # faint pixels right at the cluster centroid, which is often exactly
    # what a reviewer needs to see clearly).
    rx   = (bx - cx) * zoom
    ry   = (by - cy) * zoom
    rx2  = rx + bw * zoom
    ry2  = ry + bh * zoom
    lw   = max(1, zoom // 5)
    # Inset by half the line width so the stroke sits just outside the true
    # bbox edge instead of straddling it — reads as a sharper, cleaner box.
    inset = lw // 2
    draw.rectangle(
        [rx - inset, ry - inset, rx2 + inset, ry2 + inset],
        outline="#e63946", width=lw,
    )

    # Scale down if needed
    if pil.width > MAX_PREVIEW_PX or pil.height > MAX_PREVIEW_PX:
        scale = min(MAX_PREVIEW_PX / pil.width, MAX_PREVIEW_PX / pil.height)
        pil   = pil.resize(
            (int(pil.width * scale), int(pil.height * scale)),
            Image.NEAREST,
        )

    return pil, raw_stats


def make_colorbar_image(width: int, height: int = 38) -> Image.Image:
    """
    Horizontal grayscale colorbar: black (0) → white (255).
    Includes tick marks at 0, 64, 128, 192, 255 with their values.
    """
    BAR_H  = 18
    TICK_H = 4
    TICKS  = [0, 64, 128, 192, 255]

    img  = Image.new("RGB", (width, height), (245, 245, 247))
    draw = ImageDraw.Draw(img)

    # Gradient
    for x in range(width):
        g = int(x / max(width - 1, 1) * 255)
        draw.line([(x, 0), (x, BAR_H - 1)], fill=(g, g, g))
    draw.rectangle([0, 0, width - 1, BAR_H - 1], outline=(180, 180, 180))

    # Ticks and labels
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for v in TICKS:
        x = int(v / 255 * (width - 1))
        draw.line([(x, BAR_H), (x, BAR_H + TICK_H)], fill=(80, 80, 80))
        label = str(v)
        tw    = len(label) * 6   # approximate char width
        tx    = max(0, min(x - tw // 2, width - tw - 1))
        draw.text((tx, BAR_H + TICK_H + 1), label,
                  fill=(80, 80, 80), font=font)

    return img


# ══════════════════════════════════════════════════════════════════════════════
#  macOS-COMPATIBLE COLOURED BUTTON
# ══════════════════════════════════════════════════════════════════════════════

def _make_colour_button(parent: tk.Widget,
                        text: str,
                        bg_colour: str,
                        command,
                        font,
                        padx: int = 16,
                        pady: int = 9) -> tk.Label:
    """
    Returns a tk.Label that looks and behaves like a coloured button.

    Background: on macOS with the Aqua theme, tk.Button completely ignores
    the `bg` option — buttons always render with the system chrome.
    tk.Label *does* respect `bg`, so we fake a button with:
      • coloured background + white text
      • hover effect (slightly darkened bg) via <Enter>/<Leave>
      • click feedback via <ButtonPress>/<ButtonRelease>
      • keyboard activation is handled separately by _bind_keys()
    """
    def _darken(hex_col: str, factor: float = 0.80) -> str:
        """Return a darker version of a #rrggbb colour."""
        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b = int(hex_col[5:7], 16)
        return "#{:02x}{:02x}{:02x}".format(
            int(r * factor), int(g * factor), int(b * factor))

    dark_col = _darken(bg_colour)

    lbl = tk.Label(
        parent,
        text=text,
        bg=bg_colour,
        fg="#ffffff",
        font=font,
        padx=padx,
        pady=pady,
        cursor="hand2",
        relief="flat",
    )

    # Hover: darken background
    lbl.bind("<Enter>",        lambda _e: lbl.configure(bg=dark_col))
    lbl.bind("<Leave>",        lambda _e: lbl.configure(bg=bg_colour))
    # Click: darken more on press, restore on release, then call command
    lbl.bind("<ButtonPress-1>",   lambda _e: lbl.configure(bg=_darken(bg_colour, 0.65)))
    lbl.bind("<ButtonRelease-1>", lambda _e: (lbl.configure(bg=dark_col), command()))

    return lbl


def _make_neutral_button(parent: tk.Widget,
                         text: str,
                         command,
                         font,
                         padx: int = 16,
                         pady: int = 9) -> tk.Label:
    """Neutral (grey-bordered) version of the macOS-compatible button."""
    bg     = T["bg"]
    hover  = T["border"]

    lbl = tk.Label(
        parent,
        text=text,
        bg=bg,
        fg=T["btn_neutral"],
        font=font,
        padx=padx,
        pady=pady,
        cursor="hand2",
        relief="solid",
        bd=1,
        highlightbackground=T["border"],
        highlightthickness=1,
    )
    lbl.bind("<Enter>",           lambda _e: lbl.configure(bg=hover))
    lbl.bind("<Leave>",           lambda _e: lbl.configure(bg=bg))
    lbl.bind("<ButtonPress-1>",   lambda _e: lbl.configure(bg=T["text_muted"]))
    lbl.bind("<ButtonRelease-1>", lambda _e: (lbl.configure(bg=hover), command()))
    return lbl


# ══════════════════════════════════════════════════════════════════════════════
#  LRU IMAGE CACHE
# ══════════════════════════════════════════════════════════════════════════════

class LRUCache:
    """
    Bounded LRU cache for raw image arrays (frame_id → np.ndarray | None).

    Without a cap, caching every frame of a 10 000-frame run at 20 MP × 2 B
    (uint16) would consume ~400 GB.  With cap=20 we hold at most ~800 MB and
    still get effectively 100 % hit rate for the typical labeling pattern
    (reviewer stays within a narrow frame window).

    Implementation: OrderedDict gives O(1) get / put / evict.
    """

    def __init__(self, maxsize: int = MAX_IMG_CACHE) -> None:
        self._maxsize = max(1, maxsize)
        self._data: OrderedDict[int, Optional[np.ndarray]] = OrderedDict()

    def get(self, key: int) -> Optional[np.ndarray]:
        """Return cached array and mark as recently used, or raise KeyError."""
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: int, value: Optional[np.ndarray]) -> None:
        """Insert / update entry, evicting the LRU entry if over capacity."""
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)   # evict least-recently-used

    def __contains__(self, key: int) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION CLASS
# ══════════════════════════════════════════════════════════════════════════════

class LabelingApp:
    """Anki-style cluster labeling GUI."""

    def __init__(self,
                 root: tk.Tk,
                 csv_path: Path,
                 img_folder: Path,
                 zoom: int,
                 context: int,
                 init_min_size: int,
                 init_min_snr: float,
                 review: bool = False) -> None:

        self.review      = review
        self.root        = root
        self.csv_path    = csv_path
        self.img_folder  = img_folder
        self.zoom        = zoom
        self.context     = context

        self.df          = load_csv(csv_path)
        self.candidates  = list_images(img_folder)
        self.frame_index, self.duplicate_frames, self.fallback_images = build_frame_image_index(
            self.candidates
        )
        # Frame-level image cache: frame_id → raw numpy array (never normalised)
        self._img_cache: dict[int, Optional[np.ndarray]] = {}

        # Filter state (tk variables so sliders and entries stay in sync)
        self._min_size_var = tk.IntVar(value=init_min_size)
        self._min_snr_var  = tk.DoubleVar(value=init_min_snr)

        # Will be built by _rebuild_queue()
        self._queue: list[int] = []   # list of CSV row indices that pass filters
        self._cursor: int = 0
        # Undo history: (csv_row_idx, previous_label_value, cursor_position)
        self._undo_stack: list[tuple[int, str, int]] = []

        self._build_ui()
        self._bind_keys()
        self._rebuild_queue()
        self._show_current()

    # ══════════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        self.root.title("Cluster Labeling — Particle Detector  v5")
        self.root.configure(bg=T["bg"])
        self.root.resizable(True, True)

        # Font definitions (TkDefaultFont scales correctly on Retina/HiDPI)
        F_TITLE = ("TkDefaultFont", 14, "bold")
        F_GROUP = ("TkDefaultFont",  9, "bold")
        F_LABEL = ("TkDefaultFont", 11)
        F_VALUE = ("TkDefaultFont", 11, "bold")
        F_SMALL = ("TkDefaultFont",  9)
        F_BTN   = ("TkDefaultFont", 12, "bold")
        F_PCT   = ("TkDefaultFont", 13, "bold")
        F_ID    = ("TkDefaultFont", 15, "bold")
        F_MONO  = ("TkFixedFont",   10)
        F_MONO_B= ("TkFixedFont",   10, "bold")

        # ── Progress band ─────────────────────────────────────────────────
        prog_band = tk.Frame(self.root, bg=T["bg_panel"])
        prog_band.pack(fill="x")

        prog_inner = tk.Frame(prog_band, bg=T["bg_panel"])
        prog_inner.pack(fill="x", padx=16, pady=8)

        self._prog_count_var = tk.StringVar(value="Loading…")
        tk.Label(prog_inner, textvariable=self._prog_count_var,
                 bg=T["bg_panel"], fg=T["text_muted"],
                 font=F_SMALL, anchor="w").pack(side="left")

        self._prog_pct_var = tk.StringVar(value="0 %")
        tk.Label(prog_inner, textvariable=self._prog_pct_var,
                 bg=T["bg_panel"], fg=T["accent"],
                 font=F_PCT, width=6, anchor="e").pack(side="right")

        sty = ttk.Style(self.root)
        sty.theme_use("default")
        sty.configure("Prog.Horizontal.TProgressbar",
                       troughcolor=T["prog_trough"],
                       background=T["accent"],
                       thickness=12, borderwidth=0)
        self._progress_bar = ttk.Progressbar(
            prog_inner, style="Prog.Horizontal.TProgressbar",
            mode="determinate", length=420)
        self._progress_bar.pack(side="left", padx=(12, 12))

        tk.Frame(self.root, bg=T["border"], height=1).pack(fill="x")

        # ── Central zone: image column (left) + sidebar (right) ───────────
        center = tk.Frame(self.root, bg=T["bg"])
        center.pack(fill="both", expand=True, padx=16, pady=12)

        # ────────────────────── LEFT COLUMN ──────────────────────────────
        left_col = tk.Frame(center, bg=T["bg"])
        left_col.pack(side="left", anchor="n", padx=(0, 18))

        # Frame + Cluster ID banner
        id_frame = tk.Frame(left_col, bg=T["bg_panel"],
                            highlightbackground=T["border"], highlightthickness=1)
        id_frame.pack(fill="x", pady=(0, 6))

        self._id_var = tk.StringVar(value="Frame —  ·  Cluster —")
        tk.Label(id_frame, textvariable=self._id_var,
                 bg=T["bg_panel"], fg=T["accent"],
                 font=F_ID, anchor="center", padx=10, pady=6).pack(fill="x")

        # Image display
        img_outer = tk.Frame(left_col, bg=T["bg_img"],
                             highlightbackground=T["border"], highlightthickness=1)
        img_outer.pack()

        self._img_label = tk.Label(img_outer, bg=T["bg_img"],
                                   text="Loading…", fg=T["text_muted"],
                                   font=F_LABEL, width=44, height=22)
        self._img_label.pack(padx=2, pady=2)

        # Colorbar
        cb_outer = tk.Frame(left_col, bg=T["bg"])
        cb_outer.pack(fill="x", pady=(5, 0))

        tk.Label(cb_outer, text="Display reference (CLAHE 0–255):",
                 bg=T["bg"], fg=T["text_muted"], font=F_SMALL,
                 anchor="w").pack(anchor="w")

        self._cb_canvas = tk.Canvas(cb_outer, height=38, bg=T["bg"],
                                    highlightthickness=0)
        self._cb_canvas.pack(fill="x")
        self._cb_photo: Optional[ImageTk.PhotoImage] = None
        self._cb_canvas.bind("<Configure>", self._redraw_colorbar)

        # Raw intensity panel
        raw_outer = tk.Frame(left_col, bg=T["bg_panel"],
                             highlightbackground=T["border"], highlightthickness=1)
        raw_outer.pack(fill="x", pady=(10, 0))

        tk.Label(raw_outer, text="RAW PIXEL INTENSITY (bbox, original ADU)",
                 bg=T["bg_panel"], fg=T["text_muted"],
                 font=F_GROUP, anchor="w", padx=8, pady=4).pack(fill="x")

        # Min / Max / Mean row
        trio = tk.Frame(raw_outer, bg=T["bg_panel"])
        trio.pack(fill="x", padx=8, pady=(0, 4))

        self._raw_vars: dict[str, tk.StringVar] = {}
        for key, lbl in [("raw_min", "Min"), ("raw_max", "Max"), ("raw_mean", "Mean")]:
            cell = tk.Frame(trio, bg=T["bg_panel"])
            cell.pack(side="left", expand=True)
            tk.Label(cell, text=lbl, bg=T["bg_panel"],
                     fg=T["text_muted"], font=F_SMALL).pack()
            var = tk.StringVar(value="—")
            self._raw_vars[key] = var
            tk.Label(cell, textvariable=var, bg=T["bg_panel"],
                     fg=T["text_value"], font=F_MONO_B).pack()

        # Saturation row
        sat_row = tk.Frame(raw_outer, bg=T["bg_panel"])
        sat_row.pack(fill="x", padx=8, pady=(0, 6))

        tk.Label(sat_row, text="Sat. (max/range):",
                 bg=T["bg_panel"], fg=T["text_muted"],
                 font=F_SMALL, anchor="w").pack(side="left")

        self._sat_var   = tk.StringVar(value="—")
        self._sat_label = tk.Label(sat_row, textvariable=self._sat_var,
                                   bg=T["bg_panel"], fg=T["ok"],
                                   font=F_MONO_B, anchor="w")
        self._sat_label.pack(side="left", padx=6)

        # Dtype range (shown once)
        self._range_var = tk.StringVar(value="")
        tk.Label(sat_row, textvariable=self._range_var,
                 bg=T["bg_panel"], fg=T["text_muted"],
                 font=F_SMALL, anchor="w").pack(side="left")

        # Range label (Raw Range: min–max ADU)
        range_row = tk.Frame(raw_outer, bg=T["bg_panel"])
        range_row.pack(fill="x", padx=8, pady=(0, 4))

        tk.Label(range_row, text="Raw range in bbox:",
                 bg=T["bg_panel"], fg=T["text_muted"],
                 font=F_SMALL, anchor="w").pack(side="left")

        self._raw_range_var = tk.StringVar(value="—")
        tk.Label(range_row, textvariable=self._raw_range_var,
                 bg=T["bg_panel"], fg=T["text_value"],
                 font=F_MONO_B, anchor="w").pack(side="left", padx=6)

        # ────────────────────── RIGHT COLUMN (sidebar) ────────────────────
        right_col = tk.Frame(center, bg=T["bg"], width=340)
        right_col.pack(side="left", fill="y", anchor="n")
        right_col.pack_propagate(False)

        # ── Filter panel ──────────────────────────────────────────────────
        filt_box = tk.Frame(right_col, bg=T["bg_panel"],
                            highlightbackground=T["accent"], highlightthickness=1)
        filt_box.pack(fill="x", pady=(0, 12))

        tk.Label(filt_box, text="⚙  REAL-TIME FILTERS",
                 bg=T["bg_panel"], fg=T["accent"],
                 font=F_GROUP, anchor="w", padx=8, pady=5).pack(fill="x")

        filt_body = tk.Frame(filt_box, bg=T["bg_panel"])
        filt_body.pack(fill="x", padx=8, pady=(0, 8))

        # Min Size row
        def _filter_row(parent: tk.Frame,
                        label: str,
                        var: tk.Variable,
                        from_: float, to: float,
                        increment: float,
                        fmt: str = "%.0f") -> None:
            row = tk.Frame(parent, bg=T["bg_panel"])
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, width=14, anchor="w",
                     bg=T["bg_panel"], fg=T["text_muted"],
                     font=F_LABEL).pack(side="left")
            entry = tk.Spinbox(
                row, from_=from_, to=to, increment=increment,
                textvariable=var, width=7,
                font=F_VALUE, relief="solid", bd=1, format=fmt,
                command=self._on_filter_change,
            )
            entry.pack(side="left", padx=4)
            entry.bind("<Return>",   lambda _e: self._on_filter_change())
            entry.bind("<FocusOut>", lambda _e: self._on_filter_change())
            tk.Scale(row, from_=from_, to=to, resolution=increment,
                     orient="horizontal", variable=var, length=130,
                     bg=T["bg_panel"], fg=T["text_muted"],
                     highlightthickness=0, troughcolor=T["border"],
                     command=lambda _v: self._on_filter_change(),
                     ).pack(side="left", padx=4)

        _filter_row(filt_body, "Min Size (px):",
                    self._min_size_var, 1, 200, 1)
        _filter_row(filt_body, "Min Peak SNR:",
                    self._min_snr_var, 0.0, 100.0, 0.5, "%.1f")

        self._filt_count_var = tk.StringVar(value="")
        tk.Label(filt_box, textvariable=self._filt_count_var,
                 bg=T["bg_panel"], fg=T["text_muted"],
                 font=F_SMALL, anchor="w", padx=8, pady=3).pack(fill="x")

        # ── Metrics panel ─────────────────────────────────────────────────
        tk.Label(right_col, text="Cluster Metrics",
                 bg=T["bg"], fg=T["accent"],
                 font=F_TITLE, anchor="w").pack(fill="x", pady=(0, 6))

        groups = [
            ("Photometry", [
                ("peak_snr",     "Peak SNR"),
                ("peak_value",   "Peak value"),
                ("total_charge", "Total charge"),
            ]),
            ("Morphology", [
                ("size_px",      "Size (px)"),
                ("elongation",   "Elongation"),
                ("aspect_ratio", "Aspect ratio"),
                ("compactness",  "Compactness"),
                ("sigma_x",      "σ x"),
                ("sigma_y",      "σ y"),
            ]),
            ("Position (source px)", [
                ("center_x", "Centre X"),
                ("center_y", "Centre Y"),
                ("bbox_x",   "Bbox X"),
                ("bbox_y",   "Bbox Y"),
                ("bbox_w",   "Bbox W"),
                ("bbox_h",   "Bbox H"),
            ]),
        ]

        self._metric_vars: dict[str, tk.StringVar] = {}

        for grp_name, fields in groups:
            hdr = tk.Frame(right_col, bg=T["bg_panel"])
            hdr.pack(fill="x", pady=(6, 1))
            tk.Label(hdr, text=grp_name.upper(),
                     bg=T["bg_panel"], fg=T["text_muted"],
                     font=F_GROUP, anchor="w", padx=5, pady=2).pack(fill="x")

            for col, lbl in fields:
                row_f = tk.Frame(right_col, bg=T["bg"])
                row_f.pack(fill="x", pady=1, padx=2)
                tk.Label(row_f, text=f"{lbl}:",
                         width=16, anchor="w",
                         bg=T["bg"], fg=T["text_muted"],
                         font=F_LABEL).pack(side="left")
                var = tk.StringVar(value="—")
                self._metric_vars[col] = var
                tk.Label(row_f, textvariable=var,
                         anchor="w", width=16,
                         bg=T["bg"], fg=T["text_value"],
                         font=F_VALUE).pack(side="left")

        # ── Separator + label buttons ──────────────────────────────────────
        tk.Frame(self.root, bg=T["border"], height=1).pack(fill="x")

        btn_band = tk.Frame(self.root, bg=T["bg_panel"])
        btn_band.pack(fill="x", pady=10)

        btn_row = tk.Frame(btn_band, bg=T["bg_panel"])
        btn_row.pack()

        for value, text, hotkey, colour_key in LABELS:
            btn = _make_colour_button(
                btn_row,
                text=f"[{hotkey}]  {text}",
                bg_colour=T[colour_key],
                command=lambda v=value: self._label(v),
                font=F_BTN,
                padx=16, pady=9,
            )
            btn.pack(side="left", padx=5)

        _make_neutral_button(
            btn_row, text="[S]  Skip",
            command=self._skip, font=F_BTN,
            padx=16, pady=9,
        ).pack(side="left", padx=5)

        _make_neutral_button(
            btn_row, text="[←]  Previous",
            command=self._prev, font=F_BTN,
            padx=16, pady=9,
        ).pack(side="left", padx=5)

        _make_neutral_button(
            btn_row, text="[U]  Undo label",
            command=self._undo, font=F_BTN,
            padx=16, pady=9,
        ).pack(side="left", padx=5)

        # ── Status bar ────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready")
        self._status_label = tk.Label(
            self.root, textvariable=self._status_var,
            bg=T["status_bg"], fg=T["status_fg"],
            font=F_SMALL, anchor="w", padx=8, pady=3,
        )
        self._status_label.pack(fill="x", side="bottom")

    def _raise_window(self) -> None:
        """Make the labeling window visible and frontmost on startup."""
        try:
            self.root.update_idletasks()
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.root.attributes("-topmost", True)
            self.root.after(200, lambda: self.root.attributes("-topmost", False))
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    #  COLORBAR
    # ══════════════════════════════════════════════════════════════════════

    def _redraw_colorbar(self, _event=None) -> None:
        w = self._cb_canvas.winfo_width()
        if w < 16:
            return
        cb_img = make_colorbar_image(w, height=38)
        self._cb_photo = ImageTk.PhotoImage(cb_img)
        self._cb_canvas.delete("all")
        self._cb_canvas.create_image(0, 0, anchor="nw", image=self._cb_photo)

    # ══════════════════════════════════════════════════════════════════════
    #  FILTERING LOGIC
    # ══════════════════════════════════════════════════════════════════════

    def _passes_filter(self, row: pd.Series) -> bool:
        """Return True if this cluster row satisfies the current filter thresholds."""
        try:
            min_size = int(self._min_size_var.get())
            min_snr  = float(self._min_snr_var.get())
        except (tk.TclError, ValueError):
            return True   # invalid filter value → pass everything

        size = row.get("size_px",  0)
        snr  = row.get("peak_snr", 0.0)

        if pd.isna(size) or pd.isna(snr):
            return False
        return int(size) >= min_size and float(snr) >= min_snr

    def _rebuild_queue(self) -> None:
        """
        Rebuild self._queue as the list of unlabelled CSV indices that pass
        the current quality filters.  Preserves the cursor on the currently
        displayed row when possible.
        """
        # Remember which CSV row we're currently on
        current_idx: Optional[int] = None
        if self._queue and self._cursor < len(self._queue):
            current_idx = self._queue[self._cursor]

        # In review mode every row is queued, labelled or not, so a finished
        # set can be reopened and checked. Otherwise only the ones still empty:
        # once a set is fully labelled the normal queue is empty and the tool
        # has nothing to show, which is correct but leaves no way back in.
        if self.review:
            selected_mask = pd.Series(True, index=self.df.index)
            word = "in the set"
        else:
            selected_mask = self.df[TRUE_LABEL_COL] == ""
            word = "unlabelled"
        self._queue = [
            idx for idx, row in self.df[selected_mask].iterrows()
            if self._passes_filter(row)
        ]

        # Update filter summary label
        total_selected = int(selected_mask.sum())
        qualified      = len(self._queue)
        filtered_out   = total_selected - qualified
        self._filt_count_var.set(
            f"{qualified} shown  /  {total_selected} {word}  "
            f"({filtered_out} hidden by filters)"
        )

        # Restore position
        if current_idx is not None and current_idx in self._queue:
            self._cursor = self._queue.index(current_idx)
        else:
            self._cursor = 0

    def _on_filter_change(self) -> None:
        """Called whenever a filter slider or entry changes."""
        self._rebuild_queue()
        self._show_current()

    # ══════════════════════════════════════════════════════════════════════
    #  KEY BINDINGS
    # ══════════════════════════════════════════════════════════════════════

    def _bind_keys(self) -> None:
        for value, _, hotkey, _ in LABELS:
            self.root.bind(hotkey, lambda e, v=value: self._label(v))
        self.root.bind("s", lambda e: self._skip())
        self.root.bind("S", lambda e: self._skip())
        self.root.bind("u", lambda e: self._undo())
        self.root.bind("U", lambda e: self._undo())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Left>",  lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._skip())

    # ══════════════════════════════════════════════════════════════════════
    #  DISPLAY CURRENT CLUSTER
    # ══════════════════════════════════════════════════════════════════════

    def _status_var_color_warn(self) -> None:
        self._status_label.configure(bg=T["danger"], fg="#ffffff")

    def _status_var_color_normal(self) -> None:
        self._status_label.configure(bg=T["status_bg"], fg=T["status_fg"])

    def _show_current(self) -> None:
        total     = len(self._queue)
        remaining = total - self._cursor

        if self._cursor >= total:
            messagebox.showinfo(
                "Done!",
                f"All {total} qualifying clusters have been labelled.\n"
                f"CSV saved to: {self.csv_path}",
            )
            self.root.quit()
            return

        idx = self._queue[self._cursor]
        row = self.df.loc[idx]

        # ── ID banner ─────────────────────────────────────────────────────
        fid = int(row.get("image_id",   -1))
        cid = int(row.get("cluster_id", -1))
        self._id_var.set(f"Frame {fid}  ·  Cluster {cid}")

        # ── Progress ──────────────────────────────────────────────────────
        done = self._cursor
        pct  = int(done / total * 100) if total else 0
        self._progress_bar["maximum"] = total
        self._progress_bar["value"]   = done
        self._prog_pct_var.set(f"{pct} %")
        self._prog_count_var.set(
            f"{done} labelled  /  {total} qualifying  —  {remaining} remaining"
        )

        # ── Metrics ───────────────────────────────────────────────────────
        float_cols = {
            "peak_snr", "peak_value", "total_charge",
            "elongation", "aspect_ratio", "compactness",
            "sigma_x", "sigma_y", "center_x", "center_y",
        }
        for col, var in self._metric_vars.items():
            val = row.get(col, None)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                var.set("—")
            elif col in float_cols:
                var.set(f"{float(val):.3f}")
            else:
                var.set(str(int(val)) if isinstance(val, float) else str(val))

        # ── Image + raw stats ─────────────────────────────────────────────
        img_path, frame_mismatch = find_image_for_frame(self.frame_index, fid)
        pil_img:  Optional[Image.Image] = None
        raw_stats: dict = {}

        if img_path is not None:
            if fid not in self._img_cache:
                self._img_cache[fid] = open_image_raw(img_path)
            full_raw = self._img_cache[fid]
            if full_raw is not None:
                pil_img, raw_stats = crop_and_enhance(
                    full_raw, row, self.context, self.zoom)

        if pil_img is not None:
            self._tk_img = ImageTk.PhotoImage(pil_img)
            self._img_label.configure(
                image=self._tk_img, text="",
                width=pil_img.width, height=pil_img.height)
        else:
            self._tk_img = None
            msg = "Image not found" if img_path is None else "Load error"
            self._img_label.configure(image="", text=msg, width=44, height=22)

        # ── Update raw intensity panel ─────────────────────────────────────
        if raw_stats:
            dmax  = raw_stats["dtype_max"]
            is16  = dmax > 255
            fmt   = (lambda v: f"{v:.1f}") if is16 else (lambda v: str(int(round(v))))

            self._raw_vars["raw_min"].set(fmt(raw_stats["raw_min"]))
            self._raw_vars["raw_max"].set(fmt(raw_stats["raw_max"]))
            self._raw_vars["raw_mean"].set(f"{raw_stats['raw_mean']:.1f}")

            rmin = raw_stats["raw_min"]
            rmax = raw_stats["raw_max"]
            self._raw_range_var.set(
                f"{fmt(rmin)} – {fmt(rmax)} ADU"
                + (f"  (scale 0–{int(dmax)})" if is16 else "")
            )

            sat = raw_stats["sat_pct"]
            self._sat_var.set(f"{sat:.1f} %")
            # Colour-code saturation level
            sat_colour = T["ok"] if sat < 70 else T["warn"] if sat < 95 else T["danger"]
            self._sat_label.configure(fg=sat_colour)
            self._range_var.set(f"(sensor max = {int(dmax)} ADU)")
        else:
            for var in self._raw_vars.values():
                var.set("—")
            self._raw_range_var.set("—")
            self._sat_var.set("—")
            self._range_var.set("")

        # Redraw colorbar now that canvas width is known
        self._redraw_colorbar()

        # ── Status bar ────────────────────────────────────────────────────
        fname = img_path.name if img_path else "N/A"
        mismatch_note = ""
        if frame_mismatch:
            mismatch_note = (
                f"  |  \u26a0 frame {fid} is OUT OF RANGE for this folder "
                f"({len(self.frame_index)} parsed frame ids from {len(self.candidates)} images) "
                f"— folder likely doesn't match what the detector ran against"
            )
        self._status_var.set(
            f"  CSV row: {idx}  |  image_id: {fid}  "
            f"|  cluster_id: {cid}  |  file: {fname}{mismatch_note}"
        )
        if frame_mismatch:
            self._status_var_color_warn()
        else:
            self._status_var_color_normal()
        self.root.update_idletasks()

    # ══════════════════════════════════════════════════════════════════════
    #  ACTIONS
    # ══════════════════════════════════════════════════════════════════════

    def _label(self, value: str) -> None:
        if self._cursor >= len(self._queue):
            return
        idx = self._queue[self._cursor]
        # Record for undo BEFORE mutating: labelling pops the row from the
        # queue (which only holds unlabelled rows), so without this history
        # a mislabelled cluster becomes unreachable — "Previous" only walks
        # the remaining UNLABELLED rows and can never revisit it.
        self._undo_stack.append((idx, str(self.df.at[idx, TRUE_LABEL_COL]),
                                 self._cursor))
        self.df.at[idx, TRUE_LABEL_COL] = value
        self._save()
        # Remove from queue (row now labelled); cursor stays → points to next
        self._queue.pop(self._cursor)
        if self._cursor >= len(self._queue):
            self._cursor = max(0, len(self._queue) - 1)
        self._show_current()

    def _undo(self) -> None:
        """Revert the most recent labelling: restore the previous value,
        put the row back into the queue at its old position, and show it
        again so it can be re-labelled correctly."""
        if not self._undo_stack:
            self._status_var.set("  Nothing to undo")
            return
        idx, old_value, old_cursor = self._undo_stack.pop()
        self.df.at[idx, TRUE_LABEL_COL] = old_value
        self._save()
        pos = min(old_cursor, len(self._queue))
        self._queue.insert(pos, idx)
        self._cursor = pos
        self._show_current()
        self._status_var.set("  ↩  Label undone — re-label this cluster")

    def _skip(self) -> None:
        self._cursor = min(self._cursor + 1, len(self._queue))
        self._show_current()

    def _prev(self) -> None:
        if self._cursor > 0:
            self._cursor -= 1
            self._show_current()

    def _save(self) -> None:
        try:
            self._snapshot_once()
            self.df.to_csv(self.csv_path, index=False)
            self._status_var.set(f"  ✓  Saved → {self.csv_path}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def _snapshot_once(self) -> None:
        """Copy the CSV as it was before this session ever wrote to it.

        Every verdict overwrites the file in place, so a session that revises
        earlier judgements destroys the labels the published numbers were
        computed from -- silently, because the file is not under version
        control. One copy per session, taken before the first write, is enough
        to get them back: the labels the thesis reports and the labels a later
        sitting produced are both a physics judgement, and which one is right
        is not this tool's call to make.
        """
        if getattr(self, "_snapshotted", False):
            return
        self._snapshotted = True          # set first: a failed copy must not
                                          # retry on every subsequent verdict
        try:
            if not self.csv_path.is_file():
                return
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            dest = self.csv_path.parent / "annotation_versions"
            dest.mkdir(exist_ok=True)
            shutil.copy2(self.csv_path, dest / f"{self.csv_path.stem}_avant_{stamp}.csv")
        except Exception:
            pass                          # a missing backup must never block
                                          # the labelling itself


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # NOTE: ALL global mutations must be declared before any use of the variable.
    # Here we avoid the anti-pattern entirely by passing values as arguments.
    parser = argparse.ArgumentParser(
        description="Cluster labeling GUI — Particle Detector v5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",      type=Path,  default=DEFAULT_CSV,
                        help="Path to detections.csv")
    parser.add_argument("--folder",   type=Path,  default=DEFAULT_FOLDER,
                        help="Folder containing source images")
    parser.add_argument("--zoom",     type=int,   default=DEFAULT_ZOOM,
                        help="Integer pixel zoom factor")
    parser.add_argument("--context",  type=int,   default=DEFAULT_CONTEXT,
                        help="Context border around bbox in source pixels")
    parser.add_argument("--min-size", type=int,   default=DEFAULT_MIN_SIZE,
                        help="Initial Min Size filter (px)")
    parser.add_argument("--review", action="store_true",
                        help="show every cluster, including the ones already "
                             "labelled, so a finished set can be checked")
    parser.add_argument("--min-snr",  type=float, default=DEFAULT_MIN_SNR,
                        help="Initial Min Peak SNR filter")
    args = parser.parse_args()

    # ── Validate paths ────────────────────────────────────────────────────
    errors: list[str] = []
    if not args.csv.is_file():
        errors.append(f"CSV not found: {args.csv}")
    if not args.folder.is_dir():
        errors.append(f"Image folder not found: {args.folder}")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        print(
            "\nEdit DEFAULT_CSV / DEFAULT_FOLDER at the top of the script, "
            "or pass --csv / --folder.",
            file=sys.stderr,
        )
        sys.exit(1)

    imgs = list_images(args.folder)
    print(f"[OK] CSV    : {args.csv}")
    print(f"[OK] Images : {args.folder}  ({len(imgs)} files)")
    print(f"[OK] Zoom ×{args.zoom},  context {args.context} px")
    print(f"[OK] Filters: min_size≥{args.min_size} px,  min_snr≥{args.min_snr}")

    # ── Pre-flight: verify every sampled frame resolves to a real image ───
    # before any Tk window is created, so a folder/CSV mismatch surfaces as
    # one clear startup report instead of scattered failures during review.
    preflight_check_sample_images(args.csv, args.folder)

    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 2.0)   # HiDPI / Retina on macOS
    except Exception:
        pass

    # Pass all config as constructor arguments — no global mutation needed
    app = LabelingApp(
        root        = root,
        csv_path    = args.csv,
        img_folder  = args.folder,
        zoom        = args.zoom,
        context     = args.context,
        init_min_size = args.min_size,
        init_min_snr  = args.min_snr,
        review        = args.review,
    )
    root.after_idle(app._raise_window)
    root.mainloop()


if __name__ == "__main__":
    main()
