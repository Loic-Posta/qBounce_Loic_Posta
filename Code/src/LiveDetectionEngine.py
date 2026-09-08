#!/usr/bin/env python3
"""
live_detection_engine.py – bridges a live camera feed to particle
classification results, for the PipelineController "Live Acquisition" mode.

Pipeline per micro-batch:
    camera.get_frame() x K  ->  scratch folder  ->  existing compiled
    C++ detector (subprocess, --resume against the shared background
    model)  ->  detections.csv for that batch  ->  ml_classifier.predict_single()
    per cluster  ->  results pushed onto a thread-safe queue.Queue for the
    GUI to poll.

Why go through the EXISTING compiled detector in small batches, instead of
a pure-Python port of BackgroundModel / ClusterDetector?
    This project's background-subtraction and clustering logic lives in
    BackgroundModel.cpp / ClusterDetector.cpp. This engine reuses the
    already-correct, already-compiled `detector` binary so live results
    come from the exact same algorithm as the offline pipeline, instead
    of a second, independently-written implementation that could quietly
    drift from it. The cost is latency: results arrive in whole-batch
    chunks (BATCH_SECONDS / BATCH_MAX_FRAMES, whichever comes first), not
    frame-by-frame.

    For lower latency later: add a small persistent "server mode" to the
    C++ side (reuse BackgroundModel/ClusterDetector directly, read frames
    over a pipe, keep state across frames instead of re-launching a
    process per batch) and swap out _process_batch()/_run_detector()
    below. The threading/queue/GUI contract here does not need to change
    either way — that's the intended extension point.

CSV column names:
    CSV_COL_FRAME_ID is confirmed — labeling_ui.py's own docstring names
    it explicitly. CSV_COL_CENTER_X / CSV_COL_CENTER_Y are a best guess
    from the C++ Cluster struct's center_x/center_y members (see
    saveDebugImage() in main.cpp) — Cluster::toCsvLine() itself wasn't
    available to confirm the header. If heatmap points come out empty or
    misplaced, check these two constants first.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import threading
import time
import queue
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import io

import numpy as np
import pandas as pd
from PIL import Image

# When stdout is redirected to a pipe (e.g. PipelineController's
# subprocess.Popen), CPython defaults to full block buffering instead of
# line buffering, so print() output can sit in a buffer until it fills up
# or the process exits — the GUI would then see nothing until the very end
# even though everything "worked". Force line buffering (Python 3.7+) so
# every LIVE_BATCH / status line is flushed to the GUI as soon as it's
# printed. Every print() below also passes flush=True as a second line of
# defense (e.g. on interpreters where reconfigure() isn't available).
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Default labels included in the LIVE_BATCH "counts" dict even when absent.
# The operational live view focuses on alpha/lithium and derives a neutron
# estimate from paired alpha/lithium counts, but keeping the older labels here
# preserves compatibility with existing trained bundles.
STANDARD_CLASSES = ["alpha", "lithium", "particle", "cosmic_ray", "artifact", "hot_pixel"]
ALPHA_LABELS = {"alpha", "alpha_particle", "alpha_particles"}
LITHIUM_LABELS = {"lithium", "li", "li_particle", "li_particles",
                  "lithium_particle", "lithium_particles"}

# camera_interface.py and ml_classifier.py live next to this file in
# Code/src/ — make sure that's on sys.path regardless of how this module
# was imported or launched.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


from camera_interface import CameraInterface, create_camera  # noqa: E402
import json
import numpy as np  # already imported above, re-noted for clarity near ONNX use
import onnxruntime as ort  # noqa: E402

# ml_classifier is kept only for FEATURE_COLS / add_derived_features-style
# column order — no joblib/sklearn import happens on this path anymore.
import ml_classifier  # noqa: E402


# ⚠ See "CSV column names" in the module docstring above.
CSV_COL_FRAME_ID = "image_id"
CSV_COL_CENTER_X = "center_x"
CSV_COL_CENTER_Y = "center_y"

BATCH_SECONDS_DEFAULT = 1.0
BATCH_MAX_FRAMES_DEFAULT = 10


@dataclass
class LiveDetectionConfig:
    detector_exe: Path
    background_model_path: Path
    onnx_path: str = ml_classifier.ONNX_PATH
    frame_queue_maxsize: int = 64
    work_dir: Path = Path("live_scratch")
    batch_seconds: float = BATCH_SECONDS_DEFAULT
    batch_max_frames: int = BATCH_MAX_FRAMES_DEFAULT
    n_sigma: float = 5.0
    cluster_gap: int = 0
    cluster_grow: int = 0
    cluster_grow_min_snr: float = 1.0
    min_size_px: int = 0
    min_charge_adu: float = 0.0
    # Gain-invariant throughput guardrail: drop clusters below this peak SNR.
    # 0 = off. Preferred over min_charge_adu because it survives gain changes.
    min_peak_snr: float = 0.0
    # data_analysed export by the C++ server (masked frames, bg model,
    # arrival heatmap) — handled server-side on its own writer thread.
    export_dir: Optional[str] = None
    # Running CSV of every classified cluster (all physics columns +
    # predicted_label), appended to across the whole session. This is what
    # lets "Energy spectrum (today)" work on a full day of live data.
    live_csv_path: Optional[str] = None
    # Full raw-frame recording (IDS-Cockpit replacement): every captured
    # frame is written as .bmp by a dedicated writer thread. ~20 MB/frame.
    record_raw_dir: Optional[str] = None
    hot_pixel_consecutive_frames: int = 5
    hot_pixel_brightness_threshold: float = 50.0
    detector_timeout_s: float = 30.0

def _load_onnx_session(onnx_path: str) -> tuple[ort.InferenceSession, list[str], list[str], "float | None"]:
    """Load an ONNX model + its .meta.json sidecar (classes, feature order,
    and the saturation the model was trained at — None for legacy models).

    CPUExecutionProvider is explicit here rather than left to default
    provider auto-selection, so behaviour is identical across machines
    that may or may not have a GPU provider installed.
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta_path = Path(onnx_path).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    return session, meta["classes"], meta["feature_cols"], meta.get("train_saturation")

class LiveDetectionEngine:
    """
    Producer/consumer split:
      _capture_loop   -> camera.get_frame() -> frame_queue   (Thread 1)
      _inference_loop -> frame_queue -> detector/classifier -> result_queue (Thread 2)

    Messages pushed onto result_queue (all plain dicts/tuples, no string
    protocol):
        ("batch", {"elapsed_seconds": float, "counts": dict,
                    "particle_summary": dict, "dead_pixels": dict,
                    "detections": list[dict]})
        ("error", str)
    """

    def __init__(self, camera: CameraInterface, config: LiveDetectionConfig,
                 result_queue: "queue.Queue") -> None:
        self.camera = camera
        self.config = config
        self.result_queue = result_queue
        self.frame_queue: "queue.Queue" = queue.Queue(maxsize=config.frame_queue_maxsize)

        self.config.detector_exe = Path(self.config.detector_exe).expanduser().resolve()
        self.config.background_model_path = Path(self.config.background_model_path).expanduser().resolve()
        self.config.work_dir = Path(self.config.work_dir).expanduser().resolve()
        self.config.onnx_path = str(Path(self.config.onnx_path).expanduser().resolve())

        self._session: Optional[ort.InferenceSession] = None
        self._onnx_classes: list[str] = []
        self._onnx_feature_cols: list[str] = []
        self._train_saturation: Optional[float] = None   # from model meta
        # Running live saturation (clusters with peak>=254) vs the model's
        # training saturation, and whether we've already warned about a
        # mismatch (warn once, not every batch).
        self._sat_seen = 0
        self._sat_total = 0
        self._saturation_warned = False
        self._detector_proc: Optional[subprocess.Popen] = None
        self._csv_header: str = ""   # set by _start_detector_server()
        # Resolve BOTH sink paths to absolute here: the detector server runs
        # with cwd = the exe's folder (Code/build), so a relative --export-dir
        # would silently land the whole data_analysed tree inside build/ —
        # exactly where nobody (and no ⑤ report) looks for it.
        if self.config.export_dir:
            self.config.export_dir = str(Path(self.config.export_dir).expanduser().resolve())
        self._live_csv_path: Optional[Path] = None
        if self.config.live_csv_path:
            self._live_csv_path = Path(self.config.live_csv_path).expanduser().resolve()
            self._live_csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._stop_event = threading.Event()
        self.capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._run_start: Optional[float] = None

        self._frame_counter = 0
        self._batch_counter = 0
        self._last_batch_frames = 0
        self._hot_pixel_run_lengths: Optional[np.ndarray] = None
        self._hot_pixel_mask: Optional[np.ndarray] = None

        # ── Throughput accounting ────────────────────────────────────────
        # _frame_counter above counts frames CAPTURED from the camera.
        # _frames_processed counts frames that actually went through the
        # detector; _frames_dropped counts frames discarded because
        # inference could not keep up (drop-oldest in _capture_loop).
        # captured ≈ processed + dropped + (still queued). These numbers
        # exist so the operator SEES when the requested camera fps exceeds
        # what the pipeline can analyse — instead of silently reporting
        # rates computed on a fraction of the frames.
        self._frames_processed = 0
        self._frames_dropped = 0
        self._batch_server_ms: list[float] = []   # per-frame detector time, this batch
        self._last_classify_ms = 0.0

        # ── Raw-frame recorder (optional) ────────────────────────────────
        # Bounded queue + dedicated writer thread so disk I/O never blocks
        # the capture loop. At 2 fps a .bmp write (~40 ms) keeps up easily;
        # frames skipped because the writer lags are counted, not silent.
        self._save_queue: "queue.Queue" = queue.Queue(maxsize=32)
        self._frames_saved = 0
        self._saves_dropped = 0
        self._recorder_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop_event.clear()
        self._run_start = time.monotonic()
        if self.config.record_raw_dir:
            Path(self.config.record_raw_dir).mkdir(parents=True, exist_ok=True)
            self._recorder_thread = threading.Thread(
                target=self._recorder_loop, name="live-recorder", daemon=True)
            self._recorder_thread.start()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="live-capture", daemon=True)
        self._inference_thread = threading.Thread(
            target=self._inference_loop, name="live-inference", daemon=True)
        self._capture_thread.start()
        self._inference_thread.start()

    def _recorder_loop(self) -> None:
        """Writer thread of the raw recorder: .bmp = a plain memory dump,
        the fastest lossless option and what IDS Cockpit produced anyway
        (Mono8 sensor noise barely compresses, so PNG would only cost CPU)."""
        out_dir = Path(self.config.record_raw_dir)
        while not (self._stop_event.is_set() and self._save_queue.empty()):
            try:
                frame_id, frame = self._save_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            try:
                Image.fromarray(frame).save(out_dir / f"live_{frame_id:06d}_{ts}.bmp")
                self._frames_saved += 1
            except Exception as exc:
                self.result_queue.put(("error", f"raw recording failed: {exc}"))
                return

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        for t in (getattr(self, "_capture_thread", None),
                  getattr(self, "_inference_thread", None),
                  getattr(self, "_recorder_thread", None)):
            if t is not None:
                t.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return bool(
            (self._capture_thread and self._capture_thread.is_alive())
            or (self._inference_thread and self._inference_thread.is_alive())
        )

    def _apply_hot_pixel_mask(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Track and mask pixels that remain bright for consecutive frames."""
        arr = np.asarray(frame)
        if arr.ndim == 3:
            brightness = arr.max(axis=2)
            mask_shape = arr.shape[:2]
        else:
            brightness = arr
            mask_shape = arr.shape

        if (
            self._hot_pixel_run_lengths is None
            or self._hot_pixel_run_lengths.shape != mask_shape
        ):
            self._hot_pixel_run_lengths = np.zeros(mask_shape, dtype=np.uint16)
            self._hot_pixel_mask = np.zeros(mask_shape, dtype=bool)

        assert self._hot_pixel_run_lengths is not None
        assert self._hot_pixel_mask is not None

        min_frames = max(1, int(self.config.hot_pixel_consecutive_frames))
        bright = brightness > self.config.hot_pixel_brightness_threshold
        self._hot_pixel_run_lengths[bright] = np.minimum(
            self._hot_pixel_run_lengths[bright] + 1, min_frames)
        self._hot_pixel_run_lengths[~bright] = 0

        self._hot_pixel_mask |= self._hot_pixel_run_lengths >= min_frames

        if not self._hot_pixel_mask.any():
            return arr, self._hot_pixel_mask

        masked = arr.copy()
        if masked.ndim == 3:
            masked[self._hot_pixel_mask, :] = 0
        else:
            masked[self._hot_pixel_mask] = 0
        return masked, self._hot_pixel_mask

    def _hot_pixel_stats(self) -> dict:
        if self._hot_pixel_mask is None:
            return {"count": 0, "total": 0, "pct": 0.0}
        count = int(self._hot_pixel_mask.sum())
        total = int(self._hot_pixel_mask.size)
        pct = (100.0 * count / total) if total else 0.0
        return {"count": count, "total": total, "pct": pct}

    def _particle_summary(self, detections: list[dict], frames_in_batch: int) -> dict:
        alpha = 0
        lithium = 0
        by_label: dict[str, int] = {}
        for item in detections:
            label = _normalise_label(item.get("label", "unknown"))
            by_label[label] = by_label.get(label, 0) + 1
            if label in ALPHA_LABELS:
                alpha += 1
            if label in LITHIUM_LABELS:
                lithium += 1

        frames = max(1, int(frames_in_batch))
        # n + B-10 -> alpha + Li-7, emitted back-to-back in the converter
        # layer: only ONE of the two daughters can reach the sensor, the other
        # is absorbed in the coating (supervisor-confirmed geometry). Every
        # alpha OR lithium track therefore corresponds to one neutron capture:
        #   neutrons = alpha + lithium   (a sum, not a pair-coincidence).
        # Alpha and lithium stay reported separately because their deposited
        # energies differ (1.47 vs 0.84 MeV in the dominant branch), so their
        # relative rates are a live diagnostic of the energy response.
        neutrons = alpha + lithium
        return {
            "frames": frames,
            "alpha": alpha,
            "lithium": lithium,
            "neutrons": neutrons,
            "alpha_per_frame": alpha / frames,
            "lithium_per_frame": lithium / frames,
            "neutrons_per_frame": neutrons / frames,
            "labels": by_label,
        }

    # ── Thread 1: producer ──────────────────────────────────────────────

    def _capture_loop(self) -> None:
        try:
            self.camera.open()
        except Exception as exc:
            self.result_queue.put(("error", f"Could not open camera: {exc}"))
            self._stop_event.set()
            return
        try:
            while not self._stop_event.is_set():
                frame = self.camera.get_frame(timeout_ms=200)
                if frame is None:
                    continue
                self._frame_counter += 1

                # Archive the pristine camera frame (before any masking) so
                # the recording is bit-identical to what the sensor produced.
                if self._recorder_thread is not None:
                    try:
                        self._save_queue.put_nowait((self._frame_counter, frame))
                    except queue.Full:
                        self._saves_dropped += 1

                frame, mask = self._apply_hot_pixel_mask(frame)

                item = (self._frame_counter, frame, mask)
                try:
                    self.frame_queue.put(item, timeout=0.5)
                except queue.Full:
                    # Inference is falling behind: drop the oldest queued
                    # frame instead of blocking capture indefinitely.
                    # Counted so the GUI can report the loss instead of
                    # letting rates silently under-count.
                    try:
                        self.frame_queue.get_nowait()
                        self._frames_dropped += 1
                    except queue.Empty:
                        pass
                    self.frame_queue.put_nowait(item)
        except Exception as exc:
            # Only camera.open() used to report; anything raised mid-run went
            # to the finally, killed this thread, and left no trace -- the run
            # simply stopped acquiring and still exited 0, so the GUI showed a
            # green tick for a measurement that had crashed. Say what happened
            # and after how many frames.
            self.result_queue.put((
                "error",
                f"Acquisition stopped after {self._frame_counter} frame(s): "
                f"{type(exc).__name__}: {exc}"))
            self._stop_event.set()
        finally:
            self.camera.close()
            try:
                self.frame_queue.put_nowait(None)  # sentinel wakes inference
            except queue.Full:
                pass

    # ── Thread 2: consumer ───────────────────────────────────────────────

    def _inference_loop(self) -> None:
        try:
            (self._session, self._onnx_classes, self._onnx_feature_cols,
             self._train_saturation) = _load_onnx_session(self.config.onnx_path)
        except Exception as exc:
            self.result_queue.put(("error", f"Could not load ONNX model: {exc}"))
            self._stop_event.set()
            return

        try:
            self._start_detector_server()
        except Exception as exc:
            self.result_queue.put(("error", f"Could not start detector server: {exc}"))
            self._stop_event.set()
            return

        try:
            while not self._stop_event.is_set():
                batch = self._collect_batch()
                if batch is None:      # capture thread ended (sentinel/shutdown)
                    break
                if not batch:
                    continue
                self._process_batch(batch)
        except Exception as exc:
            # Same hole as the capture thread: only the ONNX/detector startup
            # reported. A failure while detecting or classifying killed this
            # thread quietly, so batches stopped arriving while acquisition
            # carried on -- the GUI's counters simply froze mid-measurement
            # with nothing to explain it.
            self.result_queue.put((
                "error",
                f"Detection stopped: {type(exc).__name__}: {exc}"))
            self._stop_event.set()
        finally:
            self._stop_detector_server()

    def _collect_batch(self) -> Optional[list[tuple[int, np.ndarray, str]]]:
        """Drain frame_queue until batch_seconds or batch_max_frames is hit.
        Each frame is streamed straight to the persistent detector server as
        it's pulled off the queue — no PNGs, no CSV file on disk. The third
        tuple element is that frame's CSV text (already newline-joined),
        returned in place of the old on-disk path."""
        items: list[tuple[int, np.ndarray, str]] = []
        deadline = time.monotonic() + self.config.batch_seconds
        while (
            len(items) < self.config.batch_max_frames
            and time.monotonic() < deadline
            and not self._stop_event.is_set()
        ):
            try:
                queued = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if queued is None:
                self._stop_event.set()
                return items or None
            frame_id, frame, _mask = queued
            t0 = time.monotonic()
            csv_text = self._send_frame_to_server(frame)
            self._batch_server_ms.append((time.monotonic() - t0) * 1000.0)
            self._frames_processed += 1
            items.append((frame_id, frame, csv_text))
        return items

    def _process_batch(self, items: list[tuple[int, np.ndarray, str]]) -> None:
        self._batch_counter += 1
        self._last_batch_frames = len(items)

        csv_text = "\n".join(csv for (_fid, _frame, csv) in items if csv)

        try:
            t0 = time.monotonic()
            detections = self._classify_batch(csv_text)
            self._last_classify_ms = (time.monotonic() - t0) * 1000.0
        except Exception as exc:
            self.result_queue.put((
                "error",
                f"classification failed on batch {self._batch_counter}: {exc}",
            ))
            return

        counts = {cls: 0 for cls in STANDARD_CLASSES}
        for item in detections:
            label = _normalise_label(item.get("label", "unknown"))
            counts[label] = counts.get(label, 0) + 1

        self.result_queue.put(("batch", {
            "elapsed_seconds": round(time.monotonic() - self._run_start, 3),
            "counts": counts,
            "particle_summary": self._particle_summary(detections, self._last_batch_frames),
            "dead_pixels": self._hot_pixel_stats(),
            "throughput": self._throughput_stats(),
            "detections": detections,
        }))
        self._batch_server_ms = []

    def _throughput_stats(self) -> dict:
        """Camera-vs-pipeline accounting for the operator.

        camera_fps and processing_fps are cumulative rates since run start;
        their gap — made explicit by `dropped` / `drop_pct` — is frames the
        experimenter believes were measured but that never went through the
        detector. server_ms_per_frame is this batch's mean C++ round-trip
        (bg update + clustering + pipe), classify_ms this batch's ONNX pass:
        together they say WHERE the ceiling is when processing lags."""
        elapsed = max(time.monotonic() - self._run_start, 1e-6)
        captured = self._frame_counter
        dropped = self._frames_dropped
        server_ms = (sum(self._batch_server_ms) / len(self._batch_server_ms)
                     if self._batch_server_ms else 0.0)
        # What the operator ASKED the camera for, so the GUI can compare it with
        # what actually arrived. These are not the same thing and the gap is
        # silent: an exposure longer than the frame period simply caps the rate.
        # The 2026-07-11 dataset was recorded at a requested 2 fps with a
        # 575.6 ms exposure and therefore ran at 1.80 fps -- an 11% shortfall
        # that nothing reported at the time, and that survives in the folder
        # names to this day. IDSCamera calls it target_fps, SimulatedCamera fps.
        requested = getattr(self.camera, "target_fps", None)
        if requested is None:
            requested = getattr(self.camera, "fps", None)

        return {
            "captured": captured,
            "processed": self._frames_processed,
            "dropped": dropped,
            "requested_fps": round(float(requested), 3) if requested else None,
            "camera_fps": round(captured / elapsed, 3),
            "processing_fps": round(self._frames_processed / elapsed, 3),
            "drop_pct": round(100.0 * dropped / captured, 1) if captured else 0.0,
            "server_ms_per_frame": round(server_ms, 1),
            "classify_ms": round(self._last_classify_ms, 1),
            "saved_raw": self._frames_saved,
            "save_dropped": self._saves_dropped,
        }


    def _start_detector_server(self) -> None:
        """Launch the C++ detector once, in --server mode, and keep it alive
        for the engine's lifetime. bufsize=0 keeps stdin/stdout raw and
        unbuffered so the exact-byte-count header/payload writes below go
        straight through without Python re-chunking them."""
        cmd = [
            str(self.config.detector_exe),
            "--server",
            "--model", str(self.config.background_model_path),
            "--resume",
            "--nsigma", str(self.config.n_sigma),
            "--cluster-gap", str(self.config.cluster_gap),
            "--cluster-grow", str(self.config.cluster_grow),
            "--cluster-grow-min-snr", str(self.config.cluster_grow_min_snr),
            "--min-size", str(self.config.min_size_px),
            "--min-charge", str(self.config.min_charge_adu),
        ]
        if self.config.export_dir:
            cmd += ["--export-dir", str(self.config.export_dir)]
        # Detector stderr carries the per-frame stage timings printed by
        # runServerMode ("[server] frame N: update=..ms detect=..ms ...").
        # Keep it in a log file instead of discarding it, so slow-frame
        # questions can be answered after the fact.
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self._detector_stderr_log = open(
            self.config.work_dir / "detector_stderr.log", "wb")
        self._detector_proc = subprocess.Popen(
            cmd,
            cwd=self.config.detector_exe.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._detector_stderr_log,
            bufsize=0,
        )

        # bufsize=0 is wanted for STDIN (exact-byte frame writes go straight
        # through), but it also makes proc.stdout a raw unbuffered stream —
        # and readline() on raw IO degenerates into byte-sized reads. With
        # ~10k cluster lines (~1.5 MB) per frame that alone cost ~1 s/frame
        # (measured: the C++ "csv_out" stage blocked on this slow reader).
        # A buffered wrapper restores large reads; stdin stays raw.
        self._detector_stdout = io.BufferedReader(
            self._detector_proc.stdout, buffer_size=1 << 20)

        # The server prints its CSV column header on stdout before the first
        # frame (single source of truth: Cluster::csvHeader() in the C++).
        # Consume it here so per-frame reads only ever see cluster rows plus
        # the ===BATCH_DONE=== sentinel, and so _classify_batch() can name the
        # header-less cluster lines it later parses with pandas.
        # Skip any stray non-header lines instead of trusting line #1 blindly:
        # a stopwatch line or config echo accidentally printed to stdout would
        # otherwise be adopted as the "header" and silently break every batch
        # parse afterwards (columns wouldn't match FEATURE_COLS).
        self._csv_header = ""
        for _ in range(20):   # bounded: don't hang forever on a broken server
            raw = self._detector_stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith(CSV_COL_FRAME_ID + ","):
                self._csv_header = line
                break
        if not self._csv_header:
            raise RuntimeError(
                "Detector server did not emit a CSV header line starting with "
                f"'{CSV_COL_FRAME_ID},' — is build/detector up to date with "
                "main.cpp's server mode?")

    def _stop_detector_server(self, timeout: float = 5.0) -> None:
        proc = self._detector_proc
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()   # EOF tells the server to save its model & exit
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)
        except Exception:
            proc.kill()
        finally:
            self._detector_proc = None
            log = getattr(self, "_detector_stderr_log", None)
            if log is not None:
                try:
                    log.close()
                except Exception:
                    pass
                self._detector_stderr_log = None

    def _send_frame_to_server(self, frame: np.ndarray) -> str:
        """Stream one raw Mono8 frame to the persistent detector process and
        return the CSV lines it emits before '===BATCH_DONE==='."""
        proc = self._detector_proc
        height, width = frame.shape[:2]
        header = struct.pack("<ii", width, height)
        payload = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        try:
            # proc.stdin is a raw (unbuffered) stream: a single write() on a
            # pipe may legally transfer only part of a 20 MB payload. Loop
            # over a memoryview until everything is sent — otherwise the
            # server would mis-frame every subsequent message.
            view = memoryview(header + payload)
            while view:
                n = proc.stdin.write(view)
                view = view[n:] if n else view
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"Detector server pipe broken: {exc}") from exc

        lines: list[str] = []
        while True:
            raw_line = self._detector_stdout.readline()
            if not raw_line:
                raise RuntimeError("Detector server closed stdout unexpectedly")
            # \r\n tolerated: belt-and-braces for Windows text-mode stdout.
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "===BATCH_DONE===":
                break
            lines.append(line)
        return "\n".join(lines)

    def _maybe_warn_saturation(self) -> None:
        """Warn once if the live saturation has drifted far from the model's
        training saturation. Waits for enough clusters that the live estimate
        is meaningful (Poisson), then compares."""
        if (self._saturation_warned or self._train_saturation is None
                or self._sat_total < 200):
            return
        live_sat = self._sat_seen / self._sat_total
        # 25 percentage points is a deliberately loose gate: we only want to
        # catch a regime change (saturated<->de-saturated), not normal drift.
        if abs(live_sat - self._train_saturation) >= 0.25:
            self._saturation_warned = True
            self.result_queue.put((
                "error",
                f"OPERATING-POINT MISMATCH: this model was trained at "
                f"{100*self._train_saturation:.0f}% saturated tracks, but live data "
                f"is {100*live_sat:.0f}% saturated. The classifier is seeing "
                f"particles unlike its training set — neutron labels/counts may be "
                f"wrong. Re-label & retrain at this gain, or restore the gain the "
                f"model was trained at.",
            ))

    def _classify_batch(self, batch_csv_text: str) -> list[dict]:
        # The detector server streams header-less CSV lines (one per cluster);
        # a batch with zero clusters is normal, not an error.
        if not batch_csv_text.strip():
            return []

        # Prepend the column header captured from the server so pandas can
        # address columns by name below.
        full_csv = self._csv_header + "\n" + batch_csv_text
        try:
            df = pd.read_csv(io.StringIO(full_csv))
        except pd.errors.EmptyDataError:
            return []
        if df.empty:
            return []

        # ── Operating-point (saturation) guard ────────────────────────────
        # Accumulate the fraction of clusters whose peak clips at 255 and
        # compare it to the saturation the model was TRAINED at. A large gap
        # means the classifier is being fed particles it never saw (e.g. you
        # lowered the gain to de-saturate, but this model was trained on
        # saturated tracks) — its labels, and therefore the neutron count,
        # are then untrustworthy. Warn once, with numbers.
        if "peak_value" in df.columns:
            self._sat_seen += int((df["peak_value"] >= 254).sum())
            self._sat_total += len(df)
            self._maybe_warn_saturation()

        # ── Gain-invariant pre-filter (peak SNR) ──────────────────────────
        # A throughput guardrail only (the ML classifier does the physics):
        # drop clusters whose peak SNR is below the floor. peak_snr is
        # signal/noise in σ units, which cancels gain (both scale together),
        # so this floor stays valid when you change the gain — unlike an
        # absolute ADU charge cut, which would silently start eating real
        # tracks as they dim. Requires the background model to be rebuilt at
        # the operating gain (so σ matches), which you do anyway.
        if self.config.min_peak_snr > 0 and "peak_snr" in df.columns:
            df = df[df["peak_snr"] >= self.config.min_peak_snr]
            if df.empty:
                return []

        raw_cols = [c for c in ml_classifier.FEATURE_COLS if c != "axis_ratio"]
        missing = [c for c in raw_cols if c not in df.columns]
        if missing:
            self.result_queue.put((
                "error",
                f"batch CSV is missing expected columns {missing} — has "
                f"ClusterDetector's CSV header changed?",
            ))
            return []
        for col in (CSV_COL_CENTER_X, CSV_COL_CENTER_Y):
            if col not in df.columns:
                self.result_queue.put((
                    "error",
                    f"batch CSV has no '{col}' column — update "
                    f"CSV_COL_CENTER_X / CSV_COL_CENTER_Y at the top of "
                    f"live_detection_engine.py to match your actual "
                    f"detections.csv header.",
                ))
                return []

        try:
            preds = self._predict_batch_onnx(df)
        except Exception as exc:
            self.result_queue.put(("error", f"batch classification failed: {exc}"))
            return []

        # Persist the FULL rows (all 20 detector columns + prediction) to a
        # running CSV, same layout as the offline pipeline's
        # detections_classified.csv — so study_energy.py (which needs
        # total_charge/peak_value, not just x/y/label) works unmodified on a
        # day of live data. Dropped otherwise: the dict below only keeps
        # what the GUI's live view needs (x/y/label), not the physics columns.
        if self._live_csv_path is not None:
            self._append_live_csv(df, preds)

        # Column-wise numpy access + zip instead of df.iterrows(): iterrows
        # materialises a Series per row and was the single slowest step of
        # the whole live loop at ~10k clusters/frame.
        xs = df[CSV_COL_CENTER_X].to_numpy(dtype=float)
        ys = df[CSV_COL_CENTER_Y].to_numpy(dtype=float)
        if CSV_COL_FRAME_ID in df.columns:
            frame_ids = df[CSV_COL_FRAME_ID].to_numpy(dtype=int).tolist()
        else:
            frame_ids = [None] * len(df)
        return [
            {"x": x, "y": y, "label": label, "confidence": conf, "frame_id": fid}
            for x, y, label, conf, fid in zip(
                xs.tolist(), ys.tolist(),
                preds["label"], preds["confidence"], frame_ids)
        ]

    def _append_live_csv(self, df: pd.DataFrame, preds: dict[str, list]) -> None:
        """Append this batch's classified clusters to the running live CSV.
        Single writer (the inference thread only), so no lock is needed."""
        out = df.copy()
        out["predicted_label"] = preds["label"]
        out["predicted_confidence"] = preds["confidence"]
        try:
            write_header = not self._live_csv_path.exists()
            out.to_csv(self._live_csv_path, mode="a", index=False, header=write_header)
        except Exception as exc:
            self.result_queue.put(("error", f"could not write live CSV: {exc}"))
            self._live_csv_path = None  # stop retrying every batch

    def _predict_batch_onnx(self, df: pd.DataFrame) -> dict[str, list]:
        """Run ONE inference call over every row in the batch.

        This replaces the old per-row `predict_single` loop. Batching is
        what makes onnxruntime worth switching to — a single C++ call over
        an [n_rows, n_features] array instead of n_rows separate Python
        round-trips through pandas + sklearn.
        """
        df = ml_classifier.add_derived_features(df)
        X = df[self._onnx_feature_cols].to_numpy(dtype=np.float32)

        input_name = self._session.get_inputs()[0].name
        label_out_name, proba_out_name = (o.name for o in self._session.get_outputs())
        pred_idx, proba = self._session.run(
            [label_out_name, proba_out_name], {input_name: X}
        )
        proba = np.asarray(proba, dtype=np.float32)  # [n_rows, n_classes]

        best_idx = np.argmax(proba, axis=1)
        labels = [self._onnx_classes[i] for i in best_idx]
        confidences = proba[np.arange(len(best_idx)), best_idx].tolist()
        return {"label": labels, "confidence": confidences}


# ─── small helpers ─────────────────────────────────────────────────────────

def _normalise_label(label: object) -> str:
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live/simulated qBOUNCE detection.")
    # No default ON PURPOSE. This used to default to "simulated", so a bare
    # `python LiveDetectionEngine.py --folder ...` at the beam replayed recorded
    # frames while looking exactly like a live acquisition. Choosing wrongly
    # here does not crash -- it silently produces plausible, worthless data, so
    # the operator must state which one they mean.
    parser.add_argument("--camera", choices=("simulated", "ids"), required=True,
                        help="'ids' = real IDS camera, 'simulated' = replay a "
                             "folder of recorded frames. REQUIRED, no default.")
    parser.add_argument("--folder", type=Path, default=None,
                        help="Image folder for --camera simulated.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--exposure-us", type=float, default=None)
    parser.add_argument("--gain-db", type=float, default=None)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--onnx", default=ml_classifier.ONNX_PATH,
                         help="Path to exported .onnx model "
                              "(expects a .meta.json sidecar next to it)")
    parser.add_argument("--work-dir", type=Path, default=Path("live_scratch"))
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--batch-seconds", type=float, default=BATCH_SECONDS_DEFAULT)
    parser.add_argument("--batch-max-frames", type=int, default=BATCH_MAX_FRAMES_DEFAULT)
    parser.add_argument("--nsigma", type=float, default=5.0)
    parser.add_argument("--cluster-gap", type=int, default=0)
    parser.add_argument("--cluster-grow", type=int, default=0)
    parser.add_argument("--cluster-grow-min-snr", type=float, default=1.0)
    parser.add_argument("--min-size", type=int, default=0,
                        help="drop clusters smaller than this (px); 0 = keep all")
    parser.add_argument("--min-charge", type=float, default=0.0,
                        help="drop clusters with total_charge below this (ADU); 0 = keep all")
    parser.add_argument("--min-peak-snr", type=float, default=0.0,
                        help="gain-invariant guardrail: drop clusters below this peak SNR; 0 = off")
    parser.add_argument("--export-dir", type=Path, default=None,
                        help="write the compact data_analysed products during live "
                             "(masked frames, bg model, arrival heatmap)")
    parser.add_argument("--live-csv", type=Path, default=None,
                        help="append every classified cluster (all physics columns) "
                             "to this running CSV across the session")
    parser.add_argument("--record-raw-dir", type=Path, default=None,
                        help="record every raw camera frame as .bmp (IDS-Cockpit replacement)")
    parser.add_argument("--hot-pixel-consecutive-frames", type=int, default=5)
    parser.add_argument("--hot-pixel-brightness-threshold", type=float, default=50.0)
    args = parser.parse_args()

    if args.camera == "simulated":
        if args.folder is None:
            parser.error("--folder is required for --camera simulated")
        # Loud, because the whole point is that this run must never be mistaken
        # for a measurement when someone reads the log afterwards.
        print("=" * 70, flush=True)
        print("  REPLAY MODE - NOT A MEASUREMENT", flush=True)
        print(f"  Frames are read from: {args.folder}", flush=True)
        print("  Nothing is being acquired from a camera.", flush=True)
        print("=" * 70, flush=True)
        camera = create_camera("simulated", folder=args.folder, fps=args.fps)
    else:
        print("=" * 70, flush=True)
        print("  LIVE MODE - acquiring from the IDS camera", flush=True)
        print(f"  device index: {args.device_index}", flush=True)
        print("=" * 70, flush=True)
        camera = create_camera(
            "ids",
            device_index=args.device_index,
            exposure_us=args.exposure_us,
            gain_db=args.gain_db,
        )

    result_q: "queue.Queue" = queue.Queue()
    config = LiveDetectionConfig(
        detector_exe=args.detector,
        background_model_path=args.model,
        onnx_path=args.onnx,
        work_dir=args.work_dir,
        batch_seconds=args.batch_seconds,
        batch_max_frames=args.batch_max_frames,
        n_sigma=args.nsigma,
        cluster_gap=args.cluster_gap,
        cluster_grow=args.cluster_grow,
        cluster_grow_min_snr=args.cluster_grow_min_snr,
        min_size_px=args.min_size,
        min_charge_adu=args.min_charge,
        min_peak_snr=args.min_peak_snr,
        export_dir=str(args.export_dir) if args.export_dir else None,
        record_raw_dir=str(args.record_raw_dir) if args.record_raw_dir else None,
        live_csv_path=str(args.live_csv) if args.live_csv else None,
        hot_pixel_consecutive_frames=args.hot_pixel_consecutive_frames,
        hot_pixel_brightness_threshold=args.hot_pixel_brightness_threshold,
    )
    engine = LiveDetectionEngine(camera, config, result_q)
    run_start = time.monotonic()
    engine.start()
    print(f"[live] started camera={args.camera} duration={args.duration_s}s", flush=True)

    deadline = time.monotonic() + args.duration_s
    errors: list[str] = []
    try:
        while time.monotonic() < deadline and engine.running:
            try:
                kind, payload = result_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if kind == "batch":
                # Emit the engine's telemetry dict as one compact LIVE_BATCH
                # line — exactly what PipelineController._update_live_from_line
                # parses. The full per-cluster "detections" list is stripped
                # from the wire format: the GUI only reads counts/summary/
                # dead_pixels/throughput, and serialising ~10k detection dicts
                # per batch cost more than the ONNX inference itself.
                wire = {k: v for k, v in payload.items() if k != "detections"}
                print("LIVE_BATCH: " + json.dumps(wire), flush=True)

                summary = payload.get("particle_summary", {})
                thr = payload.get("throughput", {})
                print(
                    "[live] "
                    f"frames={summary.get('frames')} "
                    f"detections={len(payload.get('detections', []))} "
                    f"neutrons={summary.get('neutrons')} "
                    f"alpha={summary.get('alpha')} lithium={summary.get('lithium')} | "
                    f"camera={thr.get('camera_fps')}fps "
                    f"processing={thr.get('processing_fps')}fps "
                    f"dropped={thr.get('dropped')} ({thr.get('drop_pct')}%)",
                    flush=True,
                )
            elif kind == "frame":
                # High-frequency per-frame arrays for the live heatmap; the
                # count/rate telemetry does not depend on them, so they are
                # not forwarded over stdout here.
                pass
            elif kind == "status":
                print(f"[live] status {payload}", flush=True)
            elif kind == "error":
                errors.append(str(payload))
                print(f"[live] ERROR {payload}", file=sys.stderr, flush=True)
    finally:
        engine.stop()
        # Drain what is still queued before reporting. A camera that cannot be
        # opened at all posts its error and immediately clears `engine.running`,
        # so the loop above can exit without ever reading it -- which is how a
        # busy camera used to look like a run that simply "stopped", with no
        # reason anywhere in the GUI console.
        while True:
            try:
                kind, payload = result_q.get_nowait()
            except queue.Empty:
                break
            if kind == "error":
                errors.append(str(payload))
                print(f"[live] ERROR {payload}", file=sys.stderr, flush=True)
        print("[live] stopped", flush=True)

    # Non-zero exit so PipelineController marks the step FAILED and shows the
    # reason, instead of a green finish with no frames.
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
