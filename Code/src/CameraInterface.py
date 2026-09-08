#!/usr/bin/env python3
"""
camera_interface.py – common camera interface for live acquisition.

Two implementations behind the same small interface (open / get_frame /
set_exposure_us / close, usable as a context manager):

  - IdsCamera        : continuous live streaming from a real IDS peak camera.
                        Reuses the exact SDK-handling helpers already proven
                        out in ids_capture.py (device discovery, node
                        get/set, Mono8 conversion) instead of duplicating
                        them, so both scripts stay in sync with one SDK
                        wrapper.
  - SimulatedCamera   : replays images from a folder at a controllable FPS,
                        looping by default. No SDK required — this is what
                        lets live_detection_engine.py and PipelineController
                        be developed and tested on a machine without IDS
                        peak installed (e.g. off the lab Windows/Linux box).

Typical use:
    from camera_interface import create_camera

    cam = create_camera("ids", device_index=0, exposure_us=500_000)
    # or, with no camera attached:
    cam = create_camera("simulated", folder="Code/Data/signal", fps=10)

    with cam:
        while running:
            frame = cam.get_frame(timeout_ms=2000)   # np.uint8 2D array, or None
            if frame is not None:
                ...
"""

from __future__ import annotations

import sys
import re
import time
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from PIL import Image

# ids_capture.py lives next to this file in Code/src/ — make sure that's on
# sys.path regardless of how camera_interface.py was imported or launched,
# so it doesn't silently depend on the caller having set that up already.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ids_capture import (  # noqa: E402  (import after sys.path fix-up, intentional)
    CaptureError,
    _import_ids_peak,
    _devices_as_list,
    _node,
    _set_value,
    _set_enum,
    _device_label,
    _mono8_numpy,
    _open_device_or_explain,
)

DEFAULT_IMAGE_EXTENSIONS = {".bmp",".raw", ".tiff", ".tif", ".png"}


@runtime_checkable
class CameraInterface(Protocol):
    """What live_detection_engine.py needs from a camera. IdsCamera and
    SimulatedCamera both satisfy this without inheriting from it — it's
    here for documentation/type-checking, not for isinstance() checks."""

    label: str

    def open(self) -> "CameraInterface": ...
    def get_frame(self, timeout_ms: Optional[int] = None) -> Optional[np.ndarray]: ...
    def set_exposure_us(self, exposure_us: float) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> "CameraInterface": ...
    def __exit__(self, *exc) -> bool: ...


def _natural_sort_key(name: str) -> list:
    """Split into digit / non-digit runs so frame_2.png sorts before
    frame_10.png. Same idea as labeling_ui.py's natural_sort_key and
    main.cpp's naturalLess — kept self-contained here to avoid pulling
    in the (much heavier) labeling UI module for one helper."""
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", name)]


# ─── Real camera ────────────────────────────────────────────────────────────

class IdsCamera:
    """Continuous live streaming from an IDS peak camera.

    Unlike ids_capture.py's capture() — which opens the camera, grabs a
    fixed --count frames, and closes again — this keeps the device, node
    map, and data stream open across many get_frame() calls, which is what
    a live GUI loop needs.
    """

    def __init__(
        self,
        device_index: int = 0,
        exposure_us: Optional[float] = None,
        gain_db: Optional[float] = None,
        pixel_format: str = "Mono8",
        target_fps: Optional[float] = None,
        buffers: int = 4,
        timeout_ms: int = 5000,
    ) -> None:
        self.device_index = device_index
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.pixel_format = pixel_format
        self.target_fps = target_fps
        self.buffers = buffers
        self.timeout_ms = timeout_ms

        self.label = f"IDS camera #{device_index}"
        self._peak = None
        self._ipl = None
        self._device = None
        self._node_map = None
        self._data_stream = None
        self._tl_locked_node = None
        self._opened = False

    def open(self) -> "IdsCamera":
        peak, ipl = _import_ids_peak()
        self._peak, self._ipl = peak, ipl
        peak.Library.Initialize()
        try:
            manager = peak.DeviceManager.Instance()
            manager.Update()
            devices = _devices_as_list(manager)
            if not devices:
                raise CaptureError(
                    "No IDS camera found. Check cable, power, driver, and IDS peak Cockpit."
                )
            if not (0 <= self.device_index < len(devices)):
                raise CaptureError(
                    f"Device index {self.device_index} is invalid; found {len(devices)} camera(s)."
                )

            self.label = _device_label(devices[self.device_index])
            # Same opener as ids_capture.py: live acquisition is where the
            # "camera already in use" case actually happens (Cockpit left
            # open, or the ① dark capture's python.exe still holding the
            # device), and the raw GC_ERR_ACCESS_DENIED text names the
            # GenICam layer rather than telling the operator what to close.
            device = _open_device_or_explain(devices[self.device_index], peak)
            self._device = device
            node_map = device.RemoteDevice().NodeMaps()[0]
            self._node_map = node_map
            data_stream = device.DataStreams()[0].OpenDataStream()
            self._data_stream = data_stream

            _set_enum(node_map, "AcquisitionMode", "Continuous")
            _set_enum(node_map, "TriggerMode", "Off")
            if self.pixel_format:
                _set_enum(node_map, "PixelFormat", self.pixel_format)
            if self.exposure_us is not None:
                _set_value(node_map, "ExposureTime", float(self.exposure_us))
            if self.gain_db is not None:
                _set_value(node_map, "Gain", float(self.gain_db))
            if self.target_fps is not None:
                # Not every camera/firmware exposes these nodes; _set_enum /
                # _set_value already degrade gracefully (print + continue)
                # when a node is missing, so this is safe either way.
                _set_enum(node_map, "AcquisitionFrameRateEnable", "true")
                _set_value(node_map, "AcquisitionFrameRate", float(self.target_fps))

            payload_size = int(node_map.FindNode("PayloadSize").Value())
            min_buffers = int(data_stream.NumBuffersAnnouncedMinRequired())
            for _ in range(max(min_buffers, self.buffers)):
                buffer = data_stream.AllocAndAnnounceBuffer(payload_size)
                data_stream.QueueBuffer(buffer)

            self._tl_locked_node = _node(node_map, "TLParamsLocked")
            if self._tl_locked_node is not None:
                self._tl_locked_node.SetValue(1)

            data_stream.StartAcquisition()
            node_map.FindNode("AcquisitionStart").Execute()
            node_map.FindNode("AcquisitionStart").WaitUntilDone()
            self._opened = True
        except Exception:
            self._safe_close()
            raise
        return self

    def get_frame(self, timeout_ms: Optional[int] = None) -> Optional[np.ndarray]:
        if not self._opened:
            raise RuntimeError("IdsCamera.get_frame() called before open()")
        timeout_ms = self.timeout_ms if timeout_ms is None else timeout_ms
        try:
            buffer = self._data_stream.WaitForFinishedBuffer(timeout_ms)
        except Exception as exc:
            # Timeouts are routine when nothing has triggered — treat as
            # "no frame yet" rather than a hard error, matching how a live
            # loop should behave (keep polling, don't crash the worker).
            print(f"[camera_interface] frame wait failed/timed out: {exc}")
            return None
        try:
            return _mono8_numpy(self._ipl, buffer)
        finally:
            self._data_stream.QueueBuffer(buffer)

    def set_exposure_us(self, exposure_us: float) -> None:
        self.exposure_us = exposure_us
        if self._node_map is not None:
            _set_value(self._node_map, "ExposureTime", float(exposure_us))

    def set_target_fps(self, fps: float) -> None:
        self.target_fps = fps
        if self._node_map is not None:
            _set_value(self._node_map, "AcquisitionFrameRate", float(fps))

    def close(self) -> None:
        self._safe_close()

    def _safe_close(self) -> None:
        node_map, data_stream, peak = self._node_map, self._data_stream, self._peak
        if node_map is not None:
            try:
                node_map.FindNode("AcquisitionStop").Execute()
                node_map.FindNode("AcquisitionStop").WaitUntilDone()
            except Exception:
                pass
            try:
                if self._tl_locked_node is not None:
                    self._tl_locked_node.SetValue(0)
            except Exception:
                pass

        if data_stream is not None and peak is not None:
            try:
                data_stream.StopAcquisition(peak.AcquisitionStopMode_Default)
            except Exception:
                pass
            try:
                data_stream.Flush(peak.DataStreamFlushMode_DiscardAll)
            except Exception:
                pass
            try:
                for buffer in data_stream.AnnouncedBuffers():
                    data_stream.RevokeBuffer(buffer)
            except Exception:
                pass

        if peak is not None:
            try:
                peak.Library.Close()
            except Exception:
                pass

        self._opened = False
        self._device = self._node_map = self._data_stream = None

    def __enter__(self) -> "IdsCamera":
        return self.open()

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


# ─── Simulated camera (no SDK / no hardware required) ─────────────────────

class SimulatedCamera:
    """Replays images from a folder at a controllable FPS, looping by
    default. Drop-in stand-in for IdsCamera so live_detection_engine.py
    and the PipelineController Live Acquisition tab can be built and
    tested without a camera attached."""

    def __init__(
        self,
        folder: str | Path,
        fps: float = 10.0,
        loop: bool = True,
        extensions: set[str] = DEFAULT_IMAGE_EXTENSIONS,
    ) -> None: 
        self.folder = Path(folder)
        print(f"\n[DEBUG CAMERA] Folder received : {self.folder.resolve()}", flush=True)
        print(f"[DEBUG CAMERA] Exists ? {self.folder.exists()}", flush=True)
        self.fps = fps
        self.loop = loop
        self.extensions = {e.lower() for e in extensions}
        self.label = f"SimulatedCamera({self.folder})"

        self._paths: list[Path] = []
        self._idx = 0
        self._opened = False
        self._last_frame_time: Optional[float] = None

    def open(self) -> "SimulatedCamera":
        if not self.folder.is_dir():
            raise CaptureError(f"Simulated camera folder not found: {self.folder}")
        self._paths = sorted(
            (p for p in self.folder.iterdir() if p.suffix.lower() in self.extensions),
            key=lambda p: _natural_sort_key(p.name),
        )
        if not self._paths:
            raise CaptureError(
                f"No images ({sorted(self.extensions)}) found in {self.folder}"
            )
        self._idx = 0
        self._last_frame_time = None
        self._opened = True
        return self

    def get_frame(self, timeout_ms: Optional[int] = None) -> Optional[np.ndarray]:
        if not self._opened:
            raise RuntimeError("SimulatedCamera.get_frame() called before open()")
        if self._idx >= len(self._paths):
            if not self.loop:
                return None
            self._idx = 0

        if self.fps and self.fps > 0:
            period = 1.0 / self.fps
            now = time.monotonic()
            if self._last_frame_time is not None:
                remaining = period - (now - self._last_frame_time)
                if remaining > 0:
                    time.sleep(remaining)
            self._last_frame_time = time.monotonic()

        path = self._paths[self._idx]
        self._idx += 1
        with Image.open(path) as img:
            return np.array(img.convert("L"), dtype=np.uint8)

    def set_exposure_us(self, exposure_us: float) -> None:
        pass  # no-op — frames are pre-recorded, nothing to adjust

    def close(self) -> None:
        self._opened = False

    def __enter__(self) -> "SimulatedCamera":
        return self.open()

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


# ─── Factory ────────────────────────────────────────────────────────────────

def create_camera(kind: str, **kwargs) -> CameraInterface:
    """kind: 'ids' for a real camera, 'simulated' for folder playback."""
    key = kind.strip().lower()
    if key in ("ids", "ids_peak", "real"):
        return IdsCamera(**kwargs)
    if key in ("simulated", "sim", "folder"):
        return SimulatedCamera(**kwargs)
    raise ValueError(f"Unknown camera kind: {kind!r} (expected 'ids' or 'simulated')")