#!/usr/bin/env python3
"""
IDS peak camera capture helper for the qBOUNCE particle pipeline.

This script is intentionally separate from the C++ detector:
  - the existing detector can still process folders of images offline;
  - the IDS SDK is imported only when this script runs;
  - the lab machine can install IDS peak while development on macOS remains
    possible without the SDK.

Typical use at the lab:
    python src/ids_capture.py --output Code/Data/dark --count 100
    python src/ids_capture.py --output Code/Data/signal --count 1000 --exposure-us 500000
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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



class CaptureError(RuntimeError):
    pass


def _import_ids_peak() -> tuple[Any, Any]:
    try:
        from ids_peak import ids_peak as peak
        from ids_peak_ipl import ids_peak_ipl as ipl
    except ImportError as exc:
        raise CaptureError(
            "IDS peak Python bindings are not installed for this Python environment.\n"
            "Install IDS peak on the lab Windows/Linux machine, then use the Python "
            "interpreter supported by that installation. On the lab PC, verify with:\n"
            "  python -c \"from ids_peak import ids_peak; "
            "from ids_peak_ipl import ids_peak_ipl\""
        ) from exc
    return peak, ipl


def _open_device_or_explain(device: Any, peak: Any) -> Any:
    """Open the camera for control, turning GenICam's error into an instruction.

    A USB3 Vision camera grants control access to ONE process at a time. When
    something else already holds it -- almost always IDS peak Cockpit left open,
    sometimes a python.exe from a crashed run -- the SDK raises
    BadAccessException / GC_ERR_ACCESS_DENIED "Control access is not available
    on the remote device". That names the GenICam layer, not the cause, and it
    arrives after a line saying the camera WAS found, which reads as a
    contradiction. Say what to do instead."""
    try:
        return device.OpenDevice(peak.DeviceAccessType_Control)
    except Exception as exc:
        text = str(exc)
        if "ACCESS" in text.upper() or "BadAccess" in text:
            raise CaptureError(
                "The camera was found, but another program is already "
                "controlling it.\n\n"
                "A USB3 Vision camera allows only one program at a time to "
                "control it (needed to set exposure/gain and start "
                "acquisition).\n\n"
                "To fix it, in this order:\n"
                "  1. close IDS peak Cockpit if it is open -- this is almost "
                "always the cause;\n"
                "  2. end any leftover python.exe from a previous attempt "
                "(Task Manager);\n"
                "  3. unplug the camera USB cable, wait 5 s, plug it back in.\n\n"
                "This is NOT about the image folders: nothing has been "
                "captured yet at this point.\n\n"
                f"Original SDK error: {text}") from exc
        raise


def _devices_as_list(device_manager: Any) -> list[Any]:
    devices = device_manager.Devices()
    try:
        return list(devices)
    except TypeError:
        return [devices[i] for i in range(devices.Size())]


def _node(node_map: Any, name: str) -> Any | None:
    try:
        return node_map.FindNode(name)
    except Exception:
        return None


def _set_value(node_map: Any, name: str, value: float | int) -> float | int | None:
    """Write a numeric node, and return the value the camera actually holds.

    Returns None when the node is absent or the write failed, so a caller that
    reasons about the value afterwards (auto_tune searching gain) works from
    what the sensor really applied rather than from what it asked for."""
    node = _node(node_map, name)
    if node is None:
        print(f"[ids_capture] camera has no node '{name}', leaving it unchanged")
        return None
    # Clamp into the node's own range before writing. Gain in particular is a
    # multiplicative factor on this sensor with a minimum of 1.0, not a dB
    # offset where 0 would mean "none" -- so the GUI's default of 0.0 was
    # rejected outright (PEAK_RETURN_CODE_OUT_OF_RANGE) and the setting was
    # silently skipped, leaving the camera at whatever gain it happened to
    # hold. Clamping applies the operator's intent (the lowest available gain)
    # and says so, instead of quietly doing nothing.
    requested = value
    try:
        lo, hi = node.Minimum(), node.Maximum()
        value = min(max(value, lo), hi)
        if value != requested:
            print(f"[ids_capture] {name}={requested} is outside the camera's "
                  f"range [{lo}, {hi}] -- using {value} instead")
    except Exception:
        pass          # node exposes no range; write the value as given
    try:
        node.SetValue(value)
        print(f"[ids_capture] {name}={value}")
        return value
    except Exception as exc:
        print(f"[ids_capture] could not set {name}: {exc}")
        return None


def _node_range(node_map: Any, name: str) -> tuple[float, float] | None:
    """(Minimum, Maximum) of a numeric node, or None when it has no range."""
    node = _node(node_map, name)
    if node is None:
        return None
    try:
        return float(node.Minimum()), float(node.Maximum())
    except Exception:
        return None


def _set_enum(node_map: Any, name: str, entry: str) -> None:
    node = _node(node_map, name)
    if node is None:
        print(f"[ids_capture] camera has no node '{name}', leaving it unchanged")
        return
    try:
        node.SetCurrentEntry(entry)
        print(f"[ids_capture] {name}={entry}")
        return
    except Exception as exc:
        enum_error = exc
    # Not every "pick one of these" setting is a GenICam enumeration: the frame
    # rate switch (AcquisitionFrameRateEnable) is a BooleanNode, which has no
    # entries and rejects SetCurrentEntry outright. Retry as a boolean before
    # reporting a failure, so asking for a target fps actually takes effect.
    if entry.strip().lower() in ("true", "false", "on", "off", "1", "0"):
        try:
            node.SetValue(entry.strip().lower() in ("true", "on", "1"))
            print(f"[ids_capture] {name}={entry} (boolean node)")
            return
        except Exception:
            pass
    print(f"[ids_capture] could not set {name}={entry}: {enum_error}")


def _device_label(device: Any) -> str:
    parts: list[str] = []
    for attr in ("DisplayName", "ModelName", "SerialNumber"):
        try:
            value = getattr(device, attr)()
            if value:
                parts.append(str(value))
        except Exception:
            pass
    return " | ".join(parts) if parts else repr(device)


def _image_from_buffer(ipl: Any, buffer: Any) -> Any:
    """Wrap a filled acquisition buffer as an IPL image.

    The spelling of this call is not stable across ids_peak_ipl releases: the
    module-level Image_CreateFromSizeAndBuffer() that older versions expose is
    gone from the current PyPI wheel (1.17.x), where it lives as a static method
    on Image, and IDS also ships a helper module that does the same thing. Try
    each form rather than pinning one -- the lab machine's SDK version is not
    ours to choose, and getting this wrong costs a beam shift."""
    args = (buffer.PixelFormat(), buffer.BasePtr(), buffer.Size(),
            buffer.Width(), buffer.Height())
    attempts = []

    # 1. Static method on the Image class (current API).
    img_cls = getattr(ipl, "Image", None)
    factory = getattr(img_cls, "CreateFromSizeAndBuffer", None) if img_cls else None
    if factory is not None:
        attempts.append(("ipl.Image.CreateFromSizeAndBuffer", lambda: factory(*args)))

    # 2. Module-level function (older API).
    legacy = getattr(ipl, "Image_CreateFromSizeAndBuffer", None)
    if legacy is not None:
        attempts.append(("ipl.Image_CreateFromSizeAndBuffer", lambda: legacy(*args)))

    # 3. IDS's own convenience module. It ships inside the ids_peak package
    #    (ids_peak/ids_peak_ipl_extension.py), NOT ids_peak_ipl -- verified in
    #    the wheels bundled under Code/wheels. Importing it from the wrong
    #    package raises ImportError, which would silently drop this fallback,
    #    so try both spellings rather than trusting either one.
    for _pkg in ("ids_peak", "ids_peak_ipl"):
        try:
            _ext = __import__(f"{_pkg}.ids_peak_ipl_extension",
                              fromlist=["ids_peak_ipl_extension"])
        except Exception:
            continue
        conv = getattr(_ext, "BufferToImage", None)
        if conv is not None:
            attempts.append((f"{_pkg}.ids_peak_ipl_extension.BufferToImage",
                             lambda c=conv: c(buffer)))
            break

    errors = []
    for name, call in attempts:
        try:
            return call()
        except Exception as exc:            # wrong signature in this release
            errors.append(f"{name}: {exc}")

    raise CaptureError(
        "Could not wrap the camera buffer as an image: this ids_peak_ipl "
        "release exposes none of the known entry points.\n"
        f"  ids_peak_ipl version: {getattr(ipl, '__version__', 'unknown')}\n"
        "  tried: " + ("; ".join(n for n, _ in attempts) or "nothing available")
        + ("\n  errors: " + " | ".join(errors) if errors else "")
        + "\nInstall an ids_peak_ipl matching the IDS peak SDK on this machine.")


def _mono8_numpy(ipl: Any, buffer: Any) -> np.ndarray:
    image = _image_from_buffer(ipl, buffer)
    converted = image.ConvertTo(ipl.PixelFormatName_Mono8)
    frame = converted.get_numpy_2D()
    return np.asarray(frame, dtype=np.uint8).copy()


def _save_frame(frame: np.ndarray, output_dir: Path, index: int, extension: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    filename = f"ids_{index:06d}_{timestamp}.{extension}"
    path = output_dir / filename
    Image.fromarray(frame).save(path)
    return path


def list_devices() -> int:
    peak, _ipl = _import_ids_peak()
    peak.Library.Initialize()
    try:
        manager = peak.DeviceManager.Instance()
        manager.Update()
        devices = _devices_as_list(manager)
        if not devices:
            print("No IDS camera found.")
            return 1
        for idx, device in enumerate(devices):
            print(f"{idx}: {_device_label(device)}")
        return 0
    finally:
        peak.Library.Close()


def capture(args: argparse.Namespace) -> int:
    peak, ipl = _import_ids_peak()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_output:
        for old in output_dir.iterdir():
            if old.is_file() and old.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"}:
                old.unlink()

    peak.Library.Initialize()
    data_stream = None
    node_map = None
    try:
        manager = peak.DeviceManager.Instance()
        manager.Update()
        devices = _devices_as_list(manager)
        if not devices:
            raise CaptureError("No IDS camera found. Check cable, power, driver, and IDS peak Cockpit.")
        if args.device_index < 0 or args.device_index >= len(devices):
            raise CaptureError(
                f"Device index {args.device_index} is invalid; found {len(devices)} camera(s)."
            )

        print(f"[ids_capture] opening camera {args.device_index}: {_device_label(devices[args.device_index])}")
        device = _open_device_or_explain(devices[args.device_index], peak)
        node_map = device.RemoteDevice().NodeMaps()[0]
        data_stream = device.DataStreams()[0].OpenDataStream()

        _set_enum(node_map, "AcquisitionMode", "Continuous")
        if args.disable_trigger:
            _set_enum(node_map, "TriggerMode", "Off")
        if args.pixel_format:
            _set_enum(node_map, "PixelFormat", args.pixel_format)
        if args.exposure_us is not None:
            _set_value(node_map, "ExposureTime", float(args.exposure_us))
        if args.gain_db is not None:
            _set_value(node_map, "Gain", float(args.gain_db))

        payload_size = int(node_map.FindNode("PayloadSize").Value())
        min_buffers = int(data_stream.NumBuffersAnnouncedMinRequired())
        for _ in range(max(min_buffers, args.buffers)):
            buffer = data_stream.AllocAndAnnounceBuffer(payload_size)
            data_stream.QueueBuffer(buffer)

        tl_locked = _node(node_map, "TLParamsLocked")
        if tl_locked is not None:
            tl_locked.SetValue(1)

        data_stream.StartAcquisition()
        node_map.FindNode("AcquisitionStart").Execute()
        node_map.FindNode("AcquisitionStart").WaitUntilDone()

        print(f"[ids_capture] capturing {args.count} frame(s) -> {output_dir}")
        saved = 0
        for idx in range(args.count):
            buffer = data_stream.WaitForFinishedBuffer(args.timeout_ms)
            try:
                frame = _mono8_numpy(ipl, buffer)
                path = _save_frame(frame, output_dir, idx, args.format)
                saved += 1
                if saved == 1 or saved % args.progress_every == 0 or saved == args.count:
                    print(f"[ids_capture] saved {saved}/{args.count}: {path.name}")
            finally:
                data_stream.QueueBuffer(buffer)
            if args.delay_ms > 0:
                time.sleep(args.delay_ms / 1000.0)

        print(f"[ids_capture] done: {saved} frame(s) saved")
        return 0
    finally:
        if node_map is not None:
            try:
                node_map.FindNode("AcquisitionStop").Execute()
                node_map.FindNode("AcquisitionStop").WaitUntilDone()
            except Exception:
                pass
            try:
                tl_locked = _node(node_map, "TLParamsLocked")
                if tl_locked is not None:
                    tl_locked.SetValue(0)
            except Exception:
                pass

        if data_stream is not None:
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

        peak.Library.Close()


def query_timing(args: argparse.Namespace) -> int:
    """Read the camera's actual ExposureTime and resulting frame rate.

    The sensor may clamp exposure below 1/fps (readout dead time), so the
    values reported HERE — not the requested ones — define the temporal
    coverage (exposure × fps) and therefore the fraction of neutrons that
    can be counted at all. Emits one machine-readable line:
        CAMERA_TIMING: exposure_us=<val> frame_rate=<val>
    """
    peak, _ipl = _import_ids_peak()
    peak.Library.Initialize()
    try:
        manager = peak.DeviceManager.Instance()
        manager.Update()
        devices = _devices_as_list(manager)
        if not devices:
            raise CaptureError("No IDS camera found.")
        device = _open_device_or_explain(devices[args.device_index], peak)
        node_map = device.RemoteDevice().NodeMaps()[0]

        def read(name: str) -> float | None:
            node = _node(node_map, name)
            if node is None:
                return None
            try:
                return float(node.Value())
            except Exception:
                return None

        exposure_us = read("ExposureTime")
        # ResultingFrameRate reflects what the sensor actually achieves;
        # AcquisitionFrameRate is only the requested value.
        fps = read("ResultingFrameRate") or read("AcquisitionFrameRate")
        if exposure_us is None or fps is None:
            raise CaptureError(
                f"Camera exposes no readable timing nodes "
                f"(ExposureTime={exposure_us}, frame rate={fps}).")
        coverage = min(100.0, exposure_us * 1e-6 * fps * 100.0)
        print(f"[query] exposure={exposure_us:.0f} us  frame_rate={fps:.3f} fps  "
              f"coverage={coverage:.1f}%")
        print(f"CAMERA_TIMING: exposure_us={exposure_us:.0f} frame_rate={fps:.4f}",
              flush=True)
        return 0
    finally:
        peak.Library.Close()


def auto_tune(args: argparse.Namespace) -> int:
    """Search the GAIN that keeps saturation below a user target.

    Why gain and not exposure: a charged particle deposits its charge on the
    CMOS essentially instantaneously (one prompt event during the frame), so
    the exposure time does not change a track's peak amplitude — it only
    changes dark accumulation and how many tracks land per frame. What sets
    whether a track clips at 255 is the gain (ADU per electron). So to
    de-saturate the tracks for the deposited-energy spectrum, we lower the
    GAIN, leaving exposure to the experimenter (it governs counting duty
    cycle, a separate concern).

    Method: measure the mean number of saturated pixels (>=254) per frame,
    and step the gain DOWN (in dB) until it drops below --target-sat-pixels
    or the gain floor is reached. Run with the BEAM ON.

    Emits one machine-readable line at the end:
        AUTO_TUNE_RESULT: gain_db=<value> sat_pixels=<mean> target=<target>
    which PipelineController parses to fill the energy-mode GAIN field.
    """
    peak, ipl = _import_ids_peak()
    peak.Library.Initialize()
    data_stream = None
    node_map = None
    try:
        manager = peak.DeviceManager.Instance()
        manager.Update()
        devices = _devices_as_list(manager)
        if not devices:
            raise CaptureError("No IDS camera found. Check cable, power, driver, and IDS peak Cockpit.")
        if args.device_index < 0 or args.device_index >= len(devices):
            raise CaptureError(
                f"Device index {args.device_index} is invalid; found {len(devices)} camera(s).")

        print(f"[auto_tune] opening camera {args.device_index}: "
              f"{_device_label(devices[args.device_index])}")
        device = _open_device_or_explain(devices[args.device_index], peak)
        node_map = device.RemoteDevice().NodeMaps()[0]
        data_stream = device.DataStreams()[0].OpenDataStream()

        _set_enum(node_map, "AcquisitionMode", "Continuous")
        _set_enum(node_map, "TriggerMode", "Off")
        _set_enum(node_map, "PixelFormat", args.pixel_format)
        # Fix exposure (it doesn't affect prompt-track saturation); only gain
        # is searched below.
        if args.exposure_us is not None:
            _set_value(node_map, "ExposureTime", float(args.exposure_us))

        payload_size = int(node_map.FindNode("PayloadSize").Value())
        min_buffers = int(data_stream.NumBuffersAnnouncedMinRequired())
        for _ in range(max(min_buffers, args.buffers)):
            data_stream.AllocAndAnnounceBuffer(payload_size)
        # Deliberately NOT queued here: every measure() below ends with
        # Flush(DiscardAll), which un-queues all buffers while leaving them
        # announced. Queueing once up front would therefore leave the second
        # and later gain steps with an empty input pool, and each one would
        # sit on WaitForFinishedBuffer until it timed out. measure() queues.

        # The Gain node's own floor, not the CLI default. On this sensor Gain
        # is a multiplicative factor with a minimum of 1.0, so searching down
        # to --min-gain-db 0 would spend its last steps below what the camera
        # accepts and then report a gain (0) the camera never held. Start from
        # what the device says; fall back to the flag when it exposes no range.
        gain_range = _node_range(node_map, "Gain")
        floor = max(float(args.min_gain_db), gain_range[0]) if gain_range else float(args.min_gain_db)
        if gain_range:
            print(f"[auto_tune] camera Gain range: [{gain_range[0]}, {gain_range[1]}] "
                  f"-> searching down to {floor}")

        def measure(gain_db: float) -> tuple[float, float]:
            """(gain actually applied, mean saturated pixels per frame)."""
            applied = _set_value(node_map, "Gain", float(gain_db))
            gain_db = float(applied) if applied is not None else float(gain_db)
            for buf in data_stream.AnnouncedBuffers():
                try:
                    data_stream.QueueBuffer(buf)
                except Exception:
                    pass          # already queued (first step) -- nothing to do
            tl_locked = _node(node_map, "TLParamsLocked")
            if tl_locked is not None:
                tl_locked.SetValue(1)
            data_stream.StartAcquisition()
            node_map.FindNode("AcquisitionStart").Execute()
            node_map.FindNode("AcquisitionStart").WaitUntilDone()
            try:
                counts = []
                for _ in range(args.tune_frames):
                    buffer = data_stream.WaitForFinishedBuffer(args.timeout_ms)
                    try:
                        frame = _mono8_numpy(ipl, buffer)
                        counts.append(int(np.count_nonzero(frame >= 254)))
                    finally:
                        data_stream.QueueBuffer(buffer)
                return gain_db, (float(np.mean(counts)) if counts else 0.0)
            finally:
                try:
                    node_map.FindNode("AcquisitionStop").Execute()
                    node_map.FindNode("AcquisitionStop").WaitUntilDone()
                except Exception:
                    pass
                try:
                    data_stream.StopAcquisition(peak.AcquisitionStopMode_Default)
                except Exception:
                    pass
                try:
                    data_stream.Flush(peak.DataStreamFlushMode_DiscardAll)
                except Exception:
                    pass
                if tl_locked is not None:
                    try:
                        tl_locked.SetValue(0)
                    except Exception:
                        pass

        gain, sat = measure(float(args.start_gain_db))
        print(f"[auto_tune] gain={gain:.1f} -> {sat:.1f} saturated px/frame "
              f"(target <= {args.target_sat_pixels})")
        # Step the gain DOWN in fixed decrements (−6 dB ≈ half the linear
        # signal). Stops at the target or the camera's own gain floor.
        for _ in range(40):
            if sat <= args.target_sat_pixels:
                break
            if gain <= floor:
                print(f"[auto_tune] gain floor {floor} reached while still "
                      "saturating — the tracks are intrinsically brighter than "
                      "the target; accept more saturation or check the beam.")
                break
            gain, sat = measure(max(gain - args.gain_step_db, floor))
            print(f"[auto_tune] gain={gain:.1f} -> {sat:.1f} saturated px/frame "
                  f"(target <= {args.target_sat_pixels})")

        print(f"AUTO_TUNE_RESULT: gain_db={gain:.2f} "
              f"sat_pixels={sat:.1f} target={args.target_sat_pixels}", flush=True)
        return 0
    finally:
        if data_stream is not None:
            try:
                for buffer in data_stream.AnnouncedBuffers():
                    data_stream.RevokeBuffer(buffer)
            except Exception:
                pass
        peak.Library.Close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture IDS peak camera frames into a folder.")
    parser.add_argument("--list-devices", action="store_true", help="List detected IDS cameras and exit.")
    parser.add_argument("--output", type=Path, default=Path("Code/Data/signal"),
                        help="Folder where captured frames will be written.")
    parser.add_argument("--count", type=int, default=100, help="Number of frames to capture.")
    parser.add_argument("--device-index", type=int, default=0, help="Camera index from --list-devices.")
    parser.add_argument("--exposure-us", type=float, default=None,
                        help="Exposure time in microseconds. Omit to keep camera/Cockpit value.")
    parser.add_argument("--gain-db", type=float, default=None,
                        help="Gain in dB. Omit to keep camera/Cockpit value.")
    parser.add_argument("--pixel-format", default="Mono8",
                        help="Camera PixelFormat to request before capture. Default: Mono8.")
    parser.add_argument("--format", choices=("png", "tiff", "bmp"), default="png",
                        help="Image file format for saved frames.")
    parser.add_argument("--timeout-ms", type=int, default=5000,
                        help="Timeout for each frame acquisition.")
    parser.add_argument("--delay-ms", type=float, default=0.0,
                        help="Optional pause after saving each frame.")
    parser.add_argument("--buffers", type=int, default=4,
                        help="Minimum number of acquisition buffers.")
    parser.add_argument("--progress-every", type=int, default=25,
                        help="Print progress every N saved frames.")
    parser.add_argument("--clear-output", action="store_true",
                        help="Delete existing image files in the output folder before capture.")
    parser.add_argument("--keep-trigger", dest="disable_trigger", action="store_false",
                        help="Do not change TriggerMode before capture.")
    parser.set_defaults(disable_trigger=True)

    # ── exposure auto-tune (see auto_tune()) ─────────────────────────────
    parser.add_argument("--query", action="store_true",
                        help="Print the camera's actual exposure/frame-rate timing and exit.")
    parser.add_argument("--auto-tune", action="store_true",
                        help="Search the GAIN keeping saturated pixels/frame below "
                             "--target-sat-pixels (run with BEAM ON), then exit.")
    parser.add_argument("--target-sat-pixels", type=float, default=50.0,
                        help="Auto-tune target: acceptable mean saturated pixels per frame.")
    parser.add_argument("--tune-frames", type=int, default=5,
                        help="Frames averaged per gain step during auto-tune.")
    parser.add_argument("--start-gain-db", type=float, default=24.0,
                        help="Auto-tune starting gain (default: 24 dB, then steps down).")
    parser.add_argument("--min-gain-db", type=float, default=0.0,
                        help="Auto-tune gain floor.")
    parser.add_argument("--gain-step-db", type=float, default=3.0,
                        help="Auto-tune gain decrement per step in dB (−6 dB ≈ ÷2 signal).")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1")

    try:
        if args.list_devices:
            return list_devices()
        if args.query:
            return query_timing(args)
        if args.auto_tune:
            return auto_tune(args)
        return capture(args)
    except CaptureError as exc:
        print(f"[ids_capture] ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ids_capture] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
