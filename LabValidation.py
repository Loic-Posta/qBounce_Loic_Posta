#!/usr/bin/env python3
"""
LabValidation.py - unattended end-to-end validation of the offline lab kit.

Run it once, walk away, read the report. It provisions the machine, builds the
C++ detector, proves the binary actually starts (i.e. that the OpenCV DLLs
resolve), and replays the exact commands the GUI's "Build" button issues - the
step that failed in the lab with [WinError 2].

    python LabValidation.py

Everything lands in lab_validation_report.txt next to this script, and every
subprocess has a timeout, so it can never hang overnight waiting for input.

Exit code 0 = everything the machine can do, it did.
"""

import io
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
CODE = BASE / "Code"
REPORT = BASE / "lab_validation_report.txt"

# Console stays ASCII (Windows consoles are often cp1252); the report is UTF-8.
_buf = io.StringIO()
_results: list[tuple[str, bool, str]] = []


def out(text: str = "") -> None:
    print(text)
    _buf.write(text + "\n")


def section(title: str) -> None:
    out("\n" + "=" * 72)
    out("  " + title)
    out("=" * 72)


def _mark(ok) -> str:
    return "SKIP" if ok is None else ("PASS" if ok else "FAIL")


def record(name: str, ok, detail: str = "") -> bool:
    """ok=True/False, or None for "could not be tested here" (SKIP).

    SKIP exists for the camera checks: without a camera attached they prove
    nothing, and turning that into a FAIL would train people to ignore a red
    report on the very machine where the camera checks are the point."""
    _results.append((name, ok, detail))
    out(f"  [{_mark(ok)}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def run(cmd, timeout=900, cwd=None, label=None):
    """Run a command, echo everything into the report, never hang."""
    shown = " ".join(str(c) for c in cmd)
    out(f"\n$ {shown}")
    started = time.time()
    try:
        proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                              capture_output=True, text=True, timeout=timeout,
                              errors="replace")
    except FileNotFoundError as exc:
        out(f"  !! could not start: {exc}")
        return None, time.time() - started
    except subprocess.TimeoutExpired:
        out(f"  !! TIMEOUT after {timeout}s")
        return None, time.time() - started
    elapsed = time.time() - started
    body = (proc.stdout or "") + (proc.stderr or "")
    for line in body.splitlines():
        out("  | " + line)
    out(f"  -> exit {proc.returncode} in {elapsed:.1f}s")
    return proc, elapsed


def venv_python() -> Path | None:
    for rel in (("venv", "Scripts", "python.exe"), ("venv", "bin", "python")):
        p = CODE.joinpath(*rel)
        if p.is_file():
            return p
    return None


def detector_exe() -> Path | None:
    name = "detector.exe" if os.name == "nt" else "detector"
    for rel in ("", "Release", "RelWithDebInfo", "Debug"):
        p = CODE / "build" / rel / name if rel else CODE / "build" / name
        if p.is_file():
            return p
    return None


# -- The introspection probe, executed BY THE VENV PYTHON ---------------------
# It imports PipelineController for real (needs cv2/PIL/matplotlib), so it can
# only run inside the venv. It prints machine-readable KEY=VALUE lines.
PROBE = r'''
import importlib.util, os, sys, json
from pathlib import Path
spec = importlib.util.spec_from_file_location("pc", sys.argv[1])
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)
print("CMAKE=" + str(pc.cmake_executable()))
print("OPENCV_DIR=" + str(pc.OPENCV_DIR))
print("DETECTOR=" + str(pc.find_detector_exe()))
print("PATH_HEAD=" + json.dumps(os.environ["PATH"].split(os.pathsep)[:4]))
rec = {}
class Fake:
    def _start_step(self, sid, cmds, cwd=None, on_success=None):
        rec["cwd"] = str(cwd)
        rec["cmds"] = [[str(x) for x in c] for c in cmds if not callable(c)]
pc.PipelineControllerApp._step_build(Fake())
print("BUILD_PLAN=" + json.dumps(rec))
'''


# -- The camera probe, executed BY THE VENV PYTHON with the camera attached ---
# Everything here needs real hardware, and is therefore exactly what could NOT
# be checked on a developer machine: the SDK's own entry points are present or
# absent depending on the ids_peak_ipl release, the Gain node is a factor on
# some sensors and dB on others, and a buffer pointer cannot be faked (the SWIG
# binding rejects anything but a genuine void* from the driver).
#
# It emits one "CHECK|name|PASS|FAIL|SKIP|detail" line per check, so the parent
# does not have to parse prose. With no camera attached every camera check comes
# back SKIP and the kit still validates.
CAMERA_PROBE = r'''
import sys, subprocess
from pathlib import Path

KIT = Path(sys.argv[1])
sys.path.insert(0, str(KIT / "Code" / "src"))

def check(name, ok, detail=""):
    print(f"CHECK|{name}|{'SKIP' if ok is None else ('PASS' if ok else 'FAIL')}|{detail}",
          flush=True)

def info(text):
    print("INFO|" + str(text), flush=True)

# ---- child mode: used to prove the "camera already in use" message ----------
if "--busy-child" in sys.argv:
    try:
        import ids_capture
        peak, _ = ids_capture._import_ids_peak()
        peak.Library.Initialize()
        mgr = peak.DeviceManager.Instance(); mgr.Update()
        devs = ids_capture._devices_as_list(mgr)
        if not devs:
            print("CHILD|nodevice"); raise SystemExit(0)
        ids_capture._open_device_or_explain(devs[0], peak)
        print("CHILD|opened")          # parent did not really hold it
    except ids_capture.CaptureError as exc:
        friendly = "already controlling it" in str(exc)
        print("CHILD|friendly" if friendly else "CHILD|captureerror")
    except Exception as exc:
        print("CHILD|raw|" + type(exc).__name__ + ": " + str(exc)[:120])
    raise SystemExit(0)

# ---- 1. bindings and which conversion entry points this release exposes -----
try:
    import ids_capture
    from ids_peak import ids_peak as peak
    from ids_peak_ipl import ids_peak_ipl as ipl
except Exception as exc:
    check("IDS peak bindings importable", False, f"{type(exc).__name__}: {exc}")
    for n in ("camera found", "camera opens", "Gain range readable",
              "one real frame converts to Mono8", "buffers survive Flush(DiscardAll)",
              "Live tab camera path (IdsCamera)"):
        check(n, None, "bindings missing")
    raise SystemExit(0)

def _dist(name):
    """Installed version of a package. The SWIG modules carry no __version__,
    and the release is the single most useful fact in this report: which entry
    points exist depends on it."""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "?"

check("IDS peak bindings importable", True,
      f"ids_peak {_dist('ids_peak')}, ids_peak_ipl {_dist('ids_peak_ipl')}")

has_static = hasattr(getattr(ipl, "Image", None), "CreateFromSizeAndBuffer")
has_module = hasattr(ipl, "Image_CreateFromSizeAndBuffer")
ext_pkg = None
for pkg in ("ids_peak", "ids_peak_ipl"):
    try:
        m = __import__(pkg + ".ids_peak_ipl_extension",
                       fromlist=["ids_peak_ipl_extension"])
        if hasattr(m, "BufferToImage"):
            ext_pkg = pkg
            break
    except Exception:
        pass
info(f"Image.CreateFromSizeAndBuffer={has_static}  "
     f"Image_CreateFromSizeAndBuffer={has_module}  BufferToImage in={ext_pkg}")
check("at least one buffer->image entry point exists",
      has_static or has_module or ext_pkg is not None,
      f"static={has_static} module={has_module} extension={ext_pkg or 'none'}")

# ---- 2. is a camera actually attached? --------------------------------------
peak.Library.Initialize()
device = node_map = stream = None
try:
    mgr = peak.DeviceManager.Instance(); mgr.Update()
    devices = ids_capture._devices_as_list(mgr)
    if not devices:
        check("camera found", None, "no IDS camera attached - camera checks skipped")
        for n in ("camera opens", "Gain range readable", "requested gain is applicable",
                  "one real frame converts to Mono8", "frame matches PayloadSize",
                  "buffers survive Flush(DiscardAll)", "camera-in-use message",
                  "Live tab camera path (IdsCamera)", "timing readable"):
            check(n, None, "no camera")
        raise SystemExit(0)
    check("camera found", True, ids_capture._device_label(devices[0]))

    try:
        device = ids_capture._open_device_or_explain(devices[0], peak)
        check("camera opens", True, "control access granted")
    except Exception as exc:
        check("camera opens", False, str(exc).splitlines()[0][:160])
        for n in ("Gain range readable", "requested gain is applicable",
                  "one real frame converts to Mono8", "frame matches PayloadSize",
                  "buffers survive Flush(DiscardAll)", "camera-in-use message",
                  "Live tab camera path (IdsCamera)", "timing readable"):
            check(n, None, "camera could not be opened")
        raise SystemExit(0)

    node_map = device.RemoteDevice().NodeMaps()[0]
    stream = device.DataStreams()[0].OpenDataStream()

    # ---- 3. Gain: the node that rejected the GUI's 0 at the lab -------------
    grange = ids_capture._node_range(node_map, "Gain")
    if grange is None:
        check("Gain range readable", None, "camera exposes no Gain range")
        check("requested gain is applicable", None, "no range to compare against")
    else:
        lo, hi = grange
        check("Gain range readable", True, f"[{lo}, {hi}]")
        # This is the lab failure in one line: is the GUI's 0 inside the range?
        # Either answer is fine now (out-of-range is clamped) -- what matters is
        # that the report SAYS which sensor this is, so nobody is surprised.
        check("requested gain is applicable", True,
              f"0 is {'inside' if lo <= 0 <= hi else 'BELOW the minimum -> clamped up to ' + str(lo)}"
              + f"; Gain is a {'dB offset' if lo <= 0 else 'multiplicative factor'} on this sensor")

    for name in ("ExposureTime", "AcquisitionFrameRate", "ResultingFrameRate"):
        r = ids_capture._node_range(node_map, name)
        info(f"{name} range: {r}")

    # AcquisitionFrameRateEnable is a BooleanNode on this family; SetCurrentEntry
    # can never set it. Report which kind it is here rather than at the beam.
    fre = ids_capture._node(node_map, "AcquisitionFrameRateEnable")
    if fre is None:
        info("AcquisitionFrameRateEnable: absent")
    else:
        info("AcquisitionFrameRateEnable: has SetCurrentEntry="
             + str(hasattr(fre, "SetCurrentEntry")) + " has SetValue="
             + str(hasattr(fre, "SetValue")))

    # ---- 4. THE untestable one: a real buffer through the real conversion ---
    ids_capture._set_enum(node_map, "AcquisitionMode", "Continuous")
    ids_capture._set_enum(node_map, "TriggerMode", "Off")
    ids_capture._set_enum(node_map, "PixelFormat", "Mono8")

    payload = int(node_map.FindNode("PayloadSize").Value())
    minbuf = int(stream.NumBuffersAnnouncedMinRequired())
    for _ in range(max(minbuf, 4)):
        stream.QueueBuffer(stream.AllocAndAnnounceBuffer(payload))

    tl = ids_capture._node(node_map, "TLParamsLocked")
    if tl is not None:
        tl.SetValue(1)
    stream.StartAcquisition()
    node_map.FindNode("AcquisitionStart").Execute()
    node_map.FindNode("AcquisitionStart").WaitUntilDone()

    buf = stream.WaitForFinishedBuffer(5000)
    try:
        # Which spellings does THIS driver actually accept on a real pointer?
        args = (buf.PixelFormat(), buf.BasePtr(), buf.Size(), buf.Width(), buf.Height())
        worked = []
        if has_static:
            try:
                ipl.Image.CreateFromSizeAndBuffer(*args); worked.append("Image.CreateFromSizeAndBuffer")
            except Exception as exc:
                info("Image.CreateFromSizeAndBuffer failed: " + str(exc)[:140])
        if has_module:
            try:
                ipl.Image_CreateFromSizeAndBuffer(*args); worked.append("Image_CreateFromSizeAndBuffer")
            except Exception as exc:
                info("Image_CreateFromSizeAndBuffer failed: " + str(exc)[:140])
        if ext_pkg:
            try:
                m = __import__(ext_pkg + ".ids_peak_ipl_extension",
                               fromlist=["ids_peak_ipl_extension"])
                m.BufferToImage(buf); worked.append(ext_pkg + ".BufferToImage")
            except Exception as exc:
                info("BufferToImage failed: " + str(exc)[:140])
        info("entry points that accept a real buffer: " + (", ".join(worked) or "NONE"))

        frame = ids_capture._mono8_numpy(ipl, buf)
        check("one real frame converts to Mono8", True,
              f"{frame.shape} {frame.dtype} min={frame.min()} max={frame.max()} "
              f"mean={frame.mean():.1f} via {worked[0] if worked else '?'}")
        expected = buf.Width() * buf.Height()
        check("frame matches PayloadSize", frame.size == expected,
              f"{frame.size} px vs {buf.Width()}x{buf.Height()}={expected}"
              + ("" if frame.size == expected else "  <-- line padding, conversion is wrong"))
    finally:
        stream.QueueBuffer(buf)

    # ---- 5. buffers after Flush(DiscardAll): the auto-tune stall ------------
    try:
        node_map.FindNode("AcquisitionStop").Execute()
        node_map.FindNode("AcquisitionStop").WaitUntilDone()
    except Exception:
        pass
    stream.StopAcquisition(peak.AcquisitionStopMode_Default)
    stream.Flush(peak.DataStreamFlushMode_DiscardAll)
    announced = list(stream.AnnouncedBuffers())
    requeued = 0
    for b in announced:
        try:
            stream.QueueBuffer(b); requeued += 1
        except Exception:
            pass
    second = None
    if requeued:
        stream.StartAcquisition()
        node_map.FindNode("AcquisitionStart").Execute()
        node_map.FindNode("AcquisitionStart").WaitUntilDone()
        try:
            b2 = stream.WaitForFinishedBuffer(5000)
            second = ids_capture._mono8_numpy(ipl, b2)
            stream.QueueBuffer(b2)
        except Exception as exc:
            info("second acquisition failed: " + str(exc)[:140])
    check("buffers survive Flush(DiscardAll)", second is not None,
          f"{len(announced)} announced, {requeued} re-queued, "
          + ("second acquisition delivered a frame" if second is not None
             else "NO frame after re-queue -- auto-tune would stall here"))

    # ---- 6. the "camera already in use" message, on real hardware -----------
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            str(KIT), "--busy-child"],
                           capture_output=True, text=True, timeout=180)
    verdict = "".join(child.stdout).strip().splitlines()
    verdict = verdict[-1] if verdict else ""
    if verdict.startswith("CHILD|friendly"):
        check("camera-in-use message", True,
              "a second process is told to close IDS peak Cockpit")
    elif verdict.startswith("CHILD|opened"):
        check("camera-in-use message", None,
              "this driver allows a second opener - nothing to explain")
    else:
        check("camera-in-use message", False, verdict[:160] or "no verdict")

finally:
    try:
        if node_map is not None:
            node_map.FindNode("AcquisitionStop").Execute()
    except Exception:
        pass
    try:
        if stream is not None:
            stream.StopAcquisition(peak.AcquisitionStopMode_Default)
            stream.Flush(peak.DataStreamFlushMode_DiscardAll)
            for b in stream.AnnouncedBuffers():
                stream.RevokeBuffer(b)
    except Exception:
        pass
    try:
        if node_map is not None:
            tl = ids_capture._node(node_map, "TLParamsLocked")
            if tl is not None:
                tl.SetValue(0)
    except Exception:
        pass
    try:
        peak.Library.Close()
    except Exception:
        pass

# ---- 7. the Live tab's own path, with the GUI's energy-preset gain ----------
# Separate process-level open: this is create_camera("ids", ...) exactly as
# LiveDetectionEngine calls it, including gain_db=0 from the Parameters tab.
try:
    from camera_interface import create_camera
    cam = create_camera("ids", device_index=0, exposure_us=None, gain_db=0.0)
    cam.open()
    try:
        frames = [cam.get_frame(timeout_ms=5000) for _ in range(3)]
    finally:
        cam.close()
    good = [f for f in frames if f is not None]
    check("Live tab camera path (IdsCamera)", len(good) == 3,
          f"{len(good)}/3 frames, shape={good[0].shape if good else '-'}")
except Exception as exc:
    check("Live tab camera path (IdsCamera)", False,
          f"{type(exc).__name__}: {str(exc).splitlines()[0][:150]}")

# ---- 8. duty cycle: what the Live tab's coverage label will show ------------
try:
    import argparse
    ns = argparse.Namespace(device_index=0)
    rc = ids_capture.query_timing(ns)
    check("timing readable", rc == 0, "exposure and frame rate read from the camera")
except Exception as exc:
    check("timing readable", False, f"{type(exc).__name__}: {str(exc)[:150]}")
'''


def main() -> int:
    started_all = time.time()
    section("LAB KIT VALIDATION")
    out(f"  when      : {datetime.now().isoformat(timespec='seconds')}")
    out(f"  kit root  : {BASE}")
    out(f"  platform  : {platform.platform()}")
    out(f"  machine   : {platform.machine()}  ({platform.processor()})")
    out(f"  python    : {sys.version.splitlines()[0]}")
    out(f"  python exe: {sys.executable}")
    out(f"  64-bit    : {sys.maxsize > 2**32}")

    # The wheels are cp314/win_amd64: a different Python cannot install them.
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    # A wheel only installs here if BOTH its interpreter tag and its platform tag
    # match. Checking the interpreter alone reported success on a Mac holding a
    # folder of win_amd64 wheels, none of which can be installed.
    plat = {"win32": "win_amd64", "darwin": "macosx", "linux": "linux"}.get(sys.platform, "")
    wheels = sorted((CODE / "wheels").glob("*.whl")) if (CODE / "wheels").is_dir() else []
    compiled = [w for w in wheels if "-none-any" not in w.name]
    matching = [w for w in compiled if tag in w.name and (not plat or plat in w.name)]
    if os.name != "nt":
        out("")
        out("  NOTE: this script validates the OFFLINE WINDOWS lab kit -- the venv")
        out("  built from Code/wheels, the bundled OpenCV, and the launcher that")
        out("  drives them. Those wheels are win_amd64, so on this machine the")
        out("  checks below will fail for the platform and not for the code.")
        out("  To use the pipeline here, build the detector with cmake and install")
        out("  the Python side from the network, as the README describes.")
        out("")

    section("STEP 1 - kit integrity")
    record("Code/wheels present", bool(wheels), f"{len(wheels)} wheels")
    record(f"wheels match this Python ({tag})",
           not compiled or bool(matching),
           f"{len(matching)}/{len(compiled)} compiled wheels tagged {tag}"
           + (f" and {plat}" if plat else "")
           + ("" if matching or not compiled else
              "  <-- WRONG PYTHON OR PLATFORM, offline install will fail here"))
    record("requirements-lab.txt", (CODE / "requirements-lab.txt").is_file())
    record("CMakeLists.txt", (CODE / "CMakeLists.txt").is_file())
    have_ocv = (BASE / "cpp_libs" / "opencv" / "build").is_dir()
    have_arc = bool(list((BASE / "cpp_libs").glob("opencv-*-windows.exe"))) \
        if (BASE / "cpp_libs").is_dir() else False
    record("OpenCV (extracted or bundled archive)", have_ocv or have_arc,
           "already extracted" if have_ocv else ("archive present" if have_arc else "ABSENT"))

    section("STEP 2 - WindowsLauncher --check (before provisioning)")
    run([sys.executable, str(BASE / "WindowsLauncher.py"), "--check"], timeout=300)

    section("STEP 3 - full provisioning (venv + wheels + OpenCV + compile)")
    proc, elapsed = run([sys.executable, str(BASE / "WindowsLauncher.py"), "--no-launch"],
                        timeout=3600)
    record("provisioning run", proc is not None and proc.returncode == 0,
           f"{elapsed:.0f}s" if proc else "did not complete")

    vpy = venv_python()
    record("venv python exists", vpy is not None, str(vpy or "-"))

    section("STEP 4 - WindowsLauncher --check (after provisioning)")
    if vpy:
        proc, _ = run([vpy, str(BASE / "WindowsLauncher.py"), "--check"], timeout=300)
        record("post-provisioning check clean", proc is not None and proc.returncode == 0)

    section("STEP 5 - does detector actually START? (proves DLLs resolve)")
    # This is the check that would have caught the missing opencv_world*.dll:
    # a binary that cannot load its DLLs fails to start at all.
    exe = detector_exe()
    record("detector binary built", exe is not None, str(exe or "not found"))
    if exe:
        proc, _ = run([exe, "--help"], timeout=120, cwd=exe.parent)
        record("detector --help runs (OpenCV DLLs load)",
               proc is not None and proc.returncode == 0,
               "started" if proc else "COULD NOT START - missing DLL?")

    section("STEP 6 - what the GUI's Build button will really do")
    plan = None
    if vpy:
        probe_file = BASE / "_probe_tmp.py"
        probe_file.write_text(PROBE)
        try:
            proc, _ = run([vpy, str(probe_file), str(BASE / "PipelineController.py")],
                          timeout=300, cwd=BASE)
            record("PipelineController imports in the venv",
                   proc is not None and proc.returncode == 0)
            if proc and proc.returncode == 0:
                import json
                for line in proc.stdout.splitlines():
                    if line.startswith("BUILD_PLAN="):
                        plan = json.loads(line.split("=", 1)[1])
                    if line.startswith("CMAKE="):
                        record("cmake resolved to an absolute path",
                               os.path.isabs(line.split("=", 1)[1]),
                               line.split("=", 1)[1])
                    if line.startswith("DETECTOR="):
                        record("GUI finds the detector",
                               line.split("=", 1)[1] not in ("None", ""),
                               line.split("=", 1)[1])
        finally:
            probe_file.unlink(missing_ok=True)

    section("STEP 7 - REPLAY the GUI Build step for real")
    # The exact failure from the lab screenshot happened here. Running the very
    # same argv the GUI produces is the only way to prove it is fixed.
    if plan and plan.get("cmds"):
        build_cwd = Path(plan["cwd"])
        build_cwd.mkdir(parents=True, exist_ok=True)
        ok = True
        for cmd in plan["cmds"]:
            proc, _ = run(cmd, timeout=1800, cwd=build_cwd)
            if proc is None or proc.returncode != 0:
                ok = False
                break
        record("GUI Build step succeeds", ok,
               "this is the step that failed with [WinError 2]")
    else:
        record("GUI Build step succeeds", False, "could not obtain the build plan")

    section("STEP 8 - the camera itself (needs the camera plugged in)")
    # Everything above can pass on a machine that has never seen a camera.
    # This step is the only one that exercises the SDK against real hardware:
    # which buffer->image entry point the driver accepts, what the Gain node
    # really is, whether buffers survive a flush, and whether the Live tab's
    # own camera class can stream. With no camera attached it reports SKIP.
    if vpy:
        cam_probe = BASE / "_camera_probe_tmp.py"
        cam_probe.write_text(CAMERA_PROBE, encoding="utf-8")
        try:
            proc, _ = run([vpy, str(cam_probe), str(BASE)], timeout=900, cwd=BASE)
            if proc is None:
                record("camera probe ran", False, "probe did not start or timed out")
            else:
                seen = 0
                for line in (proc.stdout or "").splitlines():
                    if not line.startswith("CHECK|"):
                        continue
                    _, name, verdict, detail = (line.split("|", 3) + [""])[:4]
                    record(name, None if verdict == "SKIP" else verdict == "PASS", detail)
                    seen += 1
                if seen == 0:
                    record("camera probe ran", False,
                           "probe produced no verdicts -- see the transcript above")
        finally:
            cam_probe.unlink(missing_ok=True)
    else:
        record("camera probe ran", None, "no venv python")

    section("SUMMARY")
    passed = sum(1 for _, ok, _ in _results if ok)
    skipped = sum(1 for _, ok, _ in _results if ok is None)
    for name, ok, detail in _results:
        out(f"  [{_mark(ok)}] {name}" + (f" -- {detail}" if detail else ""))
    out("")
    # A SKIP is neither a pass nor a failure: it is a check this machine could
    # not make. Counting it either way would lie about the run.
    out(f"  {passed}/{len(_results) - skipped} checks passed"
        + (f", {skipped} skipped" if skipped else "")
        + f" in {time.time() - started_all:.0f}s total")
    failed = [n for n, ok, _ in _results if ok is False]
    if failed:
        out(f"  FAILED: {', '.join(failed)}")
    if skipped:
        out("  SKIPPED (needs the camera plugged into THIS machine): "
            + ", ".join(n for n, ok, _ in _results if ok is None))

    REPORT.write_text(_buf.getvalue(), encoding="utf-8")
    print(f"\nFull report written to: {REPORT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
