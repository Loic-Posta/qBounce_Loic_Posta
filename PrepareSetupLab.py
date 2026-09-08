#!/usr/bin/env python3
"""
PrepareSetupLab.py — build the complete OFFLINE lab kit into ./ExportLab/

Run this ON A MACHINE WITH INTERNET (e.g. the Mac). It assembles everything
the offline Windows 10 lab machine needs, so the only manual action there is
copying ExportLab/ from the USB stick and running WindowsLauncher.py:

  ExportLab/
  ├── PipelineController.py         GUI
  ├── WindowsLauncher.py            one-click offline installer/builder/runner
  ├── PrepareSetupLab.py            this script (so the kit can rebuild itself)
  ├── README_LAB.txt                lab-machine instructions
  ├── cpp_requirements.txt          C++ dependency manifest (see below)
  ├── docs/                         user guide
  ├── cpp_libs/                     prebuilt OpenCV for Windows (~250 MB)
  └── Code/
      ├── CMakeLists.txt, requirements-lab.txt
      ├── src/                      all .py / .cpp / .h
      ├── wheels/                   every Python wheel (win_amd64) INCLUDING
      │                             cmake + ninja (build tools as pip wheels!)
      └── model_bundle.*            trained classifier (+ annotation sample)

About the "pipreqs for C++" question: no exact equivalent exists, because C++
has no import statement a scanner could crawl. The closest thing IS the
project's own CMakeLists.txt: its find_package() calls are the declared
dependency list. This script parses them (step 4) and turns each into an
action: OpenCV -> download the prebuilt Windows package; OpenMP/Threads ->
nothing to bundle, they ship with the compiler. The one thing that CANNOT be
bundled offline is the C++ compiler itself (Visual Studio Build Tools) — its
installer requires internet. See README_LAB.txt for the fallback.

Flags:
  --skip-wheels     don't (re)download Python wheels
  --skip-cpp-libs   don't download the OpenCV Windows package (~250 MB)
  --python-version  target Python on the lab machine (default: this one's)
"""

import argparse
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

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


# Build tools installable as plain pip wheels — this is what lets the lab
# machine get cmake/ninja OFFLINE, with no admin rights and no installers.
# NOTE: pipreqs deliberately NOT included — it depends on docopt, which has
# no wheel on PyPI (sdist only, so --only-binary rejects it), and it is a
# kit-BUILDER tool anyway: regenerating requirements on the offline machine
# would be pointless since no new wheels could be downloaded there.
EXTRA_TOOL_PKGS = ["cmake", "ninja"]

# Prebuilt OpenCV for Windows (MSVC binaries + headers). Pinned so the kit is
# reproducible; the .exe is a self-extracting 7-zip archive that
# WindowsLauncher.py extracts silently on the lab machine.
OPENCV_VERSION = "4.10.0"
OPENCV_URL = (f"https://github.com/opencv/opencv/releases/download/"
              f"{OPENCV_VERSION}/opencv-{OPENCV_VERSION}-windows.exe")

BASE = Path(__file__).resolve().parent
CODE = BASE / "Code"
CONSTRAINTS_NAME = "constraints-lab.txt"
EXPORT = BASE / "ExportLab"


def step(msg: str) -> None:
    print(f"\n-> {msg}")


def run(cmd: list, **kw) -> None:
    print(f"[EXEC] {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, patterns: tuple = ("*",)) -> int:
    """Copy matching files, skipping caches/metadata. Returns file count."""
    n = 0
    for pat in patterns:
        for f in src.rglob(pat):
            if not f.is_file():
                continue
            if any(p in ("__pycache__", ".git", ".DS_Store") for p in f.parts):
                continue
            if f.name.startswith("._") or f.name == ".DS_Store":
                continue
            copy_file(f, dst / f.relative_to(src))
            n += 1
    return n


# The IDS peak Python bindings. They are kept OUT of requirements-lab.txt (the
# analysis stack must install even when the camera side is unavailable) but ARE
# downloaded into wheels/ so the offline lab machine can get them like anything
# else.
#
# They were long assumed to be absent from PyPI and installable only from the
# SDK's own folder. That is wrong twice over: they are published on PyPI, and a
# standard IDS peak 26.06 install ships no Python wheel at all (verified on a
# real machine). Relying on Program Files therefore left the lab with no
# bindings whatsoever. The published wheels are abi3 -- ids_peak from cp310,
# ids_peak_ipl from cp311 -- so one download covers every later Python, 3.14
# included.
#
# The native SDK is still required on the lab machine: it provides the drivers
# and the GenTL producer the bindings talk to. Only the Python layer moves here.
IDS_PKGS = ["ids_peak", "ids_peak_ipl"]
# Kept out of requirements-lab.txt, but for a reason that is no longer
# "not on PyPI": WindowsLauncher installs that file with a single pip run
# that MUST succeed, then installs the bindings separately and never
# fatally. Letting them back into the file would put the whole analysis
# stack -- numpy, OpenCV, everything -- behind a wheel that may not exist
# for the lab machine's Python.
EXCLUDED_FROM_REQUIREMENTS: set = set(IDS_PKGS)
# Runtime deps pipreqs cannot resolve but the lab needs. tzdata is pandas'
# timezone database on Windows, imported by nothing. opencv_python is
# imported as "cv2" -- a module name pipreqs cannot map back to the PyPI
# distribution ("Package cv2 does not exist", it reports) -- so it falls
# out of every scan, and PipelineController.py dies on its top-level
# "import cv2" before the GUI ever opens.
ALWAYS_KEEP = {"tzdata", "opencv_python"}


def _normalise(name: str) -> str:
    """PyPI names vary in case and in - vs _; the requirements file uses one
    spelling, wheel filenames another. Fold both onto the same key."""
    return name.strip().lower().replace("-", "_")


def read_pins(req_path: Path) -> dict:
    """name -> pinned version, from a requirements or constraints file."""
    pins = {}
    if not req_path.is_file():
        return pins
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, _, version = line.partition("==")
            pins[_normalise(name)] = version.strip()
    return pins


def wheel_versions(wheels_dir: Path) -> dict:
    """name -> version, read off the wheel filenames actually downloaded.

    A wheel is <dist>-<version>-<python tag>-... , so the first two dash
    separated fields are exactly what --freeze needs to write back.
    """
    found = {}
    for w in wheels_dir.glob("*.whl"):
        parts = w.name[:-4].split("-")
        if len(parts) >= 2:
            found[_normalise(parts[0])] = parts[1]
    return found


def regenerate_requirements() -> None:
    """pipreqs scan of Code/src -> requirements-lab.txt, then sanitise.

    Raw pipreqs output is not shippable as-is: it can list the same package
    twice with conflicting versions (both an installed one and a PyPI-latest
    one — pip then refuses the file), includes the camera SDK bindings that
    must not gate the analysis install, and misses indirect runtime deps.
    So: dedupe, drop the excluded names, re-add the known runtime extras.

    No versions are written here. This file answers "what does the lab
    depend on"; constraints-lab.txt answers "which exact versions", and is
    never regenerated -- pipreqs only ever knows what PyPI serves today,
    which is the opposite of what a pinned kit wants."""
    step("Step 1: Scanning Python imports with pipreqs…")
    req_path = CODE / "requirements-lab.txt"
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pipreqs"],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, "-m", "pipreqs.pipreqs", str(CODE / "src"),
                        "--force", "--savepath", str(req_path)],
                       check=True)
    except Exception as exc:
        print(f"   [WARNING] pipreqs failed ({exc}) — keeping the existing "
              "requirements-lab.txt as-is.")
        return

    names: set = set()
    for line in req_path.read_text().splitlines():
        name = re.split(r"[=<>!~\[;]", line.strip())[0].strip().lower().replace("-", "_")
        if name and name not in EXCLUDED_FROM_REQUIREMENTS:
            names.add(name)
    dropped = EXCLUDED_FROM_REQUIREMENTS & {
        re.split(r"[=<>!~\[;]", l.strip())[0].strip().lower()
        for l in req_path.read_text().splitlines() if l.strip()}
    names |= ALWAYS_KEEP
    header = (f"# Regenerated by PrepareSetupLab.py -- edits here are lost.\n"
              f"# Exact versions live in {CONSTRAINTS_NAME}, next to this file.\n")
    req_path.write_text(header + "\n".join(sorted(names)) + "\n")
    print(f"   requirements-lab.txt: {len(names)} packages"
          + (f" (kept out of requirements: {', '.join(sorted(dropped))})" if dropped else ""))
    locked = read_pins(CODE / CONSTRAINTS_NAME)
    if not locked:
        print(f"   [WARNING] {CONSTRAINTS_NAME} is missing or empty — the kit "
              "will build against whatever PyPI serves today.")
        print("             Re-run with --freeze to write it.")
    else:
        loose = sorted(n for n in names if n not in locked)
        print(f"   {CONSTRAINTS_NAME}: {len(locked)} versions locked")
        if loose:
            print(f"   [WARNING] not locked: {', '.join(loose)} — re-run with --freeze.")


def copy_project() -> None:
    step("Step 2: Copying project files into ExportLab/…")
    for name in ("PipelineController.py", "WindowsLauncher.py", "PrepareSetupLab.py",
                 "LabValidation.py"):
        copy_file(BASE / name, EXPORT / name)
    if (BASE / "docs").is_dir():
        copy_tree(BASE / "docs", EXPORT / "docs")
    copy_file(CODE / "CMakeLists.txt", EXPORT / "Code" / "CMakeLists.txt")
    copy_file(CODE / "requirements-lab.txt", EXPORT / "Code" / "requirements-lab.txt")
    if (CODE / CONSTRAINTS_NAME).is_file():
        copy_file(CODE / CONSTRAINTS_NAME, EXPORT / "Code" / CONSTRAINTS_NAME)
    n = copy_tree(CODE / "src", EXPORT / "Code" / "src",
                  ("*.py", "*.cpp", "*.h"))
    print(f"   src/: {n} files")
    # Trained model + hand-labelled sample: without these the lab machine
    # cannot classify (and cannot conveniently retrain offline).
    for name in ("model_bundle.onnx", "model_bundle.meta.json",
                 "model_bundle.joblib", "detections_annotation_sample.csv"):
        p = CODE / name
        if p.is_file():
            copy_file(p, EXPORT / "Code" / name)
        else:
            print(f"   [WARNING] {name} not found — the lab kit will lack it.")


def download_wheels(python_version: str) -> None:
    tag = python_version.replace(".", "")
    step(f"Step 3: Downloading Python wheels (win_amd64, cp{tag})…")
    wheels = EXPORT / "Code" / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    req = EXPORT / "Code" / "requirements-lab.txt"
    base_cmd = [sys.executable, "-m", "pip", "download",
                "-d", str(wheels), "--only-binary=:all:"]
    if platform.system() != "Windows":
        base_cmd += ["--platform", "win_amd64", "--python-version", python_version]
    # Every pip run below shares one constraints file. Without it each
    # resolves on its own and the kit ends up with two numpys: the locked one
    # from requirements, and whatever latest version ids_peak_ipl drags in.
    # The lab then installs the lock and lets the IDS step upgrade over it --
    # a silent defeat of the locking this file exists to provide.
    lock = EXPORT / "Code" / CONSTRAINTS_NAME
    constrained = base_cmd + (["-c", str(lock)] if lock.is_file() else [])
    run(constrained + ["-r", str(req)])
    # Build tools as wheels: cmake + ninja mean the lab machine needs NO
    # system CMake install; pipreqs lets it regenerate requirements offline.
    run(constrained + EXTRA_TOOL_PKGS)
    # Camera bindings, non-fatally: a lab machine doing offline analysis of
    # recorded frames must still get a complete kit if these ever fail to
    # resolve (IDS yanking a release, a Python with no abi3 coverage, ...).
    try:
        run(constrained + IDS_PKGS)
    except subprocess.CalledProcessError as exc:
        print(f"   [WARNING] IDS peak bindings not downloaded ({exc}).")
        print("   The kit stays usable for analysis; LIVE camera steps will not "
              "work on the lab machine.")
    n = len(list(wheels.glob("*.whl")))
    print(f"   {n} wheels in {wheels}")


def freeze_pins() -> None:
    """Write constraints-lab.txt from every wheel that actually downloaded.

    Everything is locked, not just the names in requirements-lab.txt. The
    packages that drift hardest are precisely the transitive ones nobody
    lists: scipy arrives under scikit-learn, ml_dtypes under onnxruntime.
    Pinning only the direct dependencies leaves those free to move, which is
    how a kit rebuilt years later ends up with a stack the classifier was
    never validated against.

    This is the deliberate way to move the lock forward: build a kit, verify
    it on the lab machine, then freeze. Both copies are written -- the one in
    Code/ is the source of truth, the one in ExportLab/ is what the lab
    machine installs against.
    """
    step("Step 3b: Locking versions to the downloaded wheels…")
    found = wheel_versions(EXPORT / "Code" / "wheels")
    if not found:
        print("   [WARNING] no wheels to read — nothing locked.")
        return
    text = ("# Generated by PrepareSetupLab.py --freeze from the wheels it\n"
            "# downloaded. Every version the lab kit installs, transitive\n"
            "# dependencies included. Regenerate deliberately, never casually.\n"
            + "\n".join(f"{n}=={v}" for n, v in sorted(found.items())) + "\n")
    (CODE / CONSTRAINTS_NAME).write_text(text)
    (EXPORT / "Code" / CONSTRAINTS_NAME).write_text(text)
    print(f"   {len(found)} versions locked in {CONSTRAINTS_NAME}")


def scan_cpp_dependencies() -> list:
    """The 'pipreqs for C++': parse find_package() out of CMakeLists.txt."""
    step("Step 4: Scanning CMakeLists.txt for C++ dependencies…")
    text = (CODE / "CMakeLists.txt").read_text()
    pkgs = sorted(set(re.findall(r"find_package\(\s*(\w+)", text)))
    actions = {
        "OpenCV":  f"prebuilt package bundled in cpp_libs/ (v{OPENCV_VERSION}); "
                   "WindowsLauncher extracts it and passes -DOpenCV_DIR",
        "OpenMP":  "ships with the C++ compiler — nothing to bundle",
        "Threads": "ships with the C++ compiler — nothing to bundle",
    }
    lines = ["# C++ dependencies declared in Code/CMakeLists.txt (find_package)",
             "# and how this kit satisfies each one offline:", ""]
    for p in pkgs:
        note = actions.get(p, "UNKNOWN — add handling in PrepareSetupLab.py")
        lines.append(f"{p}: {note}")
        print(f"   {p}: {note}")
    lines += ["", "Compiler: NOT bundleable offline (see README_LAB.txt)."]
    (EXPORT / "cpp_requirements.txt").write_text("\n".join(lines) + "\n")
    return pkgs


def download_opencv() -> None:
    step(f"Step 5: Downloading prebuilt OpenCV {OPENCV_VERSION} for Windows (~250 MB)…")
    cpp_libs = EXPORT / "cpp_libs"
    cpp_libs.mkdir(parents=True, exist_ok=True)
    dest = cpp_libs / f"opencv-{OPENCV_VERSION}-windows.exe"
    if dest.is_file() and dest.stat().st_size > 100_000_000:
        print(f"   Already present: {dest.name}")
        return
    try:
        def hook(blocks, bs, total):
            done = blocks * bs
            if total > 0:
                sys.stdout.write(f"\r   {done/1e6:6.0f} / {total/1e6:.0f} MB")
                sys.stdout.flush()
        urllib.request.urlretrieve(OPENCV_URL, dest, reporthook=hook)
        print(f"\n   Saved: {dest.name}")
    except Exception as exc:
        print(f"\n   [WARNING] OpenCV download failed: {exc}")
        print("   Download it manually from opencv.org into ExportLab/cpp_libs/.")


def write_data_skeleton() -> None:
    """Create the empty dark/ and signal/ folders the GUI defaults point at.

    The kit ships with no frames, so on a fresh machine those paths do not
    exist and step 3 fails with "Dark folder not found" -- which reads as a
    broken install rather than "you have not recorded anything yet". Shipping
    the folders, each with a note saying what belongs in it, makes the intended
    workflow visible from the file tree alone."""
    notes = {
        "dark": ("DARK frames go here.\n\n"
                 "Images recorded with the camera running and the neutron beam\n"
                 "OFF. They measure the sensor's own noise, which every later\n"
                 "step subtracts. A few hundred frames is typical.\n\n"
                 "Record them from the GUI: Live tab -> 'Record raw frames to'\n"
                 "-> point it at this folder, beam off.\n\n"
                 "Accepted: .tiff .tif .bmp .png\n"),
        "signal": ("SIGNAL frames go here.\n\n"
                   "Images recorded with the beam ON -- the actual measurement.\n"
                   "Step 4 looks for particle clusters in these, against the\n"
                   "background model built from dark/.\n\n"
                   "Accepted: .tiff .tif .bmp .png\n"),
    }
    for name, text in notes.items():
        d = EXPORT / "Code" / "Data" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "PUT_FRAMES_HERE.txt").write_text(text)
    print("   Code/Data/{dark,signal}/ created with placement notes")


def write_launchers() -> None:
    """Two double-clickable entry points, so the lab operator never has to
    remember a path: one that provisions the machine (run once), one that just
    starts the GUI (run every day — seconds instead of minutes)."""
    (EXPORT / "CheckSetup.bat").write_text(
        "@echo off\r\n"
        "rem Verifies the lab machine setup. Installs nothing, builds nothing.\r\n"
        'cd /d "%~dp0"\r\n'
        'if exist "Code\\venv\\Scripts\\python.exe" (\r\n'
        '  "Code\\venv\\Scripts\\python.exe" WindowsLauncher.py --check\r\n'
        ") else (\r\n"
        "  python WindowsLauncher.py --check\r\n"
        ")\r\n"
        "pause\r\n")
    (EXPORT / "RunLab.bat").write_text(
        "@echo off\r\n"
        "rem DAILY start: the GUI only. Assumes WindowsLauncher.py already ran once.\r\n"
        'cd /d "%~dp0"\r\n'
        'if not exist "Code\\venv\\Scripts\\python.exe" (\r\n'
        "  echo No venv found - run WindowsLauncher.py once first.\r\n"
        "  pause\r\n"
        "  exit /b 1\r\n"
        ")\r\n"
        '"Code\\venv\\Scripts\\python.exe" PipelineController.py\r\n'
        "pause\r\n")
    print("   CheckSetup.bat + RunLab.bat written")


def write_readme() -> None:
    (EXPORT / "README_LAB.txt").write_text(f"""\
qBOUNCE — OFFLINE LAB KIT
=========================

On the Windows 10 lab machine:
  1. Copy this whole ExportLab folder to e.g. C:\\qBounce
  2. ONCE, to provision the machine, open PowerShell in that folder and run:
       & "C:\\Program Files\\Python314\\python.exe" WindowsLauncher.py
     (adjust to the machine's Python 3.14 path)
  3. EVERY DAY AFTER THAT: just double-click RunLab.bat — it starts the GUI
     directly (seconds). The GUI's "2 · Build C++ Pipeline" button recompiles
     the detector on its own, so the launcher is not needed again.
     Double-click CheckSetup.bat (= WindowsLauncher.py --check) if you want to
     re-verify the install without building or launching anything.

BEFORE THE FIRST BEAM TIME, with the CAMERA PLUGGED IN and IDS peak Cockpit
CLOSED, run the full validation once:
       Code\\venv\\Scripts\\python.exe LabValidation.py
It provisions, builds, replays the GUI's Build step, and then drives the
camera itself: it grabs one real frame and converts it, reads the Gain node's
true range (on the U3-380xACP-M Gain is a FACTOR with a minimum of 1.0, so the
GUI's 0 means "lowest gain" and is clamped up — the report says so), checks
buffers survive a flush, and streams three frames through the Live tab's own
camera class. Anything it cannot test without the camera it marks SKIP rather
than failing. Everything lands in lab_validation_report.txt — send that file
if something is wrong; it contains every command and its output.

The launcher then works fully OFFLINE:
  - creates Code\\venv and installs every Python wheel from Code\\wheels
  - installs cmake + ninja INTO the venv (they are pip wheels — no admin)
  - extracts the bundled OpenCV package from cpp_libs\\ on first run and
    passes -DOpenCV_DIR to CMake automatically
  - compiles the C++ detector, then launches the GUI

The ONE thing this kit cannot install offline is the C++ COMPILER
(Visual Studio Build Tools — its installer needs internet). If `cl` is not
on the machine:
  - on any online Windows PC, download "Build Tools for Visual Studio" and
    create an offline layout (vs_buildtools.exe --layout D:\\vslayout),
    copy it over USB, install the "Desktop development with C++" workload.
  - the launcher detects a missing compiler and prints exactly this advice.

C++ dependency manifest: see cpp_requirements.txt (generated from
CMakeLists.txt's find_package calls — the closest thing to a pipreqs for
C++). OpenCV bundled: {OPENCV_VERSION}.
""")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the offline lab kit (ExportLab/).")
    ap.add_argument("--skip-wheels", action="store_true")
    ap.add_argument("--skip-cpp-libs", action="store_true")
    ap.add_argument("--freeze", action="store_true",
                    help=f"after downloading, rewrite {CONSTRAINTS_NAME} to "
                         "lock every version that landed in wheels/ "
                         "(the deliberate way to move the lock)")
    ap.add_argument("--python-version",
                    default=f"{sys.version_info.major}.{sys.version_info.minor}",
                    help="Python version on the lab machine (e.g. 3.14)")
    args = ap.parse_args()

    print("=" * 50)
    print("   LAB KIT BUILDER  ->  ExportLab/")
    print("=" * 50)
    if not CODE.is_dir():
        sys.exit(f"[ERROR] {CODE} not found — run from the project root.")
    EXPORT.mkdir(exist_ok=True)

    regenerate_requirements()
    copy_project()
    if not args.skip_wheels:
        download_wheels(args.python_version)
        if args.freeze:
            freeze_pins()
    elif args.freeze:
        sys.exit("[ERROR] --freeze needs the wheels: drop --skip-wheels.")
    scan_cpp_dependencies()
    if not args.skip_cpp_libs:
        download_opencv()
    write_data_skeleton()
    write_launchers()
    write_readme()

    total = sum(f.stat().st_size for f in EXPORT.rglob("*") if f.is_file())
    print(f"\n[SUCCESS] ExportLab ready: {total/1e6:.0f} MB — copy it to the USB stick.")


if __name__ == "__main__":
    main()
