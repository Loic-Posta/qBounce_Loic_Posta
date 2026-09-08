#!/usr/bin/env python3
"""
WindowsLauncher.py - offline installer / builder / launcher for the lab machine.

Three usages, deliberately separated so the slow "provision the machine" work
is not repeated every single day:

  python WindowsLauncher.py --check     verify the setup ONLY (~10 s, no build,
                                        no GUI). Run this once on arrival, or
                                        whenever something looks wrong.
  python WindowsLauncher.py             full run: venv + packages + OpenCV +
                                        CMake build + launch the GUI.
  python WindowsLauncher.py --no-launch  everything except starting the GUI.

Once a full run has succeeded, the normal DAILY start is simply
      Code\\venv\\Scripts\\python.exe PipelineController.py
   (or just double-click RunLab.bat)
because the GUI can configure and rebuild the C++ detector by itself - it
resolves cmake inside the venv and OpenCV inside cpp_libs\\ on its own.
"""

import argparse
import glob
import os
import shutil
import sys
import subprocess
import time

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


# Directories that are generated ON Windows (or hold hundreds of MB of wheels)
# and therefore can never contain macOS AppleDouble files - skipping them keeps
# the metadata sweep to a fraction of a second instead of walking ~40 000 files.
SKIP_DIRS = {"venv", "build", "__pycache__", ".git", ".vs"}

# requirement name -> module actually imported (only where they differ)
IMPORT_NAMES = {
    "opencv_python": "cv2",
    "opencv-python": "cv2",
    "pillow": "PIL",
    "scikit_learn": "sklearn",
    "scikit-learn": "sklearn",
    "tzdata": None,          # data-only package, nothing to import
}


def clean_mac_metadata(base_dir):
    """Recursively finds and deletes useless macOS '._' metadata files."""
    print("\n-> Step 0: Cleaning macOS '._' metadata files...")
    cleaned_count = 0
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                   and not (os.path.basename(root) == "cpp_libs" and d == "opencv")]
        for file in files:
            if file.startswith("._"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    cleaned_count += 1
                except Exception as e:
                    print(f"   [Warning] Could not remove {file_path}: {e}")
    if cleaned_count > 0:
        print(f"   Removed {cleaned_count} metadata file(s).")
    else:
        print("   No macOS metadata files found.")


def run_command(cmd, cwd=None):
    """Executes a system command robustly and handles errors."""
    print(f"\n[EXEC] : {' '.join(cmd)}")
    try:
        # shell=False to prevent character escaping bugs on Windows
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"\n[ERROR] Tool not found. Ensure '{cmd[0]}' is installed and added to Windows PATH (e.g., CMake).")
        sys.exit(1)


def msvc_installed() -> bool:
    """Checking `cl` on PATH is NOT enough (it is only on PATH inside a
    "Developer Prompt"), so ask vswhere whether the MSVC C++ toolset is
    installed - that is exactly what CMake's Visual Studio generator uses."""
    vswhere = os.path.expandvars(
        r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
    if not os.path.exists(vswhere):
        return False
    try:
        out = subprocess.run(
            [vswhere, "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-latest", "-property", "installationPath"],
            capture_output=True, text=True, check=False).stdout.strip()
        return bool(out)
    except Exception:
        return False


def check_python_matches_wheels(code_dir, python_exe=None) -> None:
    """Fail EARLY and clearly when the interpreter cannot use the bundled wheels.

    The kit ships compiled wheels for one exact CPython version (cp314 today).
    Run it with another one and pip reports 'no matching distribution' once per
    package, after having already built a venv -- a wall of noise that hides the
    single real cause. Worse, that unusable venv then gets REUSED on the next
    run, because step 1 only checks whether the folder exists."""
    # Only wheels LOCKED to one interpreter version matter. A filename is
    # name-version-pytag-abitag-platform.whl; abitag "abi3" (e.g.
    # opencv_python-5.0.0.93-cp37-abi3-win_amd64.whl) means stable ABI and works
    # on that version *and every later one*, so it must not drive the verdict --
    # nor appear in the advice, or it would suggest installing Python 3.7.
    compiled = []
    for w in glob.glob(os.path.join(code_dir, "wheels", "*.whl")):
        parts = os.path.basename(w)[:-4].split("-")
        if len(parts) >= 5 and parts[-2].startswith("cp"):
            compiled.append(w)
    if not compiled:
        return
    if python_exe is None:
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        who = f"this Python ({sys.version.split()[0]})"
    else:
        out = subprocess.run(
            [python_exe, "-c", "import sys;print(f'cp{sys.version_info.major}"
                               "{sys.version_info.minor}',sys.version.split()[0])"],
            capture_output=True, text=True, check=False).stdout.split()
        if len(out) != 2:
            return
        tag, who = out[0], f"the venv Python ({out[1]})"
    if any(tag in os.path.basename(w) for w in compiled):
        return

    needed = sorted({p for w in compiled for p in os.path.basename(w).split("-")
                     if p.startswith("cp") and p[2:].isdigit()})
    print(f"\n[ERROR] The bundled wheels do not match {who}.")
    print(f"        wheels are built for : {', '.join(needed) or 'unknown'}")
    print(f"        you are running      : {tag}")
    print("\nInstall the matching Python (64-bit) and re-run this launcher with it,")
    print("or rebuild the kit with:  python PrepareSetupLab.py --python-version 3.XX")
    print("If a venv was already created with the wrong Python, delete Code\\venv first.")
    sys.exit(1)


def find_ids_binding_dir():
    """Locate the IDS peak Python bindings, wherever this SDK release put them.

    A single hard-coded path is not safe here: it was right for one IDS peak
    version and is wrong for others. Verified on IDS peak 26.06, a *standard*
    install ships NO Python bindings at all -- no binding\\ folder, no wheel --
    because they are a separate component. So search the plausible roots and
    look for an actual wheel, then say what was searched when nothing is found;
    otherwise the failure reads as "the camera is broken" instead of "the
    bindings component was never installed"."""
    roots = [r"C:\Program Files\IDS\ids_peak",
             r"C:\Program Files (x86)\IDS\ids_peak",
             r"C:\Program Files\IDS",
             os.environ.get("IDS_PEAK_PYTHON_DIR", "")]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        hits = glob.glob(os.path.join(root, "**", "ids_peak*.whl"), recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None


def have_cxx_compiler() -> bool:
    """A compiler this kit can actually build with.

    On Windows that means MSVC and nothing else -- see the note at the call
    site. Elsewhere (the dev Mac, a Linux box) any working C++ compiler is
    fine, because there the OpenCV in use was built by that same toolchain."""
    if os.name == "nt":
        return msvc_installed() or bool(shutil.which("cl"))
    return bool(shutil.which("c++") or shutil.which("g++") or shutil.which("clang++"))


def compiler_missing_advice() -> None:
    print("\n[ERROR] No C++ compiler on this machine - CMake cannot build")
    print("the detector (this is the one thing the offline kit cannot bundle).")
    print("Install 'Build Tools for Visual Studio', workload:")
    print("   'Desktop development with C++'")
    print(" - Machine WITH internet: download vs_BuildTools.exe from")
    print("   visualstudio.microsoft.com and install that workload (~2 GB).")
    print(" - Strictly OFFLINE: on an online Windows PC run:")
    print("     vs_buildtools.exe --layout D:\\vslayout ^")
    print("       --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended")
    print("   copy D:\\vslayout over USB, run its vs_buildtools.exe here.")
    print("Then simply re-run WindowsLauncher.py.")
    print("(Note: MinGW is NOT a substitute here - the bundled OpenCV")
    print(" binaries are MSVC-only and cannot link against MinGW.)")


def _venv_bin(venv_dir, exe):
    """Path to an executable inside a venv, on whichever platform we are.

    Windows puts them in Scripts/ with a .exe suffix, POSIX in bin/ without one.
    Hard-coding the Windows form made this script create a venv and then fail to
    find the interpreter it had just created, on macOS and on Linux alike."""
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", exe + ".exe")
    return os.path.join(venv_dir, "bin", exe)


def find_opencv_dir(base_dir):
    """OpenCV C++ (headers + prebuilt libs - opencv-python is NOT enough for
    the C++ detector). Search order: env var, standard install, already
    extracted kit copy."""
    import glob as _glob
    for c in (os.environ.get("OPENCV_DIR", ""),
              r"C:\opencv\build",
              os.path.join(base_dir, "cpp_libs", "opencv", "build")):
        if not c or not os.path.isdir(c):
            continue
        # Prefer the per-toolset config under build/x64/vcNN/lib over the one at
        # the root of build/. The root config refuses to load when the compiler
        # is newer than every toolset in the pack, which is what a 2026 MSVC
        # does to a vc16-only Windows Pack; the versioned one links fine.
        for sub in sorted(_glob.glob(os.path.join(c, "x64", "vc*", "lib")), reverse=True):
            if os.path.isfile(os.path.join(sub, "OpenCVConfig.cmake")):
                return sub
        if os.path.isfile(os.path.join(c, "OpenCVConfig.cmake")):
            return c
    return None


def find_detector_exe(code_dir):
    """MSVC's generator is multi-config: build\\Release\\detector.exe. Ninja and
    Unix Makefiles put it straight in build\\. Accept either."""
    build_dir = os.path.join(code_dir, "build")
    for rel in ("detector.exe", os.path.join("Release", "detector.exe"),
                os.path.join("RelWithDebInfo", "detector.exe")):
        p = os.path.join(build_dir, rel)
        if os.path.isfile(p):
            return p
    return None


# ------------------------------------------------------------------------------
#  --check : verify the machine, change nothing, start nothing
# ------------------------------------------------------------------------------

def do_check(base_dir) -> int:
    code_dir = os.path.join(base_dir, "Code")
    venv_python = _venv_bin(os.path.join(code_dir, "venv"), "python")
    if os.name != "nt" and not os.path.exists(venv_python):
        # POSIX fallback, for running this check on the dev Mac/Linux box only.
        # It must NOT apply on Windows: when the venv is simply missing, falling
        # through would print "Code\venv\bin\python" as the thing to look for --
        # a path that can never exist there, sending the lab user hunting for it.
        venv_python = os.path.join(code_dir, "venv", "bin", "python")
    requirements_path = os.path.join(code_dir, "requirements-lab.txt")

    print("=" * 62)
    print("   SETUP CHECK - read-only, nothing is installed or built")
    print("=" * 62)
    problems = []

    def report(ok, label, detail, fatal=True):
        tag = "OK " if ok else ("FAIL" if fatal else "WARN")
        print(f" [{tag}] {label:<26} {detail}")
        if not ok and fatal:
            problems.append(label)

    # 1. venv
    report(os.path.exists(venv_python), "Virtual environment", venv_python)
    if not os.path.exists(venv_python):
        print("\n=> Run WindowsLauncher.py (without --check) to create it.")
        return 1

    # 2. Python packages - import them for real, that is the only honest test
    names = []
    if os.path.isfile(requirements_path):
        with open(requirements_path) as fh:
            for line in fh:
                pkg = line.strip().split("=")[0].split(">")[0].split("<")[0].strip()
                if pkg and not pkg.startswith("#"):
                    names.append(pkg)
    mods = sorted({IMPORT_NAMES.get(n, n) for n in names} - {None})
    probe = ("import importlib,sys\n"
             "bad=[]\n"
             f"for m in {mods!r}:\n"
             "    try: importlib.import_module(m)\n"
             "    except Exception as e: bad.append(f'{m} ({type(e).__name__})')\n"
             "print('MISSING:' + ','.join(bad))\n")
    out = subprocess.run([venv_python, "-c", probe],
                         capture_output=True, text=True).stdout.strip()
    missing = out.replace("MISSING:", "").strip()
    report(not missing, "Python packages",
           f"{len(mods)} checked" if not missing else f"missing: {missing}")

    # 3. IDS camera bindings - optional (only needed for live acquisition)
    ids = subprocess.run([venv_python, "-c", "import ids_peak"],
                         capture_output=True, text=True).returncode == 0
    report(ids, "IDS peak bindings",
           "importable" if ids else "absent - live camera steps will not work",
           fatal=False)

    # 4. C++ toolchain
    cmake = _venv_bin(os.path.join(code_dir, "venv"), "cmake")
    if not os.path.isfile(cmake):
        cmake = shutil.which("cmake") or ""
    report(bool(cmake), "CMake", cmake or "not found (venv nor PATH)")

    opencv_dir = find_opencv_dir(base_dir)
    archives = glob.glob(os.path.join(base_dir, "cpp_libs", "opencv-*-windows.exe"))
    report(bool(opencv_dir), "OpenCV C++",
           opencv_dir or ("not extracted yet - archive present, a full run will "
                          "extract it" if archives else "MISSING and no archive in cpp_libs\\"))

    dll_dirs = glob.glob(os.path.join(opencv_dir or "", "x64", "vc*", "bin"))
    report(bool(dll_dirs), "OpenCV runtime DLLs",
           dll_dirs[0] if dll_dirs else "not found - detector.exe will fail to start",
           fatal=bool(opencv_dir))

    has_cc = have_cxx_compiler()
    detail = "installed" if has_cc else "NOT installed"
    if not has_cc and os.name == "nt" and shutil.which("g++"):
        # Say it out loud: someone with MinGW on PATH will otherwise be sure
        # this check is simply wrong about their machine.
        detail += " -- MinGW/g++ found, but it CANNOT link the vc16 OpenCV"
    report(has_cc, "C++ compiler (MSVC)", detail)

    exe = find_detector_exe(code_dir)
    report(bool(exe), "detector.exe",
           exe or "not built yet - run WindowsLauncher.py without --check",
           fatal=False)

    print("-" * 62)
    if problems:
        print(f" {len(problems)} problem(s): {', '.join(problems)}")
        if not has_cc:
            compiler_missing_advice()
        print("\n=> Fix the above, then run: python WindowsLauncher.py")
        return 1
    print(" Setup is complete.")
    print(" Daily use: Code\\venv\\Scripts\\python.exe PipelineController.py")
    print(" (or double-click RunLab.bat) - no need to re-run this launcher.")
    return 0


# ------------------------------------------------------------------------------
#  Full provisioning run
# ------------------------------------------------------------------------------

def check_wheels_present(code_dir) -> None:
    """Refuse early, and by name, when this is the source copy rather than the kit.

    The submitted folder and the offline kit look almost identical: same
    scripts, same Code/src, same launcher. The one difference is Code/wheels/,
    ~200 MB of third-party binaries that are never submitted because they are
    re-downloadable. Run this launcher from the submitted copy and pip is the
    first thing to notice, several steps later, with a resolver error about
    joblib that says nothing about which folder you are standing in.
    """
    if glob.glob(os.path.join(code_dir, "wheels", "*.whl")):
        return
    print("\n[ERROR] No Python wheels in Code\\wheels -- this is the source copy,")
    print("        not the offline lab kit. They are ~200 MB of third-party")
    print("        binaries, deliberately left out of what gets submitted.")
    print("\nOn a machine WITH internet, build the kit from this folder:")
    print("        python PrepareSetupLab.py --python-version 3.14")
    print("\nThat writes ExportLab\\ next to this script. Run the launcher from")
    print("THERE, not from here -- or copy that ExportLab folder to the lab")
    print("machine and run it there.")
    print("\n(--skip-deps bypasses this, for a venv that is already populated.)")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the setup only: no install, no build, no GUI")
    ap.add_argument("--no-launch", action="store_true",
                    help="install and build, but do not start the GUI")
    ap.add_argument("--skip-deps", action="store_true",
                    help="skip the pip step (fast re-run once packages are in)")
    args = ap.parse_args()

    # Dynamically determine paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.check:
        sys.exit(do_check(base_dir))

    print("==================================================")
    print("   AUTOMATIC PROJECT LAUNCHER (WINDOWS 10)       ")
    print("==================================================")

    t_step = time.time()

    def took(label):
        nonlocal t_step
        print(f"   [{label} took {time.time() - t_step:.1f} s]")
        t_step = time.time()

    # Run the cleanup routine before doing anything else
    clean_mac_metadata(base_dir)
    took("step 0")

    code_dir = os.path.join(base_dir, "Code")
    venv_dir = os.path.join(code_dir, "venv")
    venv_python = _venv_bin(venv_dir, "python")

    requirements_path = os.path.join(code_dir, "requirements-lab.txt")
    wheels_path = os.path.join(code_dir, "wheels")
    pipeline_script = os.path.join(base_dir, "PipelineController.py")

    # Standing in the right folder comes before everything else: without the
    # wheels there is nothing to check the interpreter against.
    if not args.skip_deps:
        check_wheels_present(code_dir)
    # Check the interpreter BEFORE building anything with it.
    check_python_matches_wheels(code_dir)

    # 1. Create virtual environment if it does not exist
    if os.path.exists(venv_python):
        # An existing venv is reused -- but only if it can actually install the
        # wheels. A venv left behind by a wrong-Python run would otherwise be
        # picked up silently and fail again at pip, every single time.
        check_python_matches_wheels(code_dir, venv_python)
    if not os.path.exists(venv_dir):
        print("\n-> Step 1: Creating virtual environment (venv)...")
        # Uses the specific Python version called by the user
        run_command([sys.executable, "-m", "venv", venv_dir], cwd=base_dir)
    else:
        print("\n-> Step 1: Virtual environment (venv) already exists.")
    took("step 1")

    # 2. Install dependencies via local wheels
    if args.skip_deps:
        print("\n-> Step 2: SKIPPED (--skip-deps)")
    else:
        print("\n-> Step 2: Installing Python packages (Offline Mode via Wheels)...")
        # constraints-lab.txt is the exact version list the kit was built and
        # verified with, transitive dependencies included. Applied to BOTH pip
        # runs below: without it the IDS step, which resolves separately, can
        # upgrade numpy over the version the classifier was validated against.
        lock = os.path.join(os.path.dirname(requirements_path), "constraints-lab.txt")
        lock_args = ["-c", lock] if os.path.exists(lock) else []
        # Pointing directly to the venv python executable avoids fragile 'activate' steps in scripts
        # The analysis stack first, on its own. This must succeed.
        run_command([
            venv_python, "-m", "pip", "install",
            "--no-index",
            f"--find-links={wheels_path}",
            "-r", requirements_path,
        ] + lock_args)

        # The IDS peak bindings SECOND, and never fatally. They used to share
        # the command above, which made them a single point of failure for the
        # whole machine: the bindings are wheels built for specific CPython
        # versions, so an SDK that stops at (say) cp312 makes pip fail on a
        # Python 3.14 lab machine -- taking numpy, OpenCV and everything else
        # down with it. Only the live-camera steps actually need them; offline
        # analysis of recorded frames does not.
        # Prefer the kit's own wheels/ (PrepareSetupLab downloads the bindings
        # from PyPI, where they are published as abi3 and cover every recent
        # Python). Fall back to the SDK folder for kits built before that.
        ids_wheel_dir = (wheels_path
                         if glob.glob(os.path.join(wheels_path, "ids_peak*.whl"))
                         else find_ids_binding_dir())
        if ids_wheel_dir:
            print(f"   [IDS] System folder detected : {ids_wheel_dir}")
            ids = subprocess.run(
                [venv_python, "-m", "pip", "install", "--no-index",
                 f"--find-links={ids_wheel_dir}"] + lock_args
                + ["ids_peak", "ids_peak_ipl"],
                capture_output=True, text=True)
            if ids.returncode == 0:
                print("   [IDS] bindings installed.")
            else:
                tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
                print("   [IDS] [WARNING] bindings could NOT be installed "
                      f"(no wheel for this Python, {tag}?).")
                print("   [IDS] Analysis of recorded frames still works; only the")
                print("   [IDS] LIVE camera steps are unavailable. To fix, install an")
                print("   [IDS] IDS peak release that ships bindings for this Python,")
                print("   [IDS] or run the kit on the Python version IDS supports.")
                for line in (ids.stdout + ids.stderr).splitlines()[-4:]:
                    print(f"   [IDS]   | {line}")
        else:
            print("   [IDS] No Python bindings found under C:\\Program Files\\IDS.")
            print("   [IDS] The standard IDS peak installer does NOT include them:")
            print("   [IDS] re-run it and select the Python bindings component, or")
            print("   [IDS] set IDS_PEAK_PYTHON_DIR to the folder holding the wheel.")
            print("   [IDS] Only the LIVE camera steps need this; analysis of")
            print("   [IDS] recorded frames works without it.")
    took("step 2")

    # 3. Verify/provision the C++ toolchain - everything below works OFFLINE
    #    from what PrepareSetupLab.py bundled into the kit.
    print("\n-> Step 3: Checking C++ build tools...")

    # 3a. cmake: prefer one installed INTO the venv (pip wheel - no admin, no
    #     system install). If absent, install it from the offline wheels.
    venv_cmake = _venv_bin(venv_dir, "cmake")
    if not os.path.exists(venv_cmake):
        print("   cmake not in venv -> installing from offline wheels...")
        try:
            run_command([venv_python, "-m", "pip", "install", "--no-index",
                         f"--find-links={wheels_path}", "cmake", "ninja"])
        except SystemExit:
            print("   [WARNING] cmake wheel missing from Code\\wheels - "
                  "falling back to a system-wide 'cmake' if present.")
    cmake_exe = venv_cmake if os.path.exists(venv_cmake) else "cmake"
    print(f"   cmake: {cmake_exe}")

    # 3b. OpenCV C++: if not already available anywhere, silently extract the
    #     bundled self-extracting archive from cpp_libs\ (7-zip SFX: -o<dir> -y).
    opencv_dir = find_opencv_dir(base_dir)
    if opencv_dir is None:
        archives = glob.glob(os.path.join(base_dir, "cpp_libs", "opencv-*-windows.exe"))
        if archives:
            print(f"   Extracting bundled OpenCV: {os.path.basename(archives[0])} ...")
            run_command([archives[0], f"-o{os.path.join(base_dir, 'cpp_libs')}", "-y"])
            extracted = os.path.join(base_dir, "cpp_libs", "opencv", "build")
            if os.path.isdir(extracted):
                opencv_dir = extracted
    if opencv_dir:
        print(f"   OpenCV C++: {opencv_dir}")
    else:
        print("   [WARNING] OpenCV C++ not found and no bundled archive in "
              "cpp_libs\\ - CMake will fail at find_package(OpenCV). "
              "Re-run PrepareSetupLab.py with internet, or install OpenCV to C:\\opencv.")

    # 3c. Compiler: the one piece this kit cannot bundle offline.
    #     On Windows this MUST be MSVC. Accepting g++ here (as an earlier version
    #     did) is actively harmful: a machine with MinGW installed would sail
    #     past this gate and fail much later inside CMake, with a link error
    #     about std::__cxx11::basic_string that says nothing about the real
    #     cause. The bundled OpenCV is a vc16 build and MinGW cannot link
    #     against it -- which is exactly what compiler_missing_advice() says.
    if not have_cxx_compiler():
        compiler_missing_advice()
        sys.exit(1)

    # A CMakeCache.txt left behind by a FAILED configure pins the bad
    # generator choice (e.g. NMake picked when no VS existed yet) and poisons
    # every later attempt. If a cache exists but no built detector does, the
    # cache is stale - wipe it so configuration restarts cleanly.
    build_dir = os.path.join(code_dir, "build")
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.exists(cache) and not find_detector_exe(code_dir):
        print("   Stale CMake cache from a failed configure -> wiping build\\ ...")
        shutil.rmtree(build_dir, ignore_errors=True)
    took("step 3")

    print("\n-> Step 4: Configuring CMake build...")
    # On force l'utilisation des binaires vc16 pour contourner l'incompatibilite VS2026
    cmake_cfg = [cmake_exe, "-B", "build", "-DOpenCV_RUNTIME=vc16"]
    if os.name == "nt":
        # Target x64 explicitly. The Visual Studio generator otherwise builds
        # for the HOST architecture, which is right on an x64 machine and wrong
        # on Windows-on-ARM: there the default becomes ARM64, and the link
        # fails against the bundled OpenCV, which is a vc16 x64 build with no
        # ARM64 counterpart published. x64 is the only architecture this kit
        # has binaries for, so name it rather than inheriting it. Harmless on
        # an x64 host, where it is already the default.
        cmake_cfg += ["-A", "x64"]
    if opencv_dir:
        cmake_cfg.append(f"-DOpenCV_DIR={opencv_dir}")
    run_command(cmake_cfg, cwd=code_dir)
    took("step 4")

    print("\n-> Step 5: Compiling the project (Release Configuration)...")
    run_command([cmake_exe, "--build", "build", "--config", "Release"], cwd=code_dir)
    took("step 5")

    exe = find_detector_exe(code_dir)
    print(f"\n   detector: {exe or '[WARNING] not found after build!'}")

    # 4. Execute the main pipeline script
    if args.no_launch:
        print("\n-> Step 6: SKIPPED (--no-launch). Setup is complete.")
        print("   Start the GUI with:  Code\\venv\\Scripts\\python.exe PipelineController.py")
        return

    print("\n-> Step 6: Launching PipelineController.py...")
    print("   (This terminal stays busy until you close the GUI window - that is")
    print("    normal, it is not frozen.)")
    if os.path.exists(pipeline_script):
        run_command([venv_python, "PipelineController.py"], cwd=base_dir)
    else:
        print(f"\n[ERROR] Could not locate {pipeline_script} at the root directory.")
        sys.exit(1)

    print("\n[SUCCESS] Script executed and pipeline is running!")


if __name__ == "__main__":
    main()
