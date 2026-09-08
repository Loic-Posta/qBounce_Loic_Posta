# qBounce — Autonomous Detection and Characterization of Particles within the CMOS Sensor

Bachelor thesis submission — Loïc Posta, TU Wien.
Begutachter: Privatdoz. Dipl.-Ing. Dr. Johann Marton.
Betreuer: Univ.Ass. Dipl.-Ing. Dr.techn. Joachim Bosina.
Mitwirkung: Univ.Prof. Dipl.-Phys. Dr.rer.nat. Dr.h.c. Hartmut Abele.

The thesis is the PDF at the root of this folder,
`qBounce_Loic_Posta_Autonomous_Detection_and_Characterization_of_Particles_within_the_CMOS_Sensor.pdf`.

---

## What this repository leaves out

This repository is **70 MB**. The submitted archive was 277 MB, and the whole
difference is `Code/wheels/`: 210 MB of third-party Python wheels bundled so the
lab machine at the ILL could install without a network. Redistributing other
people's binaries is not what a repository is for, so they are not here.
Section 4 rebuilds them in one command.

`Code/Data/` (51 MB) is still here: the 50 sample frames that let the detector
run out of the box. Delete it too and the tree is **19 MB** — the thesis, its
LaTeX source, and every line of code. That is enough to read and to review, and
not enough to run.

### Archiving it

If you repackage this folder on macOS, set `COPYFILE_DISABLE=1` first:

```bash
COPYFILE_DISABLE=1 tar czf qBounce_Loic_Posta.tgz qBounce_Loic_Posta
```

Without it, macOS `tar` writes a `._` companion for every file that carries an
extended attribute, and the recipient unpacks twenty-one files of Apple metadata
alongside the code. `zip -X` does the same job.

---

## Four ways in

### 1. You are here to read the thesis and grade it

The PDF at the root, 73 pages. The LaTeX source is in
`docs/BachelorThesis_Loic_Posta/` and rebuilds with:

```bash
cd docs/BachelorThesis_Loic_Posta
latexmk -pdf main.tex
```

It needs a full TeX distribution and `biber`. Keep the directory layout as it
is: the thesis quotes four code excerpts straight out of `Code/src/` by relative
path and inputs three figures from `docs/`, so flattening the tree breaks the
build.

### 2. You are here to run the code

**C++ detector.** macOS, with `libomp` from Homebrew:

```bash
cd Code
cmake -B build \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib"
cmake --build build
```

Windows, with Visual Studio and CMake on the path. **OpenCV for C++ is not in
this folder** and is not a Python package: `pip install opencv-python` does not
help the detector. Install the prebuilt Windows OpenCV from opencv.org first,
then point cmake at the per-toolset directory inside it, not at `build`
itself. The config at the root of `build` derives the runtime from your
compiler and refuses to load as soon as that compiler is newer than any toolset
the pack was built for -- MSVC 2026 against a vc16 pack fails there with "no
binaries compatible with your configuration". The versioned config skips the
check and links without complaint:

```powershell
cd Code
cmake -B build -A x64 -DOpenCV_DIR=C:\opencv\build\x64\vc16\lib
cmake --build build --config Release
```

Use `vc17` in place of `vc16` for the newer toolsets. `WindowsLauncher.py` and
the graphical interface both find that path on their own.

**Run it on the frames in this folder.** Build a background model from the dark
frames, then detect on the beam-on frames against it:

```bash
./build/detector --folder Data/dark/2026-07-11_3-001_background_575-6ms_2fps \
                 --model bg.bgm --csv dark.csv --nsigma 5 --warmup 10

./build/detector --folder Data/signal/2026-07-11_3-002_575-6ms_2fps_first_UCN \
                 --model bg.bgm --resume --csv signal.csv --nsigma 5 --warmup 10
```

On the 20 beam-on frames shipped here that gives 118 217 clusters, of which 714
are ten pixels or larger — those are the neutron tracks.

The twenty frames are not an arbitrary sample. They are the twenty that carry
the most hand-labelled clusters, so `Code/src/labeling_ui.py` opens against the
shipped `detections_annotation_sample.csv` and finds 92 of its 200 labelled
objects in the images beside it:

```bash
python3 Code/src/labeling_ui.py --csv Code/detections_annotation_sample.csv \
    --folder Code/Data/signal/2026-07-11_3-002_575-6ms_2fps_first_UCN
```

The remaining 108 sit in frames that are not shipped; point `--folder` at the
full acquisition to reach them.

The detector is deterministic across compilers and architectures. On the 500
dark frames it returns the same 39 071 clusters on every one of ten repeats,
under clang on ARM (`Code/benchmark_mac.txt`), under MSVC on x86-64
(`Code/benchmark_windows_2026-08-26.txt`) and under GCC on x86-64
(`Code/benchmark_linux_2026-09-02.txt`). The last two are the same laptop booted
into Windows and into Linux, so between them the only variables are the
operating system and the compiler -- and the same work takes 372 ms per frame
under GCC against 561 ms under MSVC. Section 3.6 of the thesis reports the
timings. The
background model it writes is 260 MB, so it is not shipped; the first command
rebuilds it in a few seconds.

**Python side.** The launcher and the interface both look for the environment in
`Code/venv`, so create it there and nowhere else:

```bash
python3 -m venv Code/venv
source Code/venv/bin/activate          # Code\venv\Scripts\activate on Windows
pip install -r Code/requirements-lab.txt -c Code/constraints-lab.txt
```

The `-c` matters. Without it, nine of the twenty-seven packages resolve to newer
versions than the ones behind the results in this thesis, `onnxruntime` and
`ml_dtypes` among them, and a classifier bundle saved by one `scikit-learn` is
not guaranteed to load in another. `PipelineController.py` at the root
is the graphical interface that drives the whole chain.

### 3. You want to reuse this for your own work

Everything in `Code/src/` is parameterised through the command line or through
the interface; nothing is hard-coded to the runs of this thesis. The three
places to start:

- `Code/src/ClusterDetector.cpp` and `BackgroundModel.cpp` — the detection
  itself. The threshold rule, the dead-pixel rule and the Welford accumulation
  are each in one place.
- `Code/src/ml_classifier.py` — training and prediction. The feature list is
  read from the saved bundle, not from a constant, so a model trained on other
  data stays consistent with the code that applies it.
- `Code/model_bundle.joblib` and `.onnx` — the classifier trained for this
  thesis, with `model_bundle.meta.json` recording what it was trained on.

Section 3.2 of the thesis maps the four poles of the codebase, one figure each,
with every arrow explained. `docs/UserGuide.pdf` is the operator's guide.

### 4. You are installing at the ILL, on a machine with no network

`Code/wheels/` is not in this repository. Rebuild it once on any machine with a
network, then carry the folder to the lab:

```powershell
pip download -r Code\requirements-lab.txt -c Code\constraints-lab.txt -d Code\wheels `
  --platform win_amd64 --python-version 3.14 --only-binary=:all:
pip download ids_peak ids_peak_ipl -d Code\wheels --platform win_amd64 --only-binary=:all:
```

That reproduces the 32 wheels the submitted archive shipped: the 11 packages
named in `requirements-lab.txt` and everything they pull in, 27 in all once
installed, built for **Windows x86-64 and Python 3.14**. On the lab machine:

```powershell
python -m venv Code\venv
Code\venv\Scripts\activate
pip install --no-index --find-links Code\wheels -r Code\requirements-lab.txt -c Code\constraints-lab.txt
pip install --no-index --find-links Code\wheels ids_peak ids_peak_ipl
```

`--no-index` is what makes it work without a network: pip never tries to reach
PyPI. `Code/constraints-lab.txt` pins the exact versions these wheels contain.

Three things the wheels cannot cover, and all three need to travel to the ILL by
other means. The **IDS peak SDK** for the camera is not a Python package and
comes from the vendor's own installer; `Code/LAB_SETUP_IDS.md` covers it, and
`PrepareSetupLab.py` checks the machine afterwards. **OpenCV for C++** is a
separate download, as in the previous section. And a **C++ toolchain** must
already be present, or a built `detector.exe` must be carried over instead.

If your machine is not Windows x86-64 on Python 3.14, the wheels do not apply
and you need a network for that step.

---

## What is in here

```
.
├── qBounce_Loic_Posta_..._CMOS_Sensor.pdf   the thesis, 73 pages
├── README.md                              this file
├── PipelineController.py                  the graphical interface
├── LabValidation.py                       end-to-end validation run
├── PrepareSetupLab.py                     lab machine preparation and checks
├── WindowsLauncher.py                     launcher for the Windows lab PC
├── Code/
│   ├── src/                  detector (C++) and analysis/ML (Python)
│   ├── Data/                 30 dark frames and 20 beam-on frames
│   ├── (wheels/)             not in this repository -- see section 4
│   ├── CMakeLists.txt        build definition
│   ├── requirements-lab.txt  Python dependencies
│   ├── constraints-lab.txt   the exact versions
│   ├── LAB_SETUP_IDS.md      IDS camera setup
│   └── model_bundle.*        the trained classifier
└── docs/
    ├── UserGuide.pdf         operator's guide
    └── BachelorThesis_Loic_Posta/   full LaTeX source
```

The 50 frames in `Code/Data/` are a sample. The runs behind the results in the
thesis are about 31 GB and are not here: the dark run of 500 frames, the beam-on
run of 1500, the background-model study, and the intermediate detection CSVs.
The analysis parameters are stated in the Results chapter.

## Verification

Checked on macOS (Darwin 25.5.0) on 26 August 2026, from this folder as it
stands:

| check | result |
|---|---|
| C++ detector configures and builds from a clean tree | pass, 0 warnings |
| detector builds a background model from the shipped dark frames | pass |
| detector finds tracks in the shipped beam-on frames | pass, 118 217 clusters, 714 at 10 px or larger |
| the labelling interface finds its frames | pass, 92 of 200 labelled clusters |
| detector rebuilt on Windows with MSVC (Build Tools 2026, OpenCV 4.10.0) | pass, 0 errors |
| detector rebuilt on Linux with GCC (cmake 3.22, OpenCV 4.5.4) | pass, 0 warnings |
| offline install from `Code/wheels/`, no network | pass, 27 packages, versions match `constraints-lab.txt` (verified on the submitted archive, which shipped the wheels) |
| all Python sources compile | pass, 28 files |
| thesis compiles from this folder | pass, 73 pages, no undefined reference |
