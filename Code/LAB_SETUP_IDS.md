# IDS peak lab setup for qBOUNCE

## Short answer

Use the lab network if IT allows it, because it gives you the current IDS peak
installer and matching documentation. Also bring the installer on a USB stick as
a fallback, because lab PCs are often locked down or isolated.

Do not rely on macOS for the final camera test. The qBOUNCE pipeline can be
developed on macOS, but IDS peak camera acquisition should be validated on the
lab Windows/Linux machine that will actually talk to the camera.

## What to bring on the USB stick

- This repository.
- The IDS peak SDK installer for the lab OS.
- The IDS peak ReadMe/changelog matching that installer.
- Python wheel/cache folder if the lab PC has no internet access.
- A small known-good dark/signal image sample so the detector can be tested
  without the camera.

## Install on the lab machine

1. Install IDS peak SDK, not only the runtime package.
   The SDK package includes development files and IDS peak Cockpit.

2. If the camera is an older IDS uEye camera with a `UI-` match code, use the
   IDS peak extended setup or install the IDS Software Suite as required by IDS.

3. Install/build the qBOUNCE dependencies:

   ```bash
   python -m pip install -r requirements-lab.txt -c constraints-lab.txt
   ```

   Offline, add `--no-index --find-links wheels`. The IDS bindings are not in
   `requirements-lab.txt` because they come from the vendor rather than PyPI;
   install them from the same folder:

   ```bash
   python -m pip install --no-index --find-links wheels ids_peak ids_peak_ipl
   ```

4. Verify IDS Python bindings:

   ```bash
   python -c "from ids_peak import ids_peak; from ids_peak_ipl import ids_peak_ipl"
   ```

5. Verify camera discovery:

   ```bash
   python src/ids_capture.py --list-devices
   ```

6. Open IDS peak Cockpit and check image quality, exposure, gain, ROI, trigger
   mode, and pixel format.

7. Close Cockpit before running the capture script if Cockpit has exclusive
   control of the camera.

## qBOUNCE capture workflow

Capture dark frames:

```bash
python src/ids_capture.py --output Data/dark --count 100 --clear-output
```

Capture signal frames:

```bash
python src/ids_capture.py --output Data/signal --count 1000 --exposure-us 500000
```

Then run the normal pipeline:

```bash
cmake -S . -B build
cmake --build build
./build/detector --folder Data/dark --csv dark_pass.csv --model background_model.yml --warmup 100
./build/detector --folder Data/signal --csv detections.csv --model background_model.yml --resume
```

The GUI controller has matching buttons in the Parameters tab:

- `List IDS Cameras`
- `Capture Dark Folder`
- `Capture Signal Folder`

## Integration design

The detector still receives ordinary image files. This is intentional:

- offline data copied from SMB still works;
- the ML and labeling tools remain unchanged;
- IDS SDK failures are isolated to `src/ids_capture.py`;
- the lab machine can own the camera acquisition, while any other machine can
  analyze the saved frames later.

`ids_capture.py` requests `Mono8` frames because the current C++ detector expects
8-bit grayscale images. If the experiment later needs 10/12/16-bit raw ADU data,
update the detector first to accept `CV_16UC1`, then change the capture script's
pixel format and output format.
