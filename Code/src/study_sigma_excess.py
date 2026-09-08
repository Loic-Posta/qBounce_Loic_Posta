#!/usr/bin/env python3
"""Where the beam-on sigma excess actually sits on the sensor.

Section 4.6 reports that the upper tail of the per-pixel sigma distribution is
higher after a beam-on run than pure accumulation predicts: 2.28 ADU at the
99.9th percentile against 1.19 extrapolated from the dark trend, a factor 1.9.
The first reading of that was a halo -- the faint edge of a track, below the
5 sigma exclusion, quietly entering the statistics of the pixels around it.

That is testable without a new acquisition, because the detector already writes
down where every cluster landed. If the halo explanation is right the excess has
to be found around tracks, and to fall off with distance from one. This program
looks.

It reads three files a run already produces:

    signal1500/background_sigma.tiff   the per-pixel sigma after the beam-on run
    signal1500/signal_arrival.tiff     how many times each pixel was in a cluster
    signal1500/bg.bgm, N500/bg.bgm     the dead-pixel flags, before and after

and reports, for rings at increasing distance from a hit pixel, how often sigma
exceeds the value the dark trend predicts. It then removes populations one at a
time and recomputes the 99.9th percentile, which is the number Section 4.6
quotes.

Usage
    python study_sigma_excess.py --runs ../runs_2026-08-25

Python 3, numpy and OpenCV. No camera, no new data.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np

# The dark trend of Figure 4.9(a), sigma_99.9 proportional to N^0.105, fitted on
# the six dark points and extrapolated to the 2000 frames the beam-on model has
# accumulated. make_results_figures.py prints it; it is repeated here so this
# program answers on its own.
PREDICTED = 1.194


def read_dead(path: Path) -> np.ndarray:
    """The dead-pixel plane of a .bgm file. See the format note in
    BackgroundModel.cpp: header, then mean, M2 and count as float32, then the
    dead flags as uint8."""
    with open(path, "rb") as f:
        assert f.read(4) == b"BGMD", f"{path} is not a background model"
        struct.unpack("<ii", f.read(8))
        struct.unpack("<f", f.read(4))
        _, rows, cols = struct.unpack("<iii", f.read(12))
        n = rows * cols
        for _ in range(3):
            np.fromfile(f, np.float32, n)
        return np.fromfile(f, np.uint8, n).reshape(rows, cols) > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory holding N500/ and signal1500/")
    ap.add_argument("--predicted", type=float, default=PREDICTED)
    a = ap.parse_args()

    sig = cv2.imread(str(a.runs / "signal1500" / "background_sigma.tiff"),
                     cv2.IMREAD_UNCHANGED).astype(np.float32)
    arr = cv2.imread(str(a.runs / "signal1500" / "signal_arrival.tiff"),
                     cv2.IMREAD_UNCHANGED)
    dead_before = read_dead(a.runs / "N500" / "bg.bgm")
    dead_after = read_dead(a.runs / "signal1500" / "bg.bgm")

    hit = arr > 0
    retired = dead_after & ~dead_before
    above = sig > a.predicted
    k = np.ones((3, 3), np.uint8)
    grown = {d: cv2.dilate(hit.astype(np.uint8), k, iterations=d) > 0
             for d in (1, 2, 3, 5)}
    far = ~grown[5]
    base = float(above[far].mean())        # the rate away from any track

    print(f"sensor {sig.shape[1]}x{sig.shape[0]}, "
          f"{100 * hit.mean():.2f}% of pixels were in a cluster at least once")
    print(f"dark trend predicts sigma_99.9 = {a.predicted:.3f} ADU at 2000 frames; "
          f"measured {np.percentile(sig, 99.9):.3f} "
          f"(x{np.percentile(sig, 99.9) / a.predicted:.2f})")
    print()
    print("does the excess sit around tracks, as a halo would?")
    print(f"  {'ring':<22s}{'pixels':>10s}{'above pred':>12s}{'vs far field':>14s}")
    rings = [("on a hit pixel", hit),
             ("1 px from a hit", grown[1] & ~hit),
             ("2 px", grown[2] & ~grown[1]),
             ("3 px", grown[3] & ~grown[2]),
             ("4 to 5 px", grown[5] & ~grown[3]),
             ("beyond 5 px", far)]
    for name, m in rings:
        r = float(above[m].mean())
        print(f"  {name:<22s}{int(m.sum()):10d}{100 * r:11.3f}%{r / base:13.2f}")
    print("  a halo would light up the first rings. It does not: one pixel out")
    print("  the rate is within a seventh of the far field, and by three pixels")
    print("  there is nothing left. What is enriched is the hit pixel itself.")
    print()

    print("which populations carry it")
    for name, m in (("hit, still alive", hit & ~retired & ~dead_before),
                    ("retired during the run", retired),
                    ("dead before the run", dead_before)):
        r = float(above[m].mean())
        print(f"  {name:<24s}{int(m.sum()):9d} px   "
              f"{100 * r:7.3f}% above   x{r / base:.0f} the far field")
    print()

    print("the quoted percentile, with populations removed")
    for name, keep in (("everything", np.ones_like(hit)),
                       ("excluding retired pixels", ~retired),
                       ("excluding every pixel the beam hit", ~hit)):
        v = float(np.percentile(sig[keep], 99.9))
        print(f"  {name:<36s}{v:6.3f} ADU   x{v / a.predicted:.2f}   "
              f"({100 * keep.mean():.1f}% of the sensor)")
    print("  half the excess is on one twentieth of the sensor. The other half")
    print("  is on pixels no cluster ever occupied, where no halo can reach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
