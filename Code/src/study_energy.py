#!/usr/bin/env python3
"""
study_energy.py — deposited-energy study of detected particle tracks.

Physics context (n + B-10 -> alpha + Li-7, back-to-back, one daughter
reaches the CMOS):
    branch (96%):  Li-7 excited   -> alpha 1.47 MeV | Li-7 0.84 MeV
    branch ( 4%):  Li-7 ground st -> alpha 1.78 MeV | Li-7 1.01 MeV
so a perfect energy axis would show "2.5 broad peaks" (two strong lines
plus two weak satellites a few percent higher).

Energy proxy:
    The detector's `total_charge` column is exactly the supervisor's
    "sum of the excessive white": sum over the cluster's pixels of
    (pixel - background_mean). It is proportional to deposited energy
    ONLY while no pixel clips at 255. Saturated events (peak_value >= 254)
    lose part of the signal — their total_charge is a LOWER BOUND, which
    compresses and merges the peaks. The saturated fraction is therefore
    printed and drawn: it is the first number to look at before believing
    (or disbelieving) any structure in the histogram.

Usage:
    python study_energy.py --csv detections_classified.csv
    python study_energy.py --csv detections_classified.csv --labels lithium alpha
    python study_energy.py --csv batch.csv --label-col label --out energy.png

Outputs one PNG (histogram + charge-vs-size scatter) and a console report
including a 1-vs-2 component Gaussian-mixture comparison (BIC) on
log(total_charge) as an objective "is it bimodal?" hint.
"""

from __future__ import annotations
import sys

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


# Track classes considered neutron-capture daughters. Extend if the
# labeling vocabulary grows (e.g. an explicit "alpha" class).
HEAVY_TRACK_LABELS = ("lithium", "alpha")
SATURATION_LEVEL = 254  # peak_value >= this => at least one clipped pixel


def load_valid(csv_path: Path, label_col: str | None) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    if "filter_status" in df.columns:
        df = df[df["filter_status"].astype(str) == "valid"]
    if label_col is None:
        label_col = "predicted_label" if "predicted_label" in df.columns else "label"
    if label_col not in df.columns:
        raise SystemExit(f"No '{label_col}' column in {csv_path} "
                         f"(columns: {list(df.columns)})")
    return df, label_col


# Known lines of n + B-10 -> alpha + Li-7 (MeV deposited by the daughter
# that reaches the sensor). 96%-branch lines anchor the calibration; the
# 4%-branch satellites are predictions to look for once calibrated.
LI_LINE_MEV, ALPHA_LINE_MEV = 0.84, 1.47          # 96% branch (Li-7 excited)
SATELLITE_LINES_MEV = (1.01, 1.78)                 # 4% branch (ground state)
KINEMATIC_RATIO = ALPHA_LINE_MEV / LI_LINE_MEV     # = 1.75


def fit_two_peaks(charges: np.ndarray):
    """2-component GMM on log(charge); returns (low_mean, high_mean) in ADU
    or None when the fit is impossible/meaningless."""
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return None
    x = np.log(charges[charges > 0]).reshape(-1, 1)
    if len(x) < 20:
        return None
    m = GaussianMixture(2, n_init=5, random_state=0).fit(x)
    means = np.sort(np.exp(m.means_.ravel()))
    return float(means[0]), float(means[1])


def calibration_report(charges: np.ndarray, sat_fraction: float) -> tuple[str, tuple | None]:
    """Anchor the two fitted peaks on the known 0.84 / 1.47 MeV lines.

    Returns (text, (a, b)) where energy_MeV = a * charge_ADU + b, or
    (text, None) if calibration could not be established."""
    peaks = fit_two_peaks(charges)
    if peaks is None:
        return "Calibration: could not fit two peaks (too few events?).", None
    lo, hi = peaks
    # Two points -> exact affine map. Also report the proportional (through-
    # zero) gain as a physics sanity check: if the response were linear with
    # no offset, both peaks would give the same ADU/MeV.
    a = (ALPHA_LINE_MEV - LI_LINE_MEV) / (hi - lo)
    b = LI_LINE_MEV - a * lo
    ratio = hi / lo
    lines = [
        f"Calibration (2-point affine): E[MeV] = {a:.5f} * charge + {b:+.4f}",
        f"  anchors: {lo:.0f} ADU -> {LI_LINE_MEV} MeV (Li)   "
        f"{hi:.0f} ADU -> {ALPHA_LINE_MEV} MeV (alpha)",
        f"  gain check: {lo / LI_LINE_MEV:.0f} vs {hi / ALPHA_LINE_MEV:.0f} ADU/MeV "
        f"(should agree if response is proportional)",
        f"  peak ratio = {ratio:.2f} vs kinematic {KINEMATIC_RATIO:.2f}",
        "  4%-branch satellites expected at: "
        + "  ".join(f"{(mev - b) / a:.0f} ADU ({mev} MeV)"
                    for mev in SATELLITE_LINES_MEV),
    ]
    if sat_fraction > 0.10:
        lines.append(
            f"  ⚠ {100 * sat_fraction:.0f}% of events are saturated — the "
            "charge scale is compressed and this calibration is biased. "
            "Reduce exposure/gain (acquisition mode 'energy') and re-run.")
    return "\n".join(lines), (a, b)


def separability_report(df: "pd.DataFrame", label_col: str,
                        signal_labels=HEAVY_TRACK_LABELS) -> tuple[str, str]:
    """Can the signal (heavy tracks) still be told apart from the noise
    (everything else)? Returns (verdict, text) with verdict in
    {"SEPARABLE","MARGINAL","MERGED"}.

    This is the safety net for lowering the gain: as tracks dim, their faint
    edge pixels fall below the detection threshold, the clusters shrink and
    lose charge, and eventually the signal population slides into the noise
    population — at which point NO filter and NO classifier can separate
    them, and the neutron count becomes meaningless. We test it in features
    that are meant to be gain-robust: size_px (physical track extent) and
    peak_snr (signal/noise in σ units, gain cancels). For each, we compare
    the low edge of the signal (5th percentile) to the high edge of the
    noise (95th percentile): a positive gap means a clean valley exists."""
    sig = df[df[label_col].isin(signal_labels)]
    noise = df[~df[label_col].isin(signal_labels)]
    if len(sig) < 10 or len(noise) < 10:
        return "MARGINAL", (f"Separability: too few events "
                            f"(signal={len(sig)}, noise={len(noise)}) to judge.")

    lines = ["Signal-vs-noise separability (gain-robust features):"]
    worst = "SEPARABLE"
    for feat in ("size_px", "peak_snr"):
        if feat not in df.columns:
            continue
        s = sig[feat].to_numpy(dtype=float)
        nz = noise[feat].to_numpy(dtype=float)
        sig_lo = np.percentile(s, 5)      # faintest signal
        noise_hi = np.percentile(nz, 95)  # brightest noise
        # Overlap fraction: signal events buried inside the noise bulk.
        buried = float(np.mean(s <= noise_hi))
        gap = sig_lo - noise_hi
        if buried >= 0.20:
            verdict = "MERGED"
        elif buried >= 0.05:
            verdict = "MARGINAL"
        else:
            verdict = "SEPARABLE"
        order = {"SEPARABLE": 0, "MARGINAL": 1, "MERGED": 2}
        if order[verdict] > order[worst]:
            worst = verdict
        lines.append(
            f"  {feat:10s}: signal p5={sig_lo:7.1f}  noise p95={noise_hi:7.1f}  "
            f"gap={gap:+7.1f}  ({100*buried:.0f}% of signal buried in noise) -> {verdict}")

    if worst == "MERGED":
        lines.append("  ⚠ MERGED: signal and noise overlap — the neutron count "
                     "cannot be trusted at this operating point. Raise the gain "
                     "back up, or accept you can no longer separate the classes.")
    elif worst == "MARGINAL":
        lines.append("  ⚠ MARGINAL: the valley between signal and noise is "
                     "closing — watch this closely if you lower the gain further.")
    return worst, "\n".join(lines)


def gmm_bimodality_report(charges: np.ndarray) -> str:
    """Fit 1- vs 2-component Gaussian mixtures on log(charge) and compare BIC.

    log-space because charge peaks are expected to be roughly proportional
    in width to their position (broad MeV lines on an ADU axis)."""
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return "scikit-learn unavailable — skipping mixture test."
    x = np.log(charges[charges > 0]).reshape(-1, 1)
    if len(x) < 20:
        return f"only {len(x)} events — too few for a meaningful mixture fit."
    fits = {k: GaussianMixture(k, n_init=5, random_state=0).fit(x) for k in (1, 2)}
    bic = {k: m.bic(x) for k, m in fits.items()}
    lines = [f"GMM BIC: 1 component = {bic[1]:.1f}   2 components = {bic[2]:.1f}"
             f"   ({'2 peaks favoured' if bic[2] < bic[1] else '1 peak favoured'})"]
    m2 = fits[2]
    means = np.exp(m2.means_.ravel())
    order = np.argsort(means)
    for i in order:
        lines.append(
            f"  component: mean charge ~ {means[i]:8.1f} ADU   "
            f"weight = {m2.weights_[i] * 100:4.1f}%")
    if bic[2] < bic[1]:
        lo, hi = means[order[0]], means[order[-1]]
        lines.append(f"  peak ratio hi/lo = {hi / lo:.2f} "
                     f"(alpha/Li kinematics predicts ~1.75 if unsaturated)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--csv", type=Path, required=True,
                    help="detections CSV (must carry total_charge/peak_value)")
    ap.add_argument("--labels", nargs="*", default=list(HEAVY_TRACK_LABELS),
                    help="classes to include (default: heavy-track classes)")
    ap.add_argument("--label-col", default=None,
                    help="label column (default: predicted_label, else label)")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--calibrate", action="store_true",
                    help="anchor the two fitted peaks on the known 0.84/1.47 "
                         "MeV lines and add a MeV axis to the histogram")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG (default: <csv_dir>/energy_histogram.png)")
    ap.add_argument("--check-separability", action="store_true",
                    help="only test whether signal and noise are still "
                         "separable; exit 3 if MERGED, 0 otherwise")
    args = ap.parse_args()

    df, label_col = load_valid(args.csv, args.label_col)

    # Separability gate: run first so a MERGED verdict is loud even when the
    # histogram itself would still draw something plausible-looking.
    verdict, sep_text = separability_report(df, label_col, tuple(args.labels))
    print(sep_text)
    if args.check_separability:
        raise SystemExit(3 if verdict == "MERGED" else 0)

    sel = df[df[label_col].isin(args.labels)].copy()
    if sel.empty:
        raise SystemExit(
            f"No cluster with {label_col} in {args.labels} "
            f"(available: {sorted(df[label_col].unique())})")

    sel["saturated"] = sel["peak_value"] >= SATURATION_LEVEL
    n = len(sel)
    n_sat = int(sel["saturated"].sum())
    charges = sel["total_charge"].to_numpy(dtype=float)

    print(f"Events: {n}  ({', '.join(args.labels)})")
    print(f"Saturated (peak_value >= {SATURATION_LEVEL}): "
          f"{n_sat} ({100.0 * n_sat / n:.1f}%)"
          + ("   <- energy scale is COMPRESSED; consider lowering "
             "exposure/gain until this drops" if n_sat else ""))
    q = np.percentile(charges, [10, 50, 90])
    print(f"total_charge quantiles: p10={q[0]:.0f}  median={q[1]:.0f}  p90={q[2]:.0f}")
    print(gmm_bimodality_report(charges))

    calib = None
    if args.calibrate:
        text, calib = calibration_report(charges, n_sat / n)
        print(text)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bins = np.histogram_bin_edges(charges, bins=args.bins)
    for saturated, part in sel.groupby("saturated"):
        ax1.hist(part["total_charge"], bins=bins, alpha=0.65,
                 label=f"{'saturated' if saturated else 'unsaturated'} "
                       f"(n={len(part)})")
    ax1.set_xlabel("total_charge  [ADU, sum of pixel − background]")
    ax1.set_ylabel("events")
    ax1.set_title(f"Deposited-charge histogram — {', '.join(args.labels)}")
    ax1.legend()

    if calib is not None:
        a, b = calib
        mev_axis = ax1.secondary_xaxis(
            "top", functions=(lambda adu: a * adu + b,
                              lambda mev: (mev - b) / a))
        mev_axis.set_xlabel("deposited energy  [MeV]  (2-point calibration)")
        for mev, style in ((LI_LINE_MEV, "-"), (ALPHA_LINE_MEV, "-"),
                           (SATELLITE_LINES_MEV[0], "--"), (SATELLITE_LINES_MEV[1], "--")):
            ax1.axvline((mev - b) / a, color="#555", linestyle=style,
                        linewidth=0.9, alpha=0.7)

    ax2.scatter(sel["size_px"], sel["total_charge"],
                c=sel["saturated"].map({True: "#c62828", False: "#2e7d32"}),
                s=12, alpha=0.6)
    ax2.set_xlabel("size_px  [pixels in cluster]")
    ax2.set_ylabel("total_charge  [ADU]")
    ax2.set_title("charge vs track size (red = saturated)")

    fig.tight_layout()
    out = args.out or args.csv.parent / "energy_histogram.png"
    fig.savefig(out, dpi=130)
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    main()
