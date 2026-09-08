"""
study_nsigma.py  –  N_sigma Sensitivity Study
===============================================
Runs the C++ detector for a range of N_sigma values, reads each resulting
detections.csv, and produces two publication-quality figures:

  Figure 1 — Cluster Size Distribution (log-scale histogram)
             One curve per N_sigma.  Shows where the noise floor ends and
             the particle signal begins.

  Figure 2 — Summary statistics vs N_sigma
             Sub-plots: total cluster count, mean cluster size,
             fraction of single-pixel clusters.

Usage
-----
    python study_nsigma.py \\
        --detector ./build/detector \\
        --folder   ./data \\
        --nsigma   3 4 5 6 7 8 10 12 14 16 \\
        --warmup   100 \\
        --outdir   ./study_results

All arguments have sensible defaults (see --help).
The script is safe to re-run: if a run's CSV already exists it is reused
unless --force is passed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless; change to "TkAgg" for interactive

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

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


# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DETECTOR = Path("./build/detector")
DEFAULT_FOLDER   = Path("./data")
DEFAULT_OUTDIR   = Path("./study_results")
# 10 values: dense at low N_sigma (noise-dominated region) and sparser at high
DEFAULT_NSIGMA   = [3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
DEFAULT_WARMUP   = 100

# Histogram bin edges for cluster size (px)
SIZE_BINS = list(range(1, 52))   # 1 … 50, one bin per integer px count


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def run_detector(detector: Path, folder: Path, csv_out: Path,
                 nsigma: float, warmup: int, timeout: int = 1800) -> None:
    """
    Invoke the C++ detector binary for one N_sigma value.
    Raises subprocess.CalledProcessError on non-zero exit.
    """
    cmd = [
        str(detector),
        "--folder", str(folder),
        "--csv",    str(csv_out),
        "--nsigma", str(nsigma),
        "--warmup", str(warmup),
    ]
    print(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [STDERR]\n{result.stderr}", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    print(f"  → done in {elapsed:.1f}s  |  {csv_out.name}")


def load_csv_safe(csv_path: Path) -> pd.DataFrame | None:
    """Load a detections CSV; return None and warn if missing or malformed."""
    if not csv_path.is_file():
        print(f"  [WARN] CSV not found: {csv_path}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(csv_path)
        # Coerce numeric columns – guards against any column-shift remnants
        for col in ["size_px", "peak_snr", "peak_value", "total_charge"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as exc:
        print(f"  [WARN] Could not read {csv_path}: {exc}", file=sys.stderr)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_size_distribution(results: dict[float, pd.DataFrame],
                           outdir: Path) -> None:
    """
    Figure 1: Cluster size distribution histogram.

    Each N_sigma value becomes one line on a log-scale plot.
    Physically: noise dominates the left (size 1-2 px); real particles
    produce a bump at larger sizes.  As N_sigma increases the noise floor
    drops, leaving the particle signal cleaner.
    """
    # Only these four N_sigma values are plotted to avoid the spaghetti effect.
    FIG1_NSIGMA_SUBSET = {4, 8, 12, 16}

    fig, ax = plt.subplots(figsize=(10, 6))

    colours = cm.viridis(np.linspace(0.1, 0.85, len(FIG1_NSIGMA_SUBSET)))
    colour_iter = iter(colours)

    for nsigma, df in sorted(results.items()):
        if nsigma not in FIG1_NSIGMA_SUBSET:
            continue
        if df is None or "size_px" not in df.columns:
            continue

        # Exclude size == 1: compresses the Y-axis, hides particle signal
        sizes = df["size_px"].dropna().astype(int)
        sizes = sizes[sizes >= 2]

        # Bins: strictly 2 to 50, linear integer steps
        bins = list(range(2, 52))
        counts, edges = np.histogram(sizes, bins=bins)

        bin_centres = 0.5 * (np.array(edges[:-1]) + np.array(edges[1:]))
        counts_plot = counts.astype(float)
        counts_plot[counts_plot == 0] = np.nan

        colour = next(colour_iter)
        ax.step(bin_centres, counts_plot, where="mid",
                color=colour, linewidth=2.0, alpha=0.82,
                label=f"N_sigma = {nsigma:.0f}  (n>=2px = {len(sizes):,})")

    ax.set_yscale("log")
    ax.set_xlabel("Cluster size  (px)", fontsize=13)
    ax.set_ylabel("Number of clusters  [log scale]", fontsize=13)
    ax.set_title("Cluster Size Distribution vs N_sigma  (size >= 2 px, linear X)", fontsize=13)

    ax.set_xlim(1.5, 20.5)
    ax.set_xticks([2, 5, 10, 15, 20])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

    ax.grid(True, which="major", alpha=0.35, linestyle="--")
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = outdir / "fig1_size_distribution.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[Figure 1] saved -> {out}")


def plot_summary_statistics(results: dict[float, pd.DataFrame],
                            outdir: Path) -> None:
    """
    Figure 2: Summary statistics vs N_sigma (3 sub-plots).
      a) Total cluster count
      b) Fraction of single-pixel clusters  (noise proxy)
      c) Mean cluster size among size >= 2 px clusters  (signal proxy)
    """
    nsigmas  = sorted(k for k, v in results.items() if v is not None)
    n_total  = []
    frac_1px = []
    mean_sz  = []

    for ns in nsigmas:
        df = results[ns]
        if df is None or "size_px" not in df.columns:
            n_total.append(np.nan)
            frac_1px.append(np.nan)
            mean_sz.append(np.nan)
            continue

        sizes = df["size_px"].dropna().astype(int)
        n     = len(sizes)
        n1    = int((sizes == 1).sum())
        multi = sizes[sizes >= 2]

        n_total.append(n)
        frac_1px.append(n1 / n * 100 if n > 0 else np.nan)
        mean_sz.append(float(multi.mean()) if len(multi) > 0 else np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle("Detector Sensitivity vs N_sigma Threshold", fontsize=14)

    kw = dict(marker="o", linewidth=2, markersize=6, color=T_PLOT["accent"])

    # a) Total count
    axes[0].plot(nsigmas, n_total, **kw)
    axes[0].set_ylabel("Total clusters", fontsize=11)
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("(a) Total cluster count  [log scale]", fontsize=10)

    # b) Noise fraction
    axes[1].plot(nsigmas, frac_1px, marker="o", linewidth=2,
                 markersize=6, color=T_PLOT["danger"])
    axes[1].set_ylabel("Single-pixel fraction (%)", fontsize=11)
    axes[1].set_ylim(0, 105)
    axes[1].axhline(50, linestyle="--", color="grey", alpha=0.5, linewidth=1)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("(b) Fraction of 1-px clusters  (noise proxy)", fontsize=10)

    # c) Mean cluster size (signal proxy)
    # Y-axis forced to start at 0: auto-scale (e.g. 7.2-8.0) looks volatile
    # but the signal is actually very stable across N_sigma values.
    axes[2].plot(nsigmas, mean_sz, marker="o", linewidth=2,
                 markersize=6, color=T_PLOT["ok"])
    axes[2].set_ylabel("Mean size, size≥2 px", fontsize=11)
    axes[2].set_xlabel("N_sigma threshold", fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("(c) Mean cluster size for size ≥ 2 px  (signal proxy)", fontsize=10)
    _valid = [v for v in mean_sz if not np.isnan(v)]
    _ymax  = max(10.0, max(_valid) * 1.25) if _valid else 10.0
    axes[2].set_ylim(0, _ymax)

    # Mark x-axis ticks at every tested value
    for ax in axes:
        ax.set_xticks(nsigmas)

    fig.tight_layout()
    out = outdir / "fig2_summary_statistics.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[Figure 2] saved → {out}")


# Minimal colour palette for plots (no Tkinter dependency)
T_PLOT = {
    "accent":  "#0071e3",
    "ok":      "#1a7f37",
    "danger":  "#b91c1c",
}


def save_summary_csv(results: dict[float, pd.DataFrame], outdir: Path) -> None:
    """Write a compact per-N_sigma summary table to CSV."""
    rows = []
    for nsigma in sorted(results):
        df = results[nsigma]
        if df is None or "size_px" not in df.columns:
            continue
        sz = df["size_px"].dropna().astype(int)
        snr = df["peak_snr"].dropna() if "peak_snr" in df.columns else pd.Series(dtype=float)
        rows.append({
            "nsigma":          nsigma,
            "total_clusters":  len(sz),
            "n_1px":           int((sz == 1).sum()),
            "n_2px_plus":      int((sz >= 2).sum()),
            "frac_1px_pct":    round((sz == 1).mean() * 100, 1),
            "mean_size_px":    round(float(sz.mean()), 2),
            "median_size_px":  round(float(sz.median()), 2),
            "mean_snr":        round(float(snr.mean()), 2) if len(snr) else np.nan,
            "median_snr":      round(float(snr.median()), 2) if len(snr) else np.nan,
        })
    out = outdir / "summary_table.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[Summary ] saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="N_sigma sensitivity study for the particle detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR,
                        help="Path to the compiled C++ detector binary")
    parser.add_argument("--folder",   type=Path, default=DEFAULT_FOLDER,
                        help="Image input folder passed to the detector")
    parser.add_argument("--nsigma",   type=float, nargs="+", default=DEFAULT_NSIGMA,
                        help="List of N_sigma values to test")
    parser.add_argument("--warmup",   type=int, default=DEFAULT_WARMUP,
                        help="--warmup value passed to the detector")
    parser.add_argument("--outdir",   type=Path, default=DEFAULT_OUTDIR,
                        help="Directory for CSV outputs and figures")
    parser.add_argument("--force",    action="store_true",
                        help="Re-run the detector even if a CSV already exists")
    parser.add_argument("--no-run",   action="store_true",
                        help="Skip detector runs; only (re-)plot existing CSVs")
    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────
    if not args.no_run and not args.detector.is_file():
        print(f"[ERROR] Detector binary not found: {args.detector}", file=sys.stderr)
        print("  Build the C++ project first, or pass --no-run to plot existing data.",
              file=sys.stderr)
        sys.exit(1)
    if not args.folder.is_dir():
        print(f"[ERROR] Image folder not found: {args.folder}", file=sys.stderr)
        sys.exit(1)

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory : {args.outdir}")
    print(f"N_sigma values   : {args.nsigma}")
    print()

    # ── Run detector for each N_sigma ─────────────────────────────────────
    results: dict[float, pd.DataFrame | None] = {}

    for ns in args.nsigma:
        csv_out = args.outdir / f"detections_nsigma{ns:.0f}.csv"
        print(f"── N_sigma = {ns:.0f} {'─' * 40}")

        if not args.no_run and (args.force or not csv_out.is_file()):
            try:
                run_detector(args.detector, args.folder, csv_out,
                             nsigma=ns, warmup=args.warmup)
            except Exception as exc:
                print(f"  [ERROR] Detector failed: {exc}", file=sys.stderr)
                results[ns] = None
                continue
        elif csv_out.is_file():
            print(f"  Reusing existing: {csv_out.name}")
        else:
            print(f"  [WARN] No CSV found and --no-run set; skipping ns={ns}")
            results[ns] = None
            continue

        df = load_csv_safe(csv_out)
        if df is not None:
            sz  = df["size_px"].dropna() if "size_px" in df.columns else pd.Series()
            snr = df["peak_snr"].dropna() if "peak_snr" in df.columns else pd.Series()
            print(f"  Clusters: {len(df):,}  |  "
                  f"1-px: {(sz==1).sum():,}  ({(sz==1).mean()*100:.1f}%)  |  "
                  f"median SNR: {snr.median():.2f}")
        results[ns] = df

    # ── Plot ──────────────────────────────────────────────────────────────
    print()
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        print("[ERROR] No valid results to plot.", file=sys.stderr)
        sys.exit(1)

    plot_size_distribution(valid, args.outdir)
    plot_summary_statistics(valid, args.outdir)
    save_summary_csv(valid, args.outdir)

    print()
    print("Study complete.  Check", args.outdir)


if __name__ == "__main__":
    main()
