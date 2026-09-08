#!/usr/bin/env python3
"""Regenerate every figure of the thesis' Results section.

Plotting rule applied throughout: two points are joined by a line only when
something was actually measured between them. N is sampled at six values, so
every panel against N shows markers alone; the per-frame series is measured at
every frame and stays a line; a fitted power law is a model, and stays a line.

Kept in the repository, next to the thesis, and writing into
figures_results/ -- an earlier version of these figures lived in a temporary
scratch directory and was lost when it was cleaned, which is exactly the
mistake this file exists to prevent.

Input: the detector outputs produced by running, for N in {10,25,50,100,200,500},
the compiled detector over the first N frames of
    Code/Data_tot/dark/2026-07-11_3-001_background_575-6ms_2fps
and once over the 1500 frames of
    Code/Data_tot/signal/2026-07-11_3-002_575-6ms_2fps_first_UCN
with --resume against the N=500 model. Point --runs at the directory holding
the resulting N*/ and signal1500/ folders.

Colour choices follow the data's job: N is an ORDERED quantity, so the
per-N histogram overlay uses a single-hue sequential ramp; "p99 vs mean" and
"before vs after" are identities and use categorical slots 1-3 of the
reference palette. One y-axis per panel throughout -- never a dual axis.
"""
import argparse
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8b8a85"
SEQ = plt.cm.Blues
NS = [10, 25, 50, 100, 200, 500]

# Every figsize below is the width main.tex asks for, not a larger canvas that
# LaTeX then shrinks. \textwidth is 16 cm = 6.30 in (header.tex l. 113), so a
# figure posted at f\linewidth is drawn f x 6.30 in wide, less the 0.11 in that
# savefig.bbox="tight" adds back. The scale factor on the page is then ~1.03
# instead of the 0.53 to 0.67 it used to be, and the 9 pt set here is 9 pt on
# paper -- the size of the \figsource line, which is the floor. savefig.dpi
# rises with it, so the printed resolution stays above 300 dpi.
# Les blocs explicatifs de plusieurs lignes sont subordonnes aux etiquettes
# d'axe : ils sont poses au plancher, la taille de la ligne \figsource, mesuree
# a 8,97 pt dans le PDF. Les figures etant dessinees a leur largeur d'impression
# le facteur d'echelle mesure va de 1,027 a 1,030 selon la figure, d'ou
# 8,97 / 1,027 : la valeur precedente, 8,72, tenait pour 1,029 et laissait la
# note de la figure 4.9 a 8,95 pt, deux centiemes sous le plancher.
NOTE_PT = 8.75
# Figures 4.2 and 4.9 are drawn at the full print width (6.19 in plus the
# 0.11 in of tight bbox = \linewidth), so their scale factor is 1 and the
# floor is the 8.97 pt of the \figsource line itself. 9 pt clears it.
FULL_W, NOTE_FULL = 6.19, 9.0

mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})


def load(folder: Path, name: str):
    a = cv2.imread(str(folder / name), cv2.IMREAD_UNCHANGED)
    return None if a is None else np.asarray(a, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory holding N10/ ... N500/ and signal1500/")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "figures_results")
    args = ap.parse_args()
    RUNS, OUT = args.runs, args.out
    OUT.mkdir(exist_ok=True)

    rows, hists = [], {}
    EDGES = np.linspace(0, 6, 121)
    for n in NS:
        f = RUNS / f"N{n}"
        mean, sig = load(f, "background_mean.tiff"), load(f, "background_sigma.tiff")
        if mean is None or sig is None:
            print(f"  ! N={n}: maps missing, skipped")
            continue
        hists[n] = np.histogram(sig.ravel(), bins=EDGES)[0]
        csv = f / "dark_pass.csv"
        cpf = (sum(1 for _ in csv.open()) - 1) / n if csv.is_file() else None
        rows.append(dict(N=n, sig_p99=float(np.percentile(sig, 99)),
                         sig_p999=float(np.percentile(sig, 99.9)),
                         sig_mean=float(sig.mean()),
                         mean_p99=float(np.percentile(mean, 99)),
                         mean_mean=float(mean.mean()), cpf=cpf))
        del mean, sig
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "convergence_table.csv", index=False)
    print(df.to_string(index=False))

    # ---- Figure 1: convergence -----------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(FULL_W, 5.55), constrained_layout=True)
    ax = ax.ravel()
    ax[0].plot(df.N, df.sig_p99, "o", color=S1, ms=5, label="99th percentile")
    ax[0].plot(df.N, df.sig_mean, "s", color=S2, ms=5, label="mean over pixels")
    ax[0].axhline(1.0, color=INK2, lw=1.6, ls="--")
    # "the whole distribution" was wrong: what this panel plots is the 99th
    # percentile and the mean, and 99.88% of pixels sit below the floor, not all
    # of them. Figure 4.2 on the next page shows the tail reaching 6 ADU.
    # Same words as before, broken to the width of the panel: at 9 pt the
    # single 36-character line ran across into panel (b).
    ax[0].annotate("1 ADU floor:\n99.9% sit below", xy=(11, 3.8), va="top",
                   fontsize=NOTE_FULL, color=INK2,
                   bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=.85))
    ax[0].set(xscale="log", yscale="log", xlabel="dark frames used, $N$",
              ylabel="per-pixel $\\sigma$  [ADU]", ylim=(None, 4))
    ax[0].set_title("(a)", loc="left")

    ax[1].plot(df.N, df.mean_p99, "o", color=S1, ms=5)
    ax[1].plot(df.N, df.mean_mean, "s", color=S2, ms=5)
    ax[1].set(xscale="log", yscale="log", xlabel="dark frames used, $N$",
              ylabel="per-pixel background mean  [ADU]")
    ax[1].set_title("(b)", loc="left")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside upper center",
               ncol=2, fontsize=NOTE_FULL)

    # Poisson error bars: cpf is a count divided by N, so its uncertainty is
    # sqrt(count)/N. They are small at these counts, which is itself worth
    # showing rather than leaving the reader to assume.
    ax[2].errorbar(df.N, df.cpf, yerr=np.sqrt(df.cpf * df.N) / df.N, fmt="o",
                   color=S3, ms=5, capsize=3, elinewidth=1)
    ax[2].set(xscale="log", yscale="log", xlabel="dark frames used, $N$",
              ylabel="clusters per frame, beam OFF")
    ax[2].set_title("(c)", loc="left")

    pc = RUNS / "N500" / "dark_pass.csv"
    if pc.is_file():
        per = pd.read_csv(pc, usecols=["image_id"]).groupby("image_id").size().sort_index()
        order = np.arange(1, len(per) + 1)
        ax[3].plot(order, per.values, lw=0.8, color=MUTED, alpha=0.7, label="per frame")
        ax[3].plot(order, pd.Series(per.values).rolling(25, center=True).mean(),
                   lw=2.2, color=S3, label="25-frame rolling mean")
        ax[3].set(yscale="log",
                  xlabel="frame index within the\n500-frame dark run",
                  ylabel="clusters in that frame")
        ax[3].set_title("(d)", loc="left")
        ax[3].legend(fontsize=NOTE_FULL, loc="lower left")
    fig.savefig(OUT / "fig1_convergence.png"); plt.close(fig)

    # ---- Figure 2: where the noise lives --------------------------------
    f500 = RUNS / "N500"
    mean5, sig5 = load(f500, "background_mean.tiff"), load(f500, "background_sigma.tiff")
    fig, ax = plt.subplots(1, 2, figsize=(5.52, 2.70), constrained_layout=True)
    ax[0].hist(mean5.ravel(), bins=np.linspace(0, 10, 201), color=S1)
    ax[0].set(yscale="log", xlabel="per-pixel background mean  [ADU]", ylabel="pixels")
    ax[0].set_title("(a)", loc="left")
    ax[0].annotate(f"{(mean5 == 0).mean() * 100:.1f}% of pixels\nread exactly 0",
                   xy=(0.30, 0.82), xycoords="axes fraction", color=INK2)
    ax[1].hist(sig5.ravel(), bins=np.linspace(0, 6, 241), color=S1)
    ax[1].axvline(1.0, color=S2, lw=2, ls="--")
    ax[1].annotate("1 ADU detection floor\n(BackgroundModel.cpp)", xy=(1.08, 0.60),
                   xycoords=("data", "axes fraction"), color=S2, fontsize=9)
    ax[1].annotate(f"{(sig5 < 1.0).mean() * 100:.1f}% of pixels sit below\n"
                   "the floor, so their threshold\n"
                   "is set by it, not by $\\sigma$",
                   xy=(0.06, 0.98), va="top",
                   xycoords="axes fraction", color=INK2, fontsize=9)
    ax[1].set(yscale="log", xlabel="per-pixel $\\sigma$  [ADU]", ylabel="pixels")
    ax[1].set_title("(b)", loc="left")
    fig.savefig(OUT / "fig2_noise_distributions.png"); plt.close(fig)

    # ---- Figure 3: sigma histogram vs N (sequential ramp) ---------------
    fig, ax = plt.subplots(figsize=(4.05, 2.55), constrained_layout=True)
    centres = 0.5 * (EDGES[:-1] + EDGES[1:])
    for i, n in enumerate(sorted(hists)):
        ax.step(centres, hists[n], where="mid", lw=1.8, label=f"$N={n}$",
                color=SEQ(0.30 + 0.68 * i / max(1, len(hists) - 1)))
    ax.axvline(1.0, color=S2, lw=1.6, ls="--")
    ax.set(yscale="log", xlabel="per-pixel $\\sigma$  [ADU]", ylabel="pixels")
    ax.legend(title="dark frames", ncol=2)
    fig.savefig(OUT / "fig3_sigma_vs_N.png"); plt.close(fig)

    # ---- Figure 4: signal ----------------------------------------------
    d = pd.read_csv(RUNS / "signal1500" / "detections.csv")
    nfr = d["image_id"].nunique()
    fig, ax = plt.subplots(1, 3, figsize=(5.88, 2.85), constrained_layout=True)
    ax[0].hist(d["size_px"], color=S1,
               bins=np.logspace(0, np.log10(max(2, d["size_px"].max())), 60))
    ax[0].set(xscale="log", yscale="log", xlabel="cluster size  [pixels]", ylabel="clusters")
    ax[0].set_title("(a)", loc="left")
    ax[1].hist(d["total_charge"], bins=200, color=S1)
    sat = (d["peak_value"] >= 254).mean() * 100
    ax[1].annotate(f"{sat:.1f}% of clusters\n"
                   "saturate (peak $\\geq$ 254):\n"
                   "their charge is\n"
                   "a lower bound", xy=(0.06, 0.98), va="top",
                   xycoords="axes fraction", color=INK2, fontsize=9)
    # One panel in three is 1.96 in wide on the page; the label keeps every
    # word and takes two lines rather than running into its neighbours.
    ax[1].set(yscale="log", xlabel="total charge  [ADU]\n(deposited-energy proxy)",
              ylabel="clusters")
    ax[1].set_title("(b)", loc="left")
    big = d[d["size_px"] >= 5]
    ax[2].scatter(big["size_px"], big["total_charge"], s=4, alpha=0.25,
                  color=S1, edgecolors="none")
    ax[2].set(xscale="log", yscale="log", xlabel="cluster size  [pixels]",
              ylabel="total charge  [ADU]")
    # The x range is 5 to 30 px, under a decade, so the automatic locator
    # labels every minor step and at 9 pt the labels collide. Four named
    # ticks; the minor marks stay, their labels go.
    ax[2].set_xticks([5, 10, 20, 30], ["5", "10", "20", "30"])
    ax[2].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax[2].set_title("(c)", loc="left")
    fig.savefig(OUT / "fig4_signal.png"); plt.close(fig)
    print(f"signal: {len(d)} clusters / {nfr} frames = {len(d)/nfr:.1f} per frame; "
          f"saturated {sat:.2f}%")

    # ---- Figure 5: sigma tail, dark trend vs beam ----------------------
    sig_after = load(RUNS / "signal1500", "background_sigma.tiff")
    k, logc = np.polyfit(np.log(NS), np.log(df.sig_p999.values), 1)
    pred = np.exp(logc) * 2000 ** k
    b999 = float(np.percentile(sig_after, 99.9))
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 3.70), constrained_layout=True,
                           width_ratios=[1.30, 1.0])
    ax[0].plot(df.N, df.sig_p999, "o", color=S1, ms=6, label="dark only, p99.9")
    ax[0].plot(df.N, df.sig_p99, "s", color=S3, ms=6, label="dark only, p99")
    xs = np.array([10, 2000])
    # The legend used to say where the line was fitted from, on three lines
    # inside the panel; the caption says it, so the legend does not have to.
    ax[0].plot(xs, np.exp(logc) * xs ** k, ":", color=MUTED, lw=1.9,
               label=f"dark trend, $\\propto N^{{{k:.3f}}}$")
    ax[0].plot([2000], [b999], "D", color=S2, ms=8, label="beam on, 1500 frames")
    ax[0].annotate(f"observed {b999:.2f}\ntrend predicts {pred:.2f}", xy=(2000, b999),
                   xytext=(120, 1.52), fontsize=NOTE_FULL, color=S2,
                   arrowprops=dict(arrowstyle="->", color=S2, lw=1.2))
    ax[0].set(xscale="log", yscale="log", xlabel="frames accumulated in the model",
              ylabel="per-pixel $\\sigma$ quantile  [ADU]")
    ax[0].set_title("(a)", loc="left")
    ax[0].legend(fontsize=NOTE_FULL, loc="upper center",
                 bbox_to_anchor=(0.5, -0.20), ncol=2, columnspacing=1.2)
    vals = [float(df.sig_p999.iloc[-1]), pred, b999]
    ax[1].bar([0, 1, 2], vals, color=[S1, MUTED, S2], width=0.6)
    ax[1].set_xticks([0, 1, 2], ["dark\n500\nframes", "dark\ntrend\nat 2000",
                                 "measured\nat 2000"])
    # Headroom for the note below, which would otherwise sit on the bars now
    # that it is set at the size of the source line.
    ax[1].set_ylim(0, max(vals) * 1.12)
    ax[1].set_ylabel("$\\sigma$ p99.9  [ADU]")
    ax[1].set_title("(b)", loc="left")
    for xi, v in enumerate(vals):
        ax[1].annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom",
                       fontsize=NOTE_FULL, color=INK2)
    # The label used to read "sub-threshold halo". Section 4.6 measures that it
    # is not one, so the figure says what it shows and no more.
    ax[1].annotate(f"beam excess ×{b999/pred:.2f}\nabove the dark trend",
                   xy=(0.02, 0.99), va="top",
                   xycoords="axes fraction", fontsize=NOTE_FULL, color=INK2)
    fig.savefig(OUT / "fig5_sigma_tail_corrected.png"); plt.close(fig)
    print(f"dark trend N^{k:.3f}; p99.9 predicted {pred:.3f}, measured {b999:.3f} "
          f"(x{b999/pred:.2f})")

    # ---- Figure 6: energy, size-cut ------------------------------------
    # The uncut spectrum is dominated by single-pixel noise clusters, so a fit on
    # it separates noise from signal, not lithium from alpha. Cut first.
    big = d[d.size_px >= 5].copy()
    big.to_csv(RUNS / "signal1500" / "det_ge5px.csv", index=False)
    sat = big.peak_value >= 254
    fig, ax = plt.subplots(1, 2, figsize=(5.88, 2.60), constrained_layout=True)
    bins = np.histogram_bin_edges(big.total_charge, bins=60)
    ax[0].hist([big.total_charge[~sat], big.total_charge[sat]], bins=bins,
               stacked=True, color=[S3, "#c0392b"], edgecolor="none",
               label=[f"unsaturated: {(~sat).sum():,}",
                      f"saturated: {sat.sum():,}"])
    ax[0].set(xlabel="total charge  [ADU]", ylabel="clusters")
    ax[0].set_title("(a)", loc="left")
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.20)   # keep the legend off the peak
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].scatter(big.size_px[~sat], big.total_charge[~sat], s=4, alpha=.25,
                  color=S3, edgecolors="none", label="unsaturated")
    ax[1].scatter(big.size_px[sat], big.total_charge[sat], s=4, alpha=.25,
                  color="#c0392b", edgecolors="none", label="saturated")
    ax[1].set(xlabel="cluster size  [pixels]", ylabel="total charge  [ADU]")
    ax[1].set_title("(b)", loc="left")
    ax[1].legend(frameon=False, fontsize=9, markerscale=3)
    fig.savefig(OUT / "fig6_energy_ge5px.png"); plt.close(fig)
    for lo in (5, 10):
        m = d[d.size_px >= lo]
        print(f"saturated among size >= {lo:2d} px: "
              f"{(m.peak_value >= 254).mean()*100:.2f}%  (n = {len(m):,})")

    # ---- Figure 7: where the particles landed --------------------------
    arrival_map(RUNS, OUT)

    print(f"\nwrote {OUT}")
    return 0


def arrival_map(runs: Path, out: Path, block: int = 48) -> None:
    """Where the particles landed, binned into superpixels.

    The heatmap DataExporter writes is a per-pixel linear map, and at 20.1 Mpixel
    a handful of hot pixels (max 135 hits against a median of 1) set the colour
    scale, so the whole sensor renders as one flat colour. Summing over 48x48
    blocks turns single hits into a density that the eye can actually read, and
    a 99th-percentile clip keeps the remaining outliers from taking the scale
    back over.
    """
    a = cv2.imread(str(runs / "signal1500" / "signal_arrival.tiff"),
                   cv2.IMREAD_UNCHANGED).astype(np.float64)
    h, w = a.shape
    a = a[: h // block * block, : w // block * block]
    dens = a.reshape(a.shape[0] // block, block,
                     a.shape[1] // block, block).sum(axis=(1, 3))
    fig, ax = plt.subplots(figsize=(5.29, 3.50), constrained_layout=True)
    im = ax.imshow(dens, cmap="magma", origin="upper",
                   vmax=np.percentile(dens, 99), interpolation="nearest")
    ax.set_xlabel(f"sensor column  [{block}-pixel blocks]")
    ax.set_ylabel(f"sensor row  [{block}-pixel blocks]")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=f"hits per {block}$\\times${block} block")
    fig.savefig(out / "fig7_arrival_map.png")
    plt.close(fig)
    print(f"arrival: {int(a.sum()):,} hits over {(a > 0).mean() * 100:.2f}% of pixels")


if __name__ == "__main__":
    raise SystemExit(main())
