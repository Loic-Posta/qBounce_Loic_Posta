#!/usr/bin/env python3
"""
make_airy_zeros.py - the Airy function and the zeros that quantise the spectrum.

Replaces a slide export (images_from_presentation/s04_grav_states_6.png) whose
axis labels measured 5.0 pt on the page, against the 8.97 pt floor set by the
\\figsource line, and which carried a floating "Ai(x)" legend entry that only
repeated its own y-axis label.

Nothing here is traced. The zeros come from scipy.special.ai_zeros and the
turning points z_n = |a_n| z_0 are computed from the same constants as
make_grav_states.py, so Figure "grav_states_zeros" and Figure "grav_states"
quote the same five numbers: 13.7, 24.0, 32.4, 39.8, 46.6 um.

    python3 make_airy_zeros.py --out figures_results

Plotting rule of the thesis: two points are joined only when something exists
between them. Ai(x) is a continuous function sampled densely, so it is a line;
the five zeros are discrete and are drawn as markers, never joined.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import airy, ai_zeros

# ---------------------------------------------------------------- house style
# Palette and rcParams copied from make_results_figures.py so that this figure
# does not read as coming from a different hand.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8b8a85"

mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})

# main.tex posts this figure at .86\linewidth of a 16 cm text block, so the
# printed width is 0.86 x 6.30 in = 5.42 in and the canvas is drawn that wide
# less the 0.11 in that savefig.bbox="tight" adds back. LaTeX then neither
# enlarges nor shrinks it (measured scale 1.00), so the 9 pt set above is 9 pt
# on paper and the PNG lands at 300 dpi. audit() re-checks both at every run.
LINEWIDTH_FRACTION = 0.86
TEXTWIDTH_IN = 6.30
FIG_W = LINEWIDTH_FRACTION * TEXTWIDTH_IN - 0.11      # 5.31 in
FIG_H = 3.55
FLOOR_PT = 8.97                                        # the \figsource size

# ------------------------------------------------------------------ constants
# CODATA 2022, the same values make_grav_states.py uses.
HBAR = 1.054_571_817e-34      # J s
M_N = 1.674_927_500e-27       # kg
G = 9.806_65                  # m s^-2
PEV = 1.602_176_634e-31       # J per peV

Z0 = (HBAR ** 2 / (2 * M_N ** 2 * G)) ** (1 / 3)       # m
E0 = M_N * G * Z0 / PEV                                # peV


def build(n_states: int):
    """Draw the figure and return (fig, ax, zeros, energies, turning points)."""
    zeros = -ai_zeros(n_states)[0]          # |a_n|, ascending: 2.338, 4.088, ...
    energies = zeros * E0                   # peV
    z_turn = zeros * Z0 * 1e6               # micrometres

    x_left, x_right = -9.4, 3.6
    x = np.linspace(x_left, x_right, 3000)
    ai = airy(x)[0]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)

    y_lo, y_hi = -0.83, 0.70
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_lo, y_hi)

    # x > 0 is above the classical turning point: the region the neutron is not
    # allowed into, and where Ai decays instead of oscillating.
    ax.axvspan(0, x_right, color=MUTED, alpha=.10, lw=0, zorder=0)
    ax.axvline(0, color=INK2, lw=.9, ls=(0, (5, 3)), zorder=2)
    ax.axhline(0, color=MUTED, lw=.8, zorder=1)

    ax.plot(x, ai, color=S1, lw=1.9, solid_capstyle="round", zorder=4)

    # the five zeros: discrete points, markers only, never joined
    ax.plot(-zeros, np.zeros_like(zeros), "o", ms=5.4, color=S2,
            mec="white", mew=1.0, zorder=6)

    # Each zero with the state it fixes and the turning point that follows.
    # Two staggered rows because the zeros crowd together towards the left.
    rows = (-0.48, -0.70)
    for i, a_n in enumerate(zeros):
        y_text = rows[i % 2]
        ax.plot([-a_n, -a_n], [-0.035, y_text + 0.135], color=S2, lw=.7,
                ls=(0, (2, 2)), alpha=.85, zorder=3)
        ax.annotate(f"$n={i+1}$\n${z_turn[i]:.1f}\\ \\mu$m",
                    xy=(-a_n, y_text), ha="center", va="center",
                    fontsize=9, color=INK, linespacing=1.35, zorder=7)

    # what the second line of every label is, said once
    ax.annotate("each zero gives one state:   $z_n = |a_n|\\,z_0$",
                xy=(-9.25, 0.67),
                ha="left", va="top", fontsize=9, color=INK2, zorder=7)
    ax.annotate("classically forbidden,\n$E < V$", xy=(2.55, 0.65),
                ha="center", va="top", fontsize=9, color=INK2,
                linespacing=1.3, zorder=7)
    ax.annotate("classical turning\npoint, $z = z_n$", xy=(0.22, -0.14),
                ha="left", va="top", fontsize=9, color=INK2,
                linespacing=1.3, zorder=7)

    ax.set_xlabel(r"$x = z/z_0 - E_n/E_0$   (dimensionless)")
    ax.set_ylabel(r"$\mathrm{Ai}(x)$   (dimensionless)")
    ax.set_xticks(np.arange(-9, 4, 1))
    ax.set_yticks(np.arange(-0.4, 0.61, 0.2))
    return fig, ax, zeros, energies, z_turn


def audit(fig, png: Path) -> None:
    """Print the printed size of every text element, in points on paper."""
    import struct
    with png.open("rb") as fh:
        head = fh.read(24)
    px_w = struct.unpack(">I", head[16:20])[0]
    saved_w_in = px_w / mpl.rcParams["savefig.dpi"]
    printed_w_in = LINEWIDTH_FRACTION * TEXTWIDTH_IN
    scale = printed_w_in / saved_w_in

    sizes = {}
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text().strip()
        if s:
            sizes.setdefault(round(t.get_size() * scale, 2), []).append(s)
    print(f"  PNG {px_w} px wide at {mpl.rcParams['savefig.dpi']} dpi "
          f"= {saved_w_in:.2f} in drawn, {printed_w_in:.2f} in printed "
          f"(scale {scale:.3f}, {px_w / printed_w_in:.0f} dpi on paper)")
    for pt in sorted(sizes):
        flag = "OK " if pt >= FLOOR_PT else "!! "
        print(f"  {flag}{pt:5.2f} pt  x{len(sizes[pt]):3d}  "
              f"e.g. {sizes[pt][0][:38]!r}")
    worst = min(sizes)
    print(f"  smallest text on paper: {worst:.2f} pt "
          f"(floor {FLOOR_PT} pt) -> {'PASS' if worst >= FLOOR_PT else 'FAIL'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("figures_results"))
    ap.add_argument("--n", type=int, default=5, help="how many zeros to mark")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    fig, ax, zeros, energies, z_turn = build(a.n)
    out = a.out / "fig_airy_zeros.png"
    fig.savefig(out, bbox_inches="tight")

    print(f"z0 = {Z0*1e6:.3f} um | m g z0 = {E0:.4f} peV")
    for i in range(a.n):
        print(f"  n={i+1}  a_n = -{zeros[i]:.4f}   E = {energies[i]:.4f} peV"
              f"   z_n = {z_turn[i]:.2f} um")
    audit(fig, out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
