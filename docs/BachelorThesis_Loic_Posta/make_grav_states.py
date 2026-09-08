#!/usr/bin/env python3
"""
make_grav_states.py - the first five gravitational states of the neutron.

Replaces a slide export that carried its title inside the image, had a
four-point axis label and no energy axis at all. Everything here is computed
from the Airy zeros rather than traced, so the energies on the vertical axis
are the physical ones.

    python3 make_grav_states.py --out figures_results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import airy, ai_zeros

# Constants, CODATA 2022 values as used throughout the thesis.
HBAR = 1.054_571_817e-34      # J s
M_N  = 1.674_927_500e-27      # kg
G    = 9.806_65               # m s^-2
PEV  = 1.602_176_634e-31      # J per peV

# z0 = (hbar^2 / (2 m^2 g))^(1/3) sets the length scale; m g z0 sets the energy.
Z0 = (HBAR ** 2 / (2 * M_N ** 2 * G)) ** (1 / 3)
E0 = M_N * G * Z0 / PEV       # peV

# Font sizes are printed sizes. main.tex places this figure at .86\linewidth
# of a 16 cm text block, i.e. 5.42 in, so the figure is drawn 5.27 in wide
# (5.15 in of axes plus the 0.12 in that bbox_inches="tight" adds back) and
# LaTeX enlarges it by 1.03 instead of shrinking it by 0.74. Nothing here is
# then smaller on the page than the 9 pt of the \figsource line.
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.grid": True, "grid.alpha": .25, "figure.dpi": 300,
})

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("figures_results"))
    ap.add_argument("--n", type=int, default=5, help="how many states to draw")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    zeros = -ai_zeros(a.n)[0]                  # |a_n|, ascending
    E = zeros * E0                             # peV
    # far enough right that the highest state passes its classical turning
    # point at |a_n| z0 = 46.6 um and is visibly decaying; stopping at 42 um
    # left n=4 and n=5 still oscillating, which reads as if they never decay.
    z_um = np.linspace(0, 68, 2200)            # micrometres
    xi = z_um * 1e-6 / Z0

    fig, ax = plt.subplots(figsize=(5.15, 3.6), constrained_layout=True)

    # the linear potential the states live in
    ax.plot(z_um, z_um * 1e-6 / Z0 * E0, color="0.25", lw=1.4,
            label=r"$V(z)=m_n g z$")

    # one state per energy, amplitude rescaled for legibility
    cmap = plt.cm.viridis(np.linspace(.12, .86, a.n))
    for n in range(a.n):
        psi = airy(xi - zeros[n])[0]
        # The amplitude is a drawing choice, not a physical quantity, so it has
        # to be small enough that no state reaches its neighbour: the closest
        # pair, n=4 and n=5, is 0.696 peV apart, and half of that is the ceiling.
        psi = psi / np.abs(psi).max() * (E0 * 0.45)      # rescaled, no units
        ax.axhline(E[n], color=cmap[n], lw=.8, ls=":", alpha=.7)
        ax.plot(z_um, E[n] + psi, color=cmap[n], lw=1.9)
        ax.fill_between(z_um, E[n], E[n] + psi, color=cmap[n], alpha=.16)

        # the classical turning point, where the diagonal crosses this energy
        zt = zeros[n] * Z0 * 1e6
        ax.plot([zt], [E[n]], marker="o", ms=4.5, color=cmap[n],
                mec="white", mew=.9, zorder=5)
        # n and energy together, in the flat region on the right where every
        # state has decayed and nothing else is drawn.
        ax.annotate(f"$n={n+1}$   {E[n]:.2f} peV", xy=(67.2, E[n] + E0 * .06),
                    fontsize=10, color=cmap[n], ha="right", va="bottom")

    ax.set_xlabel(r"height above the mirror, $z$  [$\mu$m]")
    ax.set_ylabel(r"energy  [peV]")
    ax.set_xlim(0, 68)
    ax.set_ylim(0, E[-1] + E0 * 1.0)
    ax.plot([], [], "o", ms=4.5, color="0.35", mec="white", mew=.9,
            label="classical turning point")
    ax.legend(loc="lower right", framealpha=.92)

    out = a.out / "fig_grav_states.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"z0 = {Z0*1e6:.3f} um | m g z0 = {E0:.4f} peV")
    for n in range(a.n):
        print(f"  n={n+1}  |a_n| = {zeros[n]:.4f}   E = {E[n]:.4f} peV")
    print("wrote", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
