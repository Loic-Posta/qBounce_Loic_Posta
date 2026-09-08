#!/usr/bin/env python3
"""Young's double slit against Ramsey spectroscopy, the two lineshapes side by
side.

The panel this replaces was drawn at half the column width and already carried
two sub-plots inside that half, so its tick labels printed at 2.2 pt against a
8.97 pt floor. Nothing was wrong with its physics; it was unreadable.

Both panels show the same structure: a broad envelope set by one aperture, fast
fringes set by the separation between two. In space the aperture is the slit
width and the separation is the slit spacing; in time the aperture is the
duration of one oscillating region and the separation is the free flight between
two. Both axes carry real units, the temporal one taken from the spectrometer of
Section 2.4, so a reader can see that the fringe spacing is tens of hertz rather
than an arbitrary number.

The envelope in panel (b) is computed from the Ramsey expression itself and not
from a Rabi lineshape of some other pulse area. That matters. Only the
free-flight phase carries the fringes, so maximising over it gives exactly the
curve the fringes touch and never cross; drawing an unrelated Rabi curve there
lets the fringes run outside their own envelope, which is what an earlier
version of this figure did.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

NOTE_PT = 8.72        # the \figsource line, which is the floor for any text here

# Same palette and rcParams as make_results_figures.py, so the figure does not
# announce itself as coming from a different script.
S1, S2 = "#4a9eda", "#e8232a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8b8a85"

mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

# ── the spatial side: an ordinary optical double slit ──────────────────────
WAVELENGTH = 550e-9      # m, green light
SLIT_WIDTH = 0.10e-3     # m
SLIT_SEP: float          # m, fixed below from the spectrometer's geometry
SCREEN_DIST = 1.0        # m

# ── the temporal side: the spectrometer of Section 2.4 ─────────────────────
# Representative values, of the order of the spectrometer's and chosen to put a
# legible number of fringes under the envelope. This figure explains; it does
# not measure. Chapter 4 is where the apparatus is characterised.
L_PULSE = 0.150          # m, one oscillating region
L_FREE = 0.340           # m, free flight between two
V_X = 8.0                # m/s, horizontal velocity
SLIT_SEP = SLIT_WIDTH * L_FREE / L_PULSE


def young(x: np.ndarray):
    """Single-slit envelope and the double-slit intensity it modulates.

    x is a position on the screen in metres. The small-angle forms are used; at
    one metre and a tenth of a millimetre of slit they are exact far beyond the
    width of the plotted line.
    """
    u = SLIT_WIDTH * x / (WAVELENGTH * SCREEN_DIST)
    envelope = np.sinc(u) ** 2                    # numpy's sinc carries the pi
    fringes = envelope * np.cos(np.pi * SLIT_SEP * x
                                / (WAVELENGTH * SCREEN_DIST)) ** 2
    return envelope, fringes


def ramsey(nu: np.ndarray):
    """Ramsey transition probability against detuning in hertz, and its envelope.

    Two pi/2 regions of duration tau separated by a free flight T, the standard
    separated-oscillatory-fields result

        P = 4 (W/Weff)^2 sin^2(t) [cos(t)cos(f) - (d/Weff) sin(t)sin(f)]^2

    with t = Weff*tau/2, f = d*T/2 and d the angular detuning. Only f carries the
    free flight, so the fringes live entirely inside the bracket and everything
    in front of it is the envelope. Since a*cos(f) + b*sin(f) has amplitude
    sqrt(a^2 + b^2), maximising the bracket over f gives

        cos^2(t) + (d/Weff)^2 sin^2(t)

    and the envelope returned here is the curve the fringes touch and never
    cross.
    """
    tau = L_PULSE / V_X                 # s, time spent in one oscillating region
    free = L_FREE / V_X                 # s, time of flight between the two
    omega = np.pi / (2.0 * tau)         # rad/s, a pi/2 pulse in that time
    d = 2.0 * np.pi * nu                # angular detuning

    eff = np.hypot(omega, d)
    theta = 0.5 * eff * tau
    phi = 0.5 * d * free

    front = 4.0 * (omega / eff) ** 2 * np.sin(theta) ** 2
    bracket = np.cos(theta) * np.cos(phi) - (d / eff) * np.sin(theta) * np.sin(phi)
    peak = np.cos(theta) ** 2 + (d / eff) ** 2 * np.sin(theta) ** 2
    # Le temps de phase effectif n'est pas le seul vol libre : chaque impulsion
    # pi/2 y contribue pour 2*tau/pi, d'ou une periode T + 4*tau/pi.
    return front * bracket ** 2, front * peak, 1.0 / (free + 4.0 * tau / np.pi)


def fwhm(t: np.ndarray, y: np.ndarray) -> float:
    """Full width at half maximum of a single symmetric peak, by interpolation
    on the half-maximum crossing. Used to scale the two panels to the same
    printed envelope width."""
    half = 0.5 * y.max()
    above = np.flatnonzero(y >= half)
    lo, hi = above[0], above[-1]
    # linear interpolation on each flank, so the answer does not depend on the
    # sampling step
    xl = np.interp(half, [y[lo - 1], y[lo]], [t[lo - 1], t[lo]]) if lo else t[0]
    xr = (np.interp(half, [y[hi + 1], y[hi]], [t[hi + 1], t[hi]])
          if hi + 1 < len(t) else t[-1])
    return float(xr - xl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "figures_results")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)

    # \textwidth is 16 cm = 6.30 in; posted at \linewidth, so drawn at that
    # width and the scale factor on the page is 1. The height carries the
    # legends, which sit under the axes: the labels are long and every position
    # inside a panel is crossed by one of the curves.
    fig, ax = plt.subplots(1, 2, figsize=(6.19, 3.20), constrained_layout=True)

    # ── (a) space ──────────────────────────────────────────────────────────
    x = np.linspace(-16e-3, 16e-3, 12000)
    env, fr = young(x)
    xm = x * 1e3                                     # plotted in millimetres
    w_space = fwhm(xm, env)
    ax[0].plot(xm, env, "--", color=S2, lw=1.7,
               label="Envelope (Single Slit Diffraction)")
    ax[0].plot(xm, fr, color=S1, lw=1.5,
               label="Intensity (Young's Double Slit)")
    ax[0].fill_between(xm, fr, color=S1, alpha=0.15)
    ax[0].set(xlabel="Screen Position  [mm]", ylabel="Light Intensity (normalised)",
              ylim=(-0.03, 1.08))
    ax[0].set_title("a) Spatial double-slit", loc="left", fontweight="bold",
                    fontsize=10)
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.26),
                 fontsize=NOTE_PT, frameon=False, handlelength=2.0,
                 borderpad=0.4)

    # ── (b) time ───────────────────────────────────────────────────────────
    nu = np.linspace(-90, 90, 16000)
    p_ram, p_env, spacing = ramsey(nu)
    w_time = fwhm(nu, p_env)
    ax[1].plot(nu, 1.0 - p_env, "--", color=S2, lw=1.7,
               label="Rabi Spectroscopy (Envelope)")
    ax[1].plot(nu, 1.0 - p_ram, color=S1, lw=1.5,
               label="Ramsey Spectroscopy (AM Modulation)")
    ax[1].fill_between(nu, 1.0 - p_ram, 1.0, color=S1, alpha=0.15)
    ax[1].set(xlabel=r"Vibration Frequency $\nu - \nu_0$  [Hz]",
              ylabel="Neutron Transmission", ylim=(-0.03, 1.08))
    ax[1].set_title("b) Temporal double-slit", loc="left", fontweight="bold",
                    fontsize=10)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.26),
                 fontsize=NOTE_PT, frameon=False, handlelength=2.0,
                 borderpad=0.4)

    # Le meme multiple de la largeur a mi-hauteur de chaque cote : les deux
    # enveloppes occupent alors exactement la meme fraction de leur panneau.
    SPAN = 1.85
    ax[0].set_xlim(-SPAN * w_space, SPAN * w_space)
    ax[1].set_xlim(-SPAN * w_time, SPAN * w_time)

    out = a.out / "fig_ramsey_analogy.png"
    fig.savefig(out)
    plt.close(fig)

    # Two checks worth keeping in the log: the fringe spacing must come out as
    # 1/T, and the fringes must not leave their envelope.
    inside = bool(np.all(p_ram <= p_env + 1e-9))
    print(f"wrote {out}")
    # Mesuree sur la courbe, pas deduite : c'est le seul controle qui vaille.
    from scipy.signal import find_peaks
    pk, _ = find_peaks(p_ram, height=0.05)
    measured = float(np.median(np.diff(nu[pk]))) if len(pk) > 2 else float("nan")
    print(f"  temporal fringe spacing  predicted {spacing:.1f} Hz, "
          f"measured on the curve {measured:.1f} Hz")
    print(f"  optical fringe spacing      = "
          f"{1e3 * WAVELENGTH * SCREEN_DIST / SLIT_SEP:.2f} mm")
    print(f"  fringes inside their envelope: {inside}")
    print(f"  envelope FWHM: {w_space:.2f} mm and {w_time:.1f} Hz, "
          f"both drawn to +/-{SPAN:g} FWHM")
    return 0 if inside else 1


if __name__ == "__main__":
    raise SystemExit(main())
