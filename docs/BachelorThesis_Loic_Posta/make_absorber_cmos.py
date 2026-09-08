#!/usr/bin/env python3
"""What a pixel sensor placed where the absorber sits would actually record.

The outlook argues that a position-sensitive detector in one of the analysing
regions would return the vertical density instead of a transmission. This draws
that: the densities of the first three gravitational states, then the same
densities as the sensor would see them once each capture has been displaced by
the flight of the daughter ion in the silicon, then what a change of state
population does to the profile.

Nothing here is a measurement. The state densities are Airy functions and the
displacement is the only free input -- taken as 2 um, of the order of the
stopping range of the daughters, and the figure prints how the conclusion moves
if that number is wrong.

The last two panels are the point of the whole thing. A transmission detector
returns one number per setting, the fraction that got through. A position
sensitive one returns a profile, and the profile is a mixture of densities that
are known in advance -- so there is nothing to invert: fitting the mixing
weights returns the population of every state at once, each with its own error
bar. Panel (c) is the profile as counts, panel (d) is the population vector the
fit reads out of it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import airy, ai_zeros

# The floor is the 8.97 pt of the \figsource line, measured in main.pdf.
# make_results_figures.py divides it by 1.029 because its figures are drawn
# narrower than they print; this one is posted at \linewidth and comes out of
# savefig at exactly 6.30 in, so LaTeX neither shrinks nor enlarges it and the
# floor applies unscaled. Copying the 8.72 from there put the legends and the
# note of panel (b) at 8.72 pt on paper, a quarter of a point under. 9 pt, the
# size everything else in this figure carries, clears it with nothing to argue.
NOTE_PT = 9.0
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
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

HBAR = 1.054571817e-34
M_N = 1.67492750056e-27
G = 9.80665
PITCH = 2.4          # um, the sensor of Section 2.4
DISPLACEMENT = 2.0   # um, order of the daughter's range in silicon

Z0 = (HBAR ** 2 / (2 * M_N ** 2 * G)) ** (1 / 3) * 1e6      # um
ZEROS = ai_zeros(6)[0]


def density(z: np.ndarray, n: int) -> np.ndarray:
    """|psi_n(z)|^2, normalised to unit integral. n counts from one."""
    v = airy(z / Z0 + ZEROS[n - 1])[0] ** 2
    return v / np.trapezoid(v, z)


def blur(z: np.ndarray, p: np.ndarray, sigma: float) -> np.ndarray:
    """Convolve with a Gaussian of width sigma, on the grid z."""
    dz = z[1] - z[0]
    half = int(np.ceil(5 * sigma / dz))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * dz / sigma) ** 2)
    return np.convolve(p, k / k.sum(), mode="same")


def fit_populations(counts, basis, n_boot=400, seed=0):
    """Populations of every state, from one recorded profile.

    The recorded profile is a mixture of densities that are known before the
    run, so the only unknowns are the weights. Expectation-maximisation on the
    multinomial likelihood gives them -- the update is exact for a mixture and
    keeps every weight positive without a constraint having to be imposed by
    hand. The errors come from refitting profiles resampled from the fit, which
    costs nothing at these counts and needs no assumption of a parabolic
    likelihood near a weight that sits close to zero.
    """
    def em(k, iters=600):
        w = np.full(len(basis), 1.0 / len(basis))
        for _ in range(iters):
            mix = w @ basis
            resp = basis * w[:, None] / np.clip(mix, 1e-300, None)
            w = (resp * k).sum(axis=1) / k.sum()
        return w

    w_hat = em(np.asarray(counts, float))
    rng = np.random.default_rng(seed)
    n = int(np.sum(counts))
    draws = np.array([em(rng.multinomial(n, w_hat @ basis)) for _ in range(n_boot)])
    lo, hi = np.percentile(draws, [15.865, 84.135], axis=0)
    return w_hat, lo, hi


def moments(z, p):
    m = np.trapezoid(z * p, z)
    return m, float(np.sqrt(np.trapezoid((z - m) ** 2 * p, z)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "figures_results")
    ap.add_argument("--seed", type=int, default=3,
                    help="the draw in panel (d); the figure is reproducible")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)

    z = np.linspace(0, 60, 40000)
    fig, axes = plt.subplots(2, 2, figsize=(6.19, 4.30), constrained_layout=True)
    ax = axes.ravel()

    # (a) the states themselves
    for n, c in zip((1, 2, 3), (S1, S2, S3)):
        ax[0].plot(z, density(z, n) * 1e3, color=c, lw=1.4, label=f"$n={n}$")
    ax[0].set(xlabel=r"height $z$  [$\mu$m]", ylabel=r"density  [$10^{-3}\,\mu$m$^{-1}$]",
              xlim=(0, 50))
    ax[0].set_title("a) the states", loc="left", fontweight="bold")
    ax[0].legend(fontsize=NOTE_PT, loc="upper right")

    # (b) the same, as the sensor records them
    for n, c in zip((1, 2, 3), (S1, S2, S3)):
        ax[1].plot(z, blur(z, density(z, n), DISPLACEMENT) * 1e3, color=c, lw=1.4)
    ax[1].set(xlabel=r"height $z$  [$\mu$m]",
              ylabel=r"density  [$10^{-3}\,\mu$m$^{-1}$]", xlim=(0, 50))
    ax[1].set_title("b) as recorded", loc="left", fontweight="bold")
    ax[1].annotate(f"blurred by {DISPLACEMENT:g}" + r"$\,\mu$m" + "\n(daughter range)",
                   xy=(0.97, 0.94), xycoords="axes fraction", ha="right", va="top",
                   fontsize=NOTE_PT, color=INK2)

    # (c) a transition, as the sensor would actually record it: counts
    p1 = blur(z, density(z, 1), DISPLACEMENT)
    p2 = blur(z, density(z, 2), DISPLACEMENT)

    N_STATES, N_EVENTS = 5, 660
    TRUTH = np.array([0.60, 0.25, 0.10, 0.03, 0.02])
    edges = np.arange(0, 60 + PITCH, PITCH)          # the sensor's own binning
    centres = 0.5 * (edges[1:] + edges[:-1])
    basis = np.array([np.histogram(z, bins=edges,
                                   weights=blur(z, density(z, n), DISPLACEMENT))[0]
                      for n in range(1, N_STATES + 1)])
    basis /= basis.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(a.seed)
    counts = rng.multinomial(N_EVENTS, TRUTH @ basis)
    w_hat, w_lo, w_hi = fit_populations(counts, basis, seed=a.seed)

    ax[2].step(centres, counts, where="mid", color=INK2, lw=1.0,
               label=f"{N_EVENTS} neutrons")
    ax[2].errorbar(centres, counts, yerr=np.sqrt(counts), fmt="none",
                   ecolor=INK2, elinewidth=0.7, alpha=.6)
    ax[2].plot(centres, N_EVENTS * (w_hat @ basis), color=S2, lw=1.6,
               label="fitted mixture")
    ax[2].plot(centres, N_EVENTS * basis[0], color=S1, lw=1.2, ls="--",
               label="all in $n=1$")
    ax[2].set(xlabel=r"height $z$  [$\mu$m]", ylabel="neutrons per pixel row",
              xlim=(0, 55))
    ax[2].set_title("c) as counts", loc="left", fontweight="bold")
    ax[2].legend(fontsize=NOTE_PT, loc="upper right")

    # (d) the population of each state, which is what the fit is for
    ns = np.arange(1, N_STATES + 1)
    ax[3].bar(ns, N_EVENTS * TRUTH, width=0.62, color="#dcdbd6",
              edgecolor="none", label="injected")
    err = np.vstack([N_EVENTS * (w_hat - w_lo), N_EVENTS * (w_hi - w_hat)])
    ax[3].errorbar(ns, N_EVENTS * w_hat, yerr=err, fmt="o", ms=5, color=S2,
                   capsize=3, elinewidth=1.2, label="fitted")
    ax[3].set(xlabel="state $n$", ylabel="neutrons in that state",
              xticks=ns, ylim=(0, None))
    ax[3].set_title("d) the populations", loc="left", fontweight="bold")
    ax[3].legend(fontsize=NOTE_PT, loc="upper right")

    out = a.out / "fig_absorber_cmos.png"
    fig.savefig(out)
    plt.close(fig)

    # The numbers the caption quotes, recomputed here so the two cannot drift.
    m1, s1 = moments(z, p1)
    m2, s2 = moments(z, p2)
    shift = m2 - m1
    width = 0.5 * (s1 + s2)
    print(f"wrote {out}")
    print(f"  pitch {PITCH} um, displacement {DISPLACEMENT} um")
    print(f"  mean height  n=1 {m1:.2f} um,  n=2 {m2:.2f} um,  shift {shift:.2f} um")
    print(f"  profile width ~{width:.1f} um")
    for f in (0.30, 0.10, 0.05):
        n_ev = (3 * width / (f * shift)) ** 2
        print(f"  detect a {100*f:.0f}% transfer at 3 sigma: {n_ev:.0f} events"
              f"  = {n_ev/71.1:.0f} s at 71 per second")
    print(f"  panel (c)/(d): {N_EVENTS} neutrons over {N_STATES} states")
    for n, t, w, lo, hi in zip(ns, TRUTH, w_hat, w_lo, w_hi):
        print(f"    n={n}  injected {N_EVENTS*t:6.1f}   fitted {N_EVENTS*w:6.1f}"
              f"  [{N_EVENTS*lo:5.1f}, {N_EVENTS*hi:5.1f}]"
              f"   {'ok' if lo <= t <= hi else 'OUTSIDE 68%'}")

    # One draw says nothing about whether the error bars are honest, and this
    # one has n=1 a little outside its interval -- which is what 68 % means.
    # Two hundred draws say it: the fraction of intervals that cover the
    # injected value has to come out near 0.68, and it does.
    cover = np.zeros(N_STATES)
    rng2 = np.random.default_rng(a.seed + 1000)
    for i in range(200):
        c = rng2.multinomial(N_EVENTS, TRUTH @ basis)
        _, lo_i, hi_i = fit_populations(c, basis, n_boot=120, seed=i)
        cover += (lo_i <= TRUTH) & (TRUTH <= hi_i)
    print("  coverage of the 68% intervals over 200 draws: "
          + ", ".join(f"n={n} {c/200:.2f}" for n, c in zip(ns, cover)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
