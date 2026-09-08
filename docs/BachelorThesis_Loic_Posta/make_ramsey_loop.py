#!/usr/bin/env python3
"""Where a feedback loop spends the beam, and what that is worth.

Section 5 argues that a sensor returning a profile lets a resonance scan decide
where to go next instead of following a grid laid out in advance. That argument
is made in seconds of beam time, and seconds are hard to picture. This draws it.

Panel (a) is one scan. The loop first samples coarsely across the whole range,
at a precision chosen only to answer "is there contrast, and roughly where" --
those are the open points, with the large error bars that go with a short dwell.
It then spends the rest of the beam inside half a fringe of the best sample,
which is where the filled points are. A fixed grid would have paid the same
price at all thirty frequencies whether or not the resonance was near them.

Panel (b) is the ladder of Section 5, every rung reaching the same error on the
transferred fraction so that the comparison is a comparison.

Nothing here is a measurement. The lineshape is the separated-oscillatory-fields
result at the geometry of Section 2.4, the counting is Poisson at the rate this
detector was measured to sustain, and the scan is simulated by
Code/src/ramsey_feedback.py, which this file imports rather than reimplements.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Code" / "src"))
import ramsey_feedback as rf          # noqa: E402

# Drawn at the width it prints at, so the scale factor on the page is 1 and a
# point here is a point there. The floor is the 8.97 pt of the \figsource line.
FULL_W, NOTE_PT = 6.19, 9.0
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "figures_results")
    ap.add_argument("--seeds", type=int, default=24,
                    help="scans searched for the representative draw of panel (a)")
    ap.add_argument("--repeats", type=int, default=300,
                    help="scans averaged for the beam times of panel (b)")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)

    cfg = rf.Config()
    _centres, b1, b2, _sc = rf.sensor_templates()
    fit = rf.ProfileFit(b1, b2)

    # Panel (a) shows one scan, and which one is a choice that has to be made
    # honestly. Picking the draw that places the fringe best would flatter the
    # method; picking one at random can land on a tail. The rule here is the
    # median: run the first seeds, take the scan whose residual is closest to
    # the median residual of them all, and draw that one. It is a
    # representative scan by construction, and the rule is in the code.
    draws = []
    for seed in range(1, a.seeds + 1):
        s_a = rf.adaptive_scan(np.random.default_rng([seed, 1]), fit, b1, b2, cfg)
        if s_a.abandoned:
            continue
        # Only scans that locked onto the central fringe are eligible. That is
        # not a filter for flattery: the text is explicit that the fringe order
        # has to come from knowing nu12 beforehand, and this figure illustrates
        # a scan run under that assumption. A draw that settled on a side fringe
        # would be showing the ambiguity, which is a different statement and is
        # made in prose with its own numbers.
        central = abs(s_a.centre - rf.NU12) < 0.5 * rf.FRINGE
        if central:
            draws.append((abs(rf.in_fringe(s_a.centre)), seed, s_a))
    if not draws:
        print("no seed found the central fringe; widen --seeds")
        return 1
    # Among those, the one whose residual is closest to the median of them all,
    # so the drawn scan is typical of its kind rather than the best of it.
    median = float(np.median([d[0] for d in draws]))
    _res, drawn_seed, scan = min(draws, key=lambda d: abs(d[0] - median))
    survey = scan.points[:-cfg.fine_points]
    fine = scan.points[-cfg.fine_points:]

    # Three panels, because the survey and the zoom are two different
    # measurements and drawing them on one pair of axes hid that: they occupy
    # the same few hertz, so their error bars stacked on top of one another and
    # the reader could not see which was which. Separated, the difference is
    # the point of the figure -- a wide coarse pass, then a narrow precise one.
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 2.85), constrained_layout=True,
                           width_ratios=[1.30, 1.0, 0.95])

    nu = np.linspace(rf.NU12 - cfg.span, rf.NU12 + cfg.span, 4000)
    curve = cfg.f_max * rf.ramsey_transfer(nu)
    half = 0.5 * rf.FRINGE
    lo = min(p.nu for p in fine) - rf.NU12
    hi = max(p.nu for p in fine) - rf.NU12

    # Panels (a) and (b) are two stages of ONE method, the bottom rung of (c),
    # so they carry that rung's colour and are told apart by the fill rather
    # than by the hue. Colouring the survey blue made it echo the live-stop bar
    # of panel (c), which is a different method entirely and is not drawn here.
    # ── (a) stage one: a coarse pass over the whole range ──────────────────
    ax[0].plot(nu - rf.NU12, curve, color=MUTED, lw=1.0, zorder=1)
    ax[0].axvspan(lo, hi, color=S2, alpha=0.13, lw=0, zorder=0)
    x = np.array([q.nu for q in survey]) - rf.NU12
    ax[0].errorbar(x, [q.f_hat for q in survey], yerr=[q.f_err for q in survey],
                   fmt="o", ms=4.0, color=S2, mfc="white", elinewidth=1.0,
                   capsize=2, lw=0, zorder=3)
    ax[0].set(xlabel=r"$\nu - \nu_{12}$  [Hz]", ylabel="transferred fraction $f$")
    ax[0].set_title("a) the loop: survey", loc="left", fontweight="bold")
    ax[0].annotate(f"{len(survey)} points, coarse", xy=(0.03, 0.97),
                   xycoords="axes fraction", va="top", fontsize=NOTE_PT,
                   color=INK2)

    # ── (b) stage two: the same beam, spent inside one fringe ──────────────
    ax[1].plot(nu - rf.NU12, curve, color=MUTED, lw=1.0, zorder=1)
    ax[1].axvspan(lo, hi, color=S2, alpha=0.13, lw=0, zorder=0)
    x = np.array([q.nu for q in fine]) - rf.NU12
    ax[1].errorbar(x, [q.f_hat for q in fine], yerr=[q.f_err for q in fine],
                   fmt="o", ms=4.0, color=S2, elinewidth=1.0, capsize=2, lw=0,
                   zorder=3)
    ax[1].axvline(scan.centre - rf.NU12, color=INK2, lw=1.0, ls="--", zorder=2)
    ax[1].set(xlabel=r"$\nu - \nu_{12}$  [Hz]", xlim=(lo - 0.35 * half,
                                                       hi + 0.35 * half))
    ax[1].set_title("b) the loop: zoom", loc="left", fontweight="bold")
    ax[1].annotate(f"{len(fine)} points, to ${cfg.sigma_fine:g}$", xy=(0.03, 0.97),
                   xycoords="axes fraction", va="top",
                   fontsize=NOTE_PT, color=INK2)
    top = max(q.f_hat + q.f_err for q in survey + fine)
    bot = min(q.f_hat - q.f_err for q in survey + fine)
    ax[0].set_ylim(bot - 0.02, top + 0.16 * (top - bot))
    ax[1].sharey(ax[0])
    ax[1].tick_params(labelleft=False)

    # ── (b) the ladder, every rung at the same error bar ───────────────────
    # The bars are the numbers Section 5 quotes, so they are the means over the
    # same repeats and not the single scan of panel (a), which would differ by
    # its own draw.
    adapt_s, naive_s = [], []
    for r in range(1, a.repeats + 1):
        s_a, s_n = rf.run_pair(r, fit, b1, b2, cfg)
        adapt_s.append(s_a.seconds)
        naive_s.append(s_n.seconds)
    absorber = cfg.naive_points * (1.0 / cfg.sigma_fine ** 2 / rf.RATE
                                   + cfg.settle)
    fixed_dwell = cfg.naive_points * (
        (_sc["width"] / (cfg.sigma_fine * _sc["shift"])) ** 2 / rf.RATE
        + cfg.settle)
    rungs = [("absorber,\nfixed dwell", absorber, MUTED),
             ("sensor,\nfixed dwell", fixed_dwell, S3),
             ("sensor,\nlive stop", float(np.mean(naive_s)), S1),
             ("sensor,\nfeedback loop", float(np.mean(adapt_s)), S2)]
    ypos = np.arange(len(rungs))[::-1]
    ax[2].barh(ypos, [r[1] for r in rungs], height=0.6,
               color=[r[2] for r in rungs])
    for y, (_lab, secs, _c) in zip(ypos, rungs):
        ax[2].annotate(f"{int(secs + 0.5)} s", (secs, y), xytext=(4, 0),
                       textcoords="offset points", va="center",
                       fontsize=NOTE_PT, color=INK2)
    ax[2].set_yticks(ypos, [r[0] for r in rungs], fontsize=NOTE_PT)
    ax[2].set_xlim(0, max(r[1] for r in rungs) * 1.26)
    ax[2].set_xlabel("beam time, one scan  [s]")
    ax[2].set_title("c) at the same error bar", loc="left", fontweight="bold")
    ax[2].grid(axis="y", visible=False)

    out = a.out / "fig_ramsey_loop.png"
    fig.savefig(out)
    plt.close(fig)

    loop_s = rungs[-1][1]
    print(f"wrote {out}")
    print(f"  panel (a): seed {drawn_seed}, chosen among the {len(draws)} of "
          f"{a.seeds} that found the central fringe as the one closest to "
          f"their median residual ({median:.2f} Hz)")
    print(f"  survey {len(survey)} points, zoom {len(fine)} points, "
          f"{scan.events} neutrons, {scan.seconds:.1f} s, "
          f"fringe {rf.in_fringe(scan.centre):+.2f} Hz off centre")
    print(f"  panel (b), means over {a.repeats} scans: "
          + ", ".join(f"{l.replace(chr(10), ' ')} {v:.0f} s"
                      for l, v, _ in rungs))
    print(f"  end to end x{absorber / loop_s:.1f}, "
          f"of which x{absorber / fixed_dwell:.1f} is the profile against the rate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
