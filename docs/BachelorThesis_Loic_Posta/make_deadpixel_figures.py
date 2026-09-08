#!/usr/bin/env python3
"""Figures 4.x -- where pixels are lost and what they do before they go, plus
the numbers Section "What Kills a Pixel" quotes.

  fig11_dead_pixel_map.png     losses and tracks on one colour scale, and the
                               two against each other block by block
  fig12_dead_pixel_traces.png  pixel value against frame, lost against surviving

Also printed, and quoted in the text rather than plotted: what the doomed pixels
read in the DARK run, the fraction of losses that fall inside a track footprint,
and whether losses cluster with their eight neighbours beyond what the beam
profile alone would give.

    python3 make_deadpixel_figures.py --runs ../../Code/runs_2026-08-25
"""
import argparse, re, struct
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

CODE  = Path(__file__).resolve().parents[2] / "Code"
BLOCK, STEP, NTRACE = 96, 15, 250


def read_bgm(p):
    """mean, M2, count, dead -- see the format note in BackgroundModel.cpp."""
    with open(p, "rb") as f:
        assert f.read(4) == b"BGMD"
        struct.unpack("<ii", f.read(8)); struct.unpack("<f", f.read(4))
        _, R, C = struct.unpack("<iii", f.read(12))
        n = R * C
        for _ in range(3):
            np.fromfile(f, np.float32, n)
        return np.fromfile(f, np.uint8, n).reshape(R, C), R, C


def natural(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--dark", type=Path,
                    default=CODE / "Data_tot/dark/2026-07-11_3-001_background_575-6ms_2fps")
    ap.add_argument("--signal", type=Path,
                    default=CODE / "Data_tot/signal/2026-07-11_3-002_575-6ms_2fps_first_UCN")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures_results")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    rng = np.random.default_rng(3)

    before, R, C = read_bgm(a.runs / "N500/bg.bgm")
    after, _, _ = read_bgm(a.runs / "signal1500/bg.bgm")
    new = (after > 0) & (before == 0)
    ys, xs = np.nonzero(new)
    print("dead before %d | after %d | lost during the beam-on run %d"
          % ((before > 0).sum(), (after > 0).sum(), new.sum()))

    # Only whole blocks. 5472 / 96 is exactly 57, so the old "+1" added a 58th
    # column holding no pixels at all -- 39 blocks sitting at (0, 0) that
    # anchored the regression on the origin -- and a 39th row a quarter of a
    # block high whose counts were four times too low. Both inflated r.
    hb = (R // BLOCK, C // BLOCK)
    dens = np.zeros(hb)
    ok = (ys // BLOCK < hb[0]) & (xs // BLOCK < hb[1])
    np.add.at(dens, (ys[ok] // BLOCK, xs[ok] // BLOCK), 1)
    d = pd.read_csv(a.runs / "signal1500/detections_classified.csv",
                    usecols=["center_x", "center_y", "bbox_x", "bbox_y",
                             "bbox_w", "bbox_h", "predicted_label"])
    li = d[d.predicted_label == "lithium"]
    arr = np.zeros(hb)
    ay = li.center_y.to_numpy().astype(int) // BLOCK
    ax = li.center_x.to_numpy().astype(int) // BLOCK
    ok = (ay < hb[0]) & (ax < hb[1])
    np.add.at(arr, (ay[ok], ax[ok]), 1)

    vmax = float(np.percentile(arr[arr > 0], 99))
    r = float(np.corrcoef(dens.ravel(), arr.ravel())[0, 1])
    k1, k0 = np.polyfit(arr.ravel(), dens.ravel(), 1)
    print("shared vmax %.1f | r %.3f | slope %.3f | intercept %.2f per block "
          "(= %.2f per frame)" % (vmax, r, k1, k0, k0 * dens.size / 1500))

    # main.tex posts this at \linewidth = 16 cm = 6.30 in. Drawing it at that
    # width instead of 10.6 in means LaTeX no longer shrinks it by 0.66, so the
    # 9 and 10 pt set here are 9 and 10 pt on the page -- at or above the 9 pt
    # of the \figsource line, which is the floor.
    fig = plt.figure(figsize=(6.20, 3.90))
    gs = GridSpec(2, 3, height_ratios=[1, .95], width_ratios=[1, 1, .05], hspace=.30, wspace=.10)
    for i, (m, t) in enumerate([(dens, "(a)"), (arr, "(b)")]):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(m, cmap="magma", origin="upper", vmin=0, vmax=vmax)
        ax.set_title(t, fontsize=10, loc="left"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]), label="counts per $96\\times96$ block")
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(arr.ravel(), dens.ravel(), "o", ms=2.6, alpha=.30, color="#2b6cb0", markeredgewidth=0)
    # The line is drawn only where blocks exist. polyfit above still sees
    # every block, so the slope 0.35, the intercept 6.48 and r = 0.493 are
    # untouched; what changes is that the segment no longer runs back to
    # x = 0, where nothing was measured. The intercept is an extrapolation
    # and the caption says so -- drawing it made it look like a datum.
    x = np.linspace(arr.min(), arr.max(), 50)
    ax.plot(x, k1 * x + k0, "-", color="#c0392b", lw=1.4, label="least squares, slope $%.2f$" % k1)
    ax.set_xlabel("detected neutron tracks in the block")
    ax.set_ylabel("pixels lost\nin the block")
    ax.set_title("(c)", fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(a.out / "fig11_dead_pixel_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── do the losses fall under recorded tracks? ──────────────────────────
    hit = np.zeros((R, C), bool)
    for x0, y0, w, h in zip(li.bbox_x, li.bbox_y, li.bbox_w, li.bbox_h):
        hit[y0:y0 + h, x0:x0 + w] = True
    cov, on = hit.mean(), hit[ys, xs].mean()
    print("track footprint covers %.3f%% of the sensor; %.3f%% of the losses sit in it "
          "-> x%.2f" % (100 * cov, 100 * on, on / cov))

    # ── do they cluster with their neighbours, beyond the beam profile? ────
    k = np.ones((3, 3), np.float32); k[1, 1] = 0
    def pair_frac(mask):
        nb = cv2.filter2D(mask.astype(np.float32), cv2.CV_32F, k, borderType=cv2.BORDER_CONSTANT)
        return float((nb[mask] > 0).mean())
    obs = pair_frac(new)
    null = []
    for _ in range(5):
        sh = np.zeros_like(new)
        for by in range(0, R, BLOCK):
            for bx in range(0, C, BLOCK):
                blk = new[by:by + BLOCK, bx:bx + BLOCK]
                cnt = int(blk.sum())
                if not cnt:
                    continue
                h, w = blk.shape
                f = np.zeros(h * w, bool); f[rng.choice(h * w, cnt, replace=False)] = True
                sh[by:by + BLOCK, bx:bx + BLOCK] = f.reshape(h, w)
        null.append(pair_frac(sh))
    null = np.array(null)
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8), connectivity=8)
    sz = stats[1:, cv2.CC_STAT_AREA]
    print("adjacent pairs: observed %.4f%% | density-preserving null %.4f%% +- %.4f%% -> x%.2f"
          % (100 * obs, 100 * null.mean(), 100 * null.std(), obs / null.mean()))
    print("clumps: %d | singletons %.2f%% | largest %d px" % (nlab - 1, 100 * (sz == 1).mean(), sz.max()))

    # ── traces, beam-on and dark ───────────────────────────────────────────
    sel = rng.choice(len(ys), NTRACE, replace=False)
    py, px = ys[sel], xs[sel]
    cy = rng.integers(0, R, NTRACE * 5); cx = rng.integers(0, C, NTRACE * 5)
    keep = after[cy, cx] == 0
    cy, cx = cy[keep][:NTRACE], cx[keep][:NTRACE]

    def sample(folder, step):
        files = sorted(folder.glob("*.tiff"), key=natural)
        idx = list(range(0, len(files), step))
        D = np.zeros((len(py), len(idx))); K = np.zeros((len(cy), len(idx)))
        for j, i in enumerate(idx):
            im = cv2.imread(str(files[i]), cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            D[:, j] = im[py, px]; K[:, j] = im[cy, cx]
        return np.array(idx) + 1, D, K

    fr, D, K = sample(a.signal, STEP)
    # .78\linewidth = 4.91 in on the page; drawn 4.77 in wide, so the scale
    # factor is 1.03 upward rather than 0.68 downward.
    fig, ax = plt.subplots(figsize=(4.77, 2.65))
    ax.plot(fr, D.mean(0), "o", color="#c0392b", ms=4,
            label="pixels lost during the run (mean of %d)" % len(py))
    ax.plot(fr, K.mean(0), "s", color="#2b6cb0", ms=4,
            label="surviving pixels (mean of %d)" % len(cy))
    ax.set_xlabel("beam-on frame"); ax.set_ylabel("pixel value  [ADU]")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(a.out / "fig12_dead_pixel_traces.png", dpi=300)
    plt.close(fig)
    print("beam-on: doomed %.2f -> %.2f ADU | control %.2f -> %.2f ADU"
          % (D[:, :3].mean(), D[:, -3:].mean(), K[:, :3].mean(), K[:, -3:].mean()))

    _, Dd, Kd = sample(a.dark, 25)
    print("dark   : doomed mean %.3f ADU (non-zero %.1f%%) | control %.3f ADU "
          "(non-zero %.1f%%) -> x%.0f"
          % (Dd.mean(), 100 * (Dd > 0).mean(), Kd.mean(), 100 * (Kd > 0).mean(),
             Dd.mean() / max(Kd.mean(), 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
