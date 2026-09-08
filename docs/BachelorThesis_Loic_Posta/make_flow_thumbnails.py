#!/usr/bin/env python3
"""The eight thumbnails embedded in the data-flow figure (dataflow_figure.tex).

Every one is made from the real data rather than drawn as an icon: the same
256x256 corner of the same two frames runs through the first four stages, so the
reader can follow one patch of sensor from raw pixels to a labelled cluster.

    python3 make_flow_thumbnails.py --runs ../../Code/runs_2026-08-25
"""
import argparse, re, struct
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cv2

CODE = Path(__file__).resolve().parents[2] / "Code"
DARKD = CODE / "Data_tot/dark/2026-07-11_3-001_background_575-6ms_2fps"
SIGD  = CODE / "Data_tot/signal/2026-07-11_3-002_575-6ms_2fps_first_UCN"
X0, Y0, S = 2600, 1700, 128          # the patch every stage shows


def natural(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def read_bgm(p):
    with open(p, "rb") as f:
        assert f.read(4) == b"BGMD"
        struct.unpack("<ii", f.read(8)); struct.unpack("<f", f.read(4))
        _, R, C = struct.unpack("<iii", f.read(12))
        n = R * C
        mean = np.fromfile(f, np.float32, n).reshape(R, C)
        M2 = np.fromfile(f, np.float32, n).reshape(R, C)
        cnt = np.fromfile(f, np.float32, n).reshape(R, C)
    return mean, np.sqrt(M2 / np.maximum(cnt, 1))


def save(arr, path, cmap="magma", vmax=None):
    fig = plt.figure(figsize=(1.6, 1.6), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    fig.savefig(path, transparent=False); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures_flow")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    # pick the patch rather than fixing it by hand: the 128x128 window of the
    # chosen beam-on frame that holds the most clusters, so the four thumbnails
    # of row 1 actually show something
    det = pd.read_csv(a.runs / "signal1500/detections_classified.csv",
                      usecols=["image_id", "center_x", "center_y", "size_px"])
    fid = int(det.image_id.min()) + 400
    f = det[(det.image_id == fid) & (det.size_px >= 3)]
    if len(f) > 3:
        gx = (f.center_x // S).astype(int); gy = (f.center_y // S).astype(int)
        best = f.groupby([gy, gx]).size().idxmax()
        y0, x0 = int(best[0]) * S, int(best[1]) * S
    else:
        y0, x0 = Y0, X0
    print("patch at x=%d y=%d on frame %d" % (x0, y0, fid))
    sl = (slice(y0, y0 + S), slice(x0, x0 + S))

    dark = cv2.imread(str(sorted(DARKD.glob("*.tiff"), key=natural)[250]), cv2.IMREAD_UNCHANGED)
    sig  = cv2.imread(str(sorted(SIGD.glob("*.tiff"), key=natural)[400]), cv2.IMREAD_UNCHANGED)
    mean, sigma = read_bgm(a.runs / "N500/bg.bgm")

    # 1 dark frame, 2 model (sigma), 3 threshold mask, 4 clusters
    save(dark[sl], a.out / "s1_dark.png", "magma", vmax=6)
    save(sigma[sl], a.out / "s2_model.png", "viridis", vmax=1.2)
    mask = (sig > mean + 5 * np.maximum(sigma, 1)).astype(np.uint8)[sl]
    save(mask, a.out / "s3_threshold.png", "gray_r", vmax=1)
    # Stage 4 zooms in instead of showing the same wide patch: connected
    # components are about what happens around ONE track, and at the density of
    # this run a 128x128 window holds only a handful of single pixels.
    big = det[(det.image_id == fid)].nlargest(1, "size_px").iloc[0]
    W = 24
    cx0 = int(np.clip(big.center_x - W // 2, 0, sig.shape[1] - W))
    cy0 = int(np.clip(big.center_y - W // 2, 0, sig.shape[0] - W))
    zsl = (slice(cy0, cy0 + W), slice(cx0, cx0 + W))
    zmask = (sig > mean + 5 * np.maximum(sigma, 1)).astype(np.uint8)[zsl]
    nlab, lab = cv2.connectedComponents(zmask, connectivity=8)
    rgb = np.zeros(lab.shape + (3,), np.uint8)
    pal = np.array([[228, 26, 28], [55, 126, 184], [77, 175, 74], [255, 127, 0],
                    [152, 78, 163], [255, 255, 51], [166, 86, 40], [247, 129, 191]], np.uint8)
    for k in range(1, nlab):
        rgb[lab == k] = pal[(k - 1) % len(pal)]
    fig = plt.figure(figsize=(1.6, 1.6), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(rgb, interpolation="nearest")
    fig.savefig(a.out / "s4_clusters.png"); plt.close(fig)
    print("stage 4 zoom: %dx%d around a %d-pixel cluster, %d components"
          % (W, W, int(big.size_px), nlab - 1))
    print("patch: %d mask pixels, %d components" % (mask.sum(), nlab - 1))

    # 5 measurements: the box beside this thumbnail promises twenty columns per
    # cluster, so a scatter of two of them was the wrong picture. Parallel
    # coordinates show many columns at once, and at thumbnail size the eye reads
    # "a table with a lot of columns", which is exactly the step.
    COLS = ["size_px", "aspect_ratio", "elongation", "compactness",
            "total_charge", "peak_value", "peak_snr", "sigma_x", "sigma_y"]
    d = pd.read_csv(a.runs / "signal1500/detections_classified.csv",
                    usecols=COLS).sample(45, random_state=0)
    # each column scaled to its own range, which is what makes the lines
    # comparable rather than dominated by total_charge
    v = d[COLS].to_numpy(dtype=float)
    lo, hi = v.min(0), v.max(0)
    v = (v - lo) / np.where(hi - lo == 0, 1, hi - lo)
    fig = plt.figure(figsize=(1.6, 1.6), dpi=110)
    ax = fig.add_axes([.02, .02, .96, .96]); ax.axis("off")
    xs = np.arange(len(COLS))
    # les axes d'abord, en trait franc : c'est eux qui doivent se lire a la
    # taille d'une vignette, les donnees ne sont qu'une texture par-dessus
    for x in xs:
        ax.axvline(x, c="#5f6368", lw=1.0, alpha=.9, zorder=1)
    for row in v:
        ax.plot(xs, row, c="#2b6cb0", alpha=.30, lw=.8, zorder=2)
    ax.set_xlim(-0.3, len(COLS) - 0.7); ax.set_ylim(-0.05, 1.05)
    fig.savefig(a.out / "s5_measure.png"); plt.close(fig)

    # 6 hand labels: one mean tile per class, from the class plate's own recipe.
    # The plate's pixel geometry is not a constant of this script -- it follows
    # ZOOM and the common window make_class_plate.py picks at run time -- so the
    # tiles are found from the 1-px frames the plate draws round them. The
    # offsets that used to be written in here belonged to a plate two versions
    # old and cropped the gap between two rows.
    plate = cv2.imread(str(Path(__file__).resolve().parent / "figures_results/fig10_class_plate.png"),
                       cv2.IMREAD_GRAYSCALE)
    if plate is not None:
        frame = plate == 150
        xb = np.flatnonzero(frame.sum(0) > .5 * plate.shape[0])   # 2 per column
        yb = np.flatnonzero(frame.sum(1) > .5 * plate.shape[1])   # 2 per row
        tiles = [plate[yb[2 * j] + 1:yb[2 * j + 1], xb[-2] + 1:xb[-1]]
                 for j in range(len(yb) // 2)]
        strip = np.concatenate([cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=246)
                                for t in tiles], axis=1)
        cv2.imwrite(str(a.out / "s6_labels.png"), strip)

    # 7 classifier: the class balance it returns
    vc = pd.read_csv(a.runs / "signal1500/detections_classified.csv",
                     usecols=["predicted_label"]).predicted_label.value_counts()
    fig = plt.figure(figsize=(1.6, 1.6), dpi=110); ax = fig.add_axes([.05, .05, .9, .9]); ax.axis("off")
    order = ["artifact", "gamma", "lithium"]
    vals = [int(vc.get(k, 0)) for k in order]
    # dataflow_figure.tex posts this thumbnail NODE_PT wide (a 17 mm node in a
    # picture resized to \textwidth), so the 1.6 in figure is reduced by 0.365
    # on the page and the 6.5 pt these labels used to carry printed at 2.4 pt.
    # The floor is the 9 pt of the \figsource line. Widen the node in
    # dataflow_figure.tex and this follows on its own.
    # At 24.6 pt the longest name is wider than the shortest bar, so the labels
    # are dark with a white halo rather than white inside the bar: they stay
    # readable where they run past the bar end. The bars are taller to match.
    NODE_PT  = 42.1                       # measured in main.pdf, page 19
    LABEL_PT = 9.0 * (1.6 * 72) / NODE_PT
    ax.barh(range(3), np.log10(np.maximum(vals, 1)),
            color=["#7f8c8d", "#f39c12", "#c0392b"], height=.86)
    ax.set_xlim(0, np.log10(max(vals)) * 1.05); ax.set_ylim(-.7, 2.7)
    for i, v in enumerate(vals):
        ax.text(.05, i, order[i], va="center", ha="left", fontsize=LABEL_PT,
                color="#111111",
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
    fig.savefig(a.out / "s7_classify.png"); plt.close(fig)

    # 8 analysis: the arrival map, the end product
    arr = cv2.imread(str(a.runs / "signal1500/signal_arrival.tiff"), cv2.IMREAD_UNCHANGED)
    if arr is not None:
        B = 48
        R, C = arr.shape
        blk = arr[:R // B * B, :C // B * B].reshape(R // B, B, C // B, B).sum((1, 3))
        save(blk, a.out / "s8_analysis.png", "magma", vmax=np.percentile(blk, 99))
    print("wrote thumbnails to", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
