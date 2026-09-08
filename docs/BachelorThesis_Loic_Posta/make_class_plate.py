#!/usr/bin/env python3
"""Figure 3.7 -- the three classes as the classifier assigns them on the beam-on
run, in the manner of Filter (2018) figure 8.14.

One row per class: seven specimens, one drawn at random from each seventh of that
class's size distribution so the row spans the range instead of repeating its
commonest case, then the mean image of the class over up to 400 members.

Every tile is drawn at ONE magnification, so a larger object really is larger;
the zoom is chosen so the biggest specimen fills its tile. Each specimen keeps
its own tight crop -- the bounding box plus a single pixel of border -- so the
tiles are deliberately not all the same size. The mean is the one exception: an
average needs a common frame, so it is taken on a fixed window.

    python3 make_class_plate.py --runs ../../Code/runs_2026-08-25
"""
import argparse, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2

CODE = Path(__file__).resolve().parents[2] / "Code"
CLS  = ["lithium", "gamma", "artifact"]
# ZOOM is how many sheet pixels one sensor pixel gets. It sets the resolution
# of the plate, not its printed size: main.tex posts the sheet at .88\linewidth
# whatever its pixel width, and one sensor pixel still prints 5.0 pt wide at
# ZOOM 22 as it did at ZOOM 11. Doubling it takes the plate from 158 to 317 dpi
# on the page and lets the two labels be drawn at a size that survives the
# reduction -- the row label prints at 10.1 pt and the count at 9.1 pt, against
# the 9 pt of the \figsource line.
NSPEC, MEANMAX, PAD, ZOOM = 7, 400, 1, 22
GAP, LEFT, TOPLAB = 10, 210, 20
LAB_SCALE, CNT_SCALE, LAB_THICK = 1.55, 1.40, 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--signal", type=Path,
                    default=CODE / "Data_tot/signal/2026-07-11_3-002_575-6ms_2fps_first_UCN")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures_results")
    a = ap.parse_args()
    run, sig, out = a.runs / "signal1500", a.signal, a.out
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)

    cls_csv = run / "detections_classified.csv"
    if not cls_csv.is_file():
        print("classifying the beam-on run ...", flush=True)
        subprocess.run([sys.executable, str(CODE / "src/ml_classifier.py"), "--predict",
                        "--input", str(run / "detections.csv"), "--output", str(cls_csv),
                        "--bundle", str(CODE / "model_bundle.joblib")],
                       cwd=str(CODE), check=True, capture_output=True)
    d = pd.read_csv(cls_csv)
    print("clusters:", len(d), dict(d.predicted_label.value_counts()))

    files = sorted(sig.glob("*.tiff"),
                   key=lambda p: [int(t) if t.isdigit() else t.lower()
                                  for t in re.split(r"(\d+)", p.name)])
    off, cache = int(d.image_id.min()), {}

    def frame(image_id):
        i = int(image_id) - off
        if not (0 <= i < len(files)):
            return None
        if i not in cache:
            if len(cache) > 40:
                cache.clear()
            cache[i] = cv2.imread(str(files[i]), cv2.IMREAD_UNCHANGED)
        return cache[i]

    def crop_tight(r):
        im = frame(r.image_id)
        if im is None:
            return None
        x0, y0 = max(0, int(r.bbox_x) - PAD), max(0, int(r.bbox_y) - PAD)
        x1 = min(im.shape[1], int(r.bbox_x + r.bbox_w) + PAD)
        y1 = min(im.shape[0], int(r.bbox_y + r.bbox_h) + PAD)
        c = im[y0:y1, x0:x1]
        return c if c.size else None

    def crop_fixed(r, w):
        im = frame(r.image_id)
        if im is None:
            return None
        h = w // 2
        x, y = int(round(r.center_x)), int(round(r.center_y))
        if x - h < 0 or y - h < 0 or x + h + 1 > im.shape[1] or y + h + 1 > im.shape[0]:
            return None
        return im[y - h:y + h + 1, x - h:x + h + 1].astype(np.float64)

    # One window for the whole plate, wide enough that the largest cluster kept
    # has a single pixel of border. Every tile is then the same size and every
    # black pixel inside one is real sensor, not padding -- so a small cluster
    # looks small because it is, and the sizes can be compared across rows.
    picked = {}
    for cl in CLS:
        sub = d[d.predicted_label == cl]
        if sub.empty:
            continue
        qs = np.quantile(sub.size_px, np.linspace(0, 1, NSPEC + 1))
        rows_ = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            bin_ = sub[(sub.size_px >= lo) & (sub.size_px <= hi)]
            if bin_.empty:
                bin_ = sub
            rows_.append(bin_.iloc[rng.integers(len(bin_))])
        picked[cl] = (sub, rows_)
    W = int(max(max(r.bbox_w, r.bbox_h) for _, rs in picked.values() for r in rs)) + 2 * PAD
    W += (W + 1) % 2                      # odd, so a window can be centred
    print("largest kept bounding box -> common window %d x %d px" % (W, W))

    rows = []
    for cl, (sub, chosen) in picked.items():
        spec = [c for c in (crop_fixed(r, W) for r in chosen) if c is not None]
        samp = sub.sample(min(MEANMAX, len(sub)), random_state=1).sort_values("image_id")
        acc, k = None, 0
        for _, r in samp.iterrows():
            c = crop_fixed(r, W)
            if c is None:
                continue
            acc = c if acc is None else acc + c
            k += 1
        rows.append((cl, len(sub), spec, (acc / k) if k else np.zeros((W, W)), k))
        print("  %-9s n=%7d  specimens=%d  mean over %d" % (cl, len(sub), len(spec), k))

    def norm(x):
        x = x.astype(np.float64)
        lo, hi = x.min(), x.max()
        return np.zeros_like(x, np.uint8) if hi <= lo else ((x - lo) / (hi - lo) * 255).astype(np.uint8)

    TILE = W * ZOOM
    tiles = [[cv2.resize(norm(img), (TILE, TILE), interpolation=cv2.INTER_NEAREST)
              for img in (spec + [mimg])] for _, _, spec, mimg, _ in rows]

    LEFT2, GAP2 = 380, 16
    sheet = np.full((GAP2 + len(rows) * (TILE + GAP2) + GAP2,
                     LEFT2 + 8 * (TILE + GAP2) + GAP2), 246, np.uint8)
    for j, (cl, n, _, _, _) in enumerate(rows):
        y = GAP2 + j * (TILE + GAP2)
        cv2.putText(sheet, cl, (20, y + TILE // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    LAB_SCALE, 40, LAB_THICK, cv2.LINE_AA)
        cv2.putText(sheet, "n = %d" % n, (20, y + TILE // 2 + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, CNT_SCALE, 110, LAB_THICK, cv2.LINE_AA)
        for i, t in enumerate(tiles[j]):
            x = LEFT2 + i * (TILE + GAP2)
            sheet[y:y + TILE, x:x + TILE] = t
            cv2.rectangle(sheet, (x - 1, y - 1), (x + TILE, y + TILE), 150, 1)
    cv2.imwrite(str(out / "fig10_class_plate.png"), sheet)
    print("wrote", out / "fig10_class_plate.png", sheet.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
