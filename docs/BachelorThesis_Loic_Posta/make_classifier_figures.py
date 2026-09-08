#!/usr/bin/env python3
"""Figures 4.x -- classifier stability and learning curve, plus the numbers the
false-positive section quotes.

  fig8_classifier_stability.png  the counterpart of Filter (2018) fig. 8.17:
                                 1000 independent retrainings, histogrammed
  fig9_learning_curve.png        cross-validated F1 against the number of labels

It also prints the confusion matrix and the beam-off floor, which are quoted in
the text rather than plotted.

    python3 make_classifier_figures.py [--runs ../../Code/runs_2026-08-25]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, learning_curve,
                                     StratifiedKFold, cross_val_predict)
from sklearn.metrics import confusion_matrix, classification_report

CODE  = Path(__file__).resolve().parents[2] / "Code"
FEATS = ["size_px", "aspect_ratio", "elongation", "compactness", "total_charge",
         "peak_value", "peak_snr", "sigma_x", "sigma_y", "axis_ratio"]
N_RUNS = 1000
SIZES  = [6, 10, 16, 24, 40, 60, 80, 110, 140, 160]
# the trained model's own hyper-parameters, so the study describes the shipped
# classifier and not a different one
FOREST = dict(n_estimators=300, max_depth=None, min_samples_leaf=2,
              class_weight="balanced", n_jobs=-1)


def add_axis_ratio(d):
    d["axis_ratio"] = (d.sigma_x / d.sigma_y.replace(0, np.nan)).fillna(1.0)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=CODE / "detections_annotation_sample.csv")
    ap.add_argument("--dark", type=Path, default=CODE / "dark_pass_tot_classified.csv")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures_results")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)

    d = add_axis_ratio(pd.read_csv(a.labels))
    X, y_all = d[FEATS].to_numpy(), d.true_label.to_numpy()
    y = (y_all == "lithium")
    print("labelled clusters:", len(d), dict(pd.Series(y_all).value_counts()))
    print("lithium prevalence %.1f%%  -> majority baseline accuracy %.1f%%"
          % (100 * y.mean(), 100 * (1 - y.mean())))

    # ── stability under repeated retraining ────────────────────────────────
    acc, fp = [], []
    for seed in range(N_RUNS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        p = RandomForestClassifier(random_state=seed, **FOREST).fit(Xtr, ytr).predict(Xte)
        acc.append(((p & yte).sum() + (~p & ~yte).sum()) / len(yte))
        fp.append((p & ~yte).sum() / len(yte))
    acc, fp = np.array(acc), np.array(fp)
    print("accuracy %.4f +- %.4f | FP fraction %.4f +- %.4f"
          % (acc.mean(), acc.std(), fp.mean(), fp.std()))

    # main.tex places this at .72\linewidth of a 16 cm text block = 4.54 in,
    # so drawing it 4.40 in wide makes LaTeX enlarge it by 1.03 rather than
    # shrink it by 0.71: the 9 pt box text lands at 9.3 pt on the page, just
    # above the 9 pt of the \figsource line.
    fig, ax = plt.subplots(2, 1, figsize=(4.40, 4.20))
    for A, v, lab, ttl in [(ax[0], acc, "Accuracy", "(a)"),
                           (ax[1], fp, "False positives", "(b)")]:
        A.hist(v, bins=20, color="0.45", edgecolor="white", linewidth=0.6)
        A.set_xlabel(lab); A.set_ylabel("Counts"); A.set_title(ttl, fontsize=10, loc="left")
        A.text(0.03, 0.95, "Mean: %.2f%%\nSTD: %.2f%%" % (100 * v.mean(), 100 * v.std()),
               transform=A.transAxes, va="top", fontsize=9,
               bbox=dict(boxstyle="square", fc="0.94", ec="0.6"))
        # The box holds its size while the axes shrank to the printed width, so
        # it needs headroom of its own; without it the tallest bar of (b) hides
        # behind it.
        A.set_ylim(0, A.get_ylim()[1] * 1.22)
        A.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(a.out / "fig8_classifier_stability.png", dpi=300)
    plt.close(fig)

    # ── how much would more labels buy? ────────────────────────────────────
    sizes, _, te = learning_curve(
        RandomForestClassifier(random_state=0, **FOREST), X, y_all, train_sizes=SIZES,
        cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="f1_macro")
    m, s = te.mean(1), te.std(1)
    # .70\linewidth = 4.41 in on the page; drawn 4.28 in wide, so LaTeX
    # enlarges by 1.03 and the 10 pt axis text prints at 10.3 pt.
    fig, A = plt.subplots(figsize=(4.28, 2.60))
    A.errorbar(sizes, m, yerr=s, fmt="o", color="#2b6cb0", ms=5.5, capsize=3.5,
               elinewidth=1.1, markeredgecolor="white", markeredgewidth=0.6)
    A.axhline(m[-1], color="0.55", lw=0.9, ls=":", zorder=0)
    A.set_xlabel("labelled clusters used for training")
    A.set_ylabel("cross-validated $F_1$ (macro)")
    A.set_xlim(0, max(SIZES) + 12); A.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(a.out / "fig9_learning_curve.png", dpi=300)
    plt.close(fig)
    for n, v, e in zip(sizes, m, s):
        print("  train %3d -> F1 %.3f +- %.3f" % (n, v, e))

    # ── where the error actually is ────────────────────────────────────────
    yp = cross_val_predict(RandomForestClassifier(random_state=42, **FOREST), X, y_all,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0))
    lab = sorted(set(y_all))
    print("\nconfusion (rows = truth):")
    print(pd.DataFrame(confusion_matrix(y_all, yp, labels=lab), index=lab, columns=lab).to_string())
    print(classification_report(y_all, yp, labels=lab, digits=3))

    # ── the beam-off floor ─────────────────────────────────────────────────
    if a.dark.is_file():
        k = pd.read_csv(a.dark, usecols=["image_id", "size_px", "predicted_label"])
        n = k.image_id.nunique()
        print("beam-off: %d clusters over %d frames = %.1f/frame"
              % (len(k), n, len(k) / n))
        for c, v in k.predicted_label.value_counts().items():
            print("   %-9s %8d  %8.3f/frame" % (c, v, v / n))
        for lo in (3, 5, 10):
            print("   size_px >= %2d : %d" % (lo, int((k.size_px >= lo).sum())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
