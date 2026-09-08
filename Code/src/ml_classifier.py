"""
ml_classifier.py  –  Level 1 ML : cluster classification
=========================================================
Input  : detections.csv  produced by the C++ detector
Output : detections_classified.csv  with added  predicted_label / confidence
         columns (batch mode) — or a single dict (live / single-frame mode)

Dependencies:
    pip install pandas scikit-learn imbalanced-learn matplotlib seaborn joblib

Usage
-----
# Step 1 – label a subset manually (open detections.csv, add a 'true_label'
#           column with values matching whatever categories you're training,
#           e.g. particle | artifact | cosmic_ray | hot_pixel)
# Step 2 – train
    python ml_classifier.py --train --labeled detections_labeled.csv

# Step 3 – predict on new data
    python ml_classifier.py --predict --input detections.csv

Model storage
-------------
Everything the classifier needs to run — the fitted model, the label
encoder, the feature column order, and provenance about how/when it was
trained — is saved together in ONE file:

    model_bundle.joblib

This replaces the old two-file layout (rf_classifier.joblib +
label_encoder.joblib). If this script is asked to load a bundle that
doesn't exist yet, but the old-format pair is sitting next to it, it
transparently migrates them into a bundle so nothing breaks for a model
trained before this change.

Live / single-frame use (PipelineController "Live Acquisition" mode)
----------------------------------------------------------------------
    from ml_classifier import predict_single, load_bundle, bundle_classes

    bundle_classes()                        # -> e.g. ["artifact", "cosmic_ray", "hot_pixel", "particle"]
    result = predict_single(feature_dict)   # -> {"label": ..., "confidence": ..., "proba": {...}}

predict_single() is intended to be called repeatedly from a single
background worker thread (see live_detection_engine.py). The bundle is
loaded once per path and cached behind a lock; predict_proba() on an
already-fitted estimator does not mutate it, so that's the only part
that needs the lock — inference itself is safe to call as often as
needed afterwards.

Architecture
------------
We use a Random Forest on the 10-dimensional feature vector extracted by
the C++ ClusterDetector. Random Forests are:
  - Robust to class imbalance (particles << artefacts in early runs)
  - Interpretable (feature importances)
  - Fast to train / infer on CPU
  - A good baseline before moving to a CNN (Level 2 / 3)
"""

from __future__ import annotations
import sys

import argparse
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageDraw, ImageFont

# On Windows, sys.stdout defaults to the console's legacy code page (cp1252 on a
# European install), and every print() below that carries an arrow, a degree
# sign or an accent then dies with UnicodeEncodeError. That is not cosmetic: the
# CSV is written before the plots, so the script exits 1 having produced the data
# and none of the figures. Force UTF-8 on the two streams, and fall back to
# replacing the offending character rather than crashing.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass



# ─── Feature columns (must match Cluster::toFeatureVector() order) ────────────
FEATURE_COLS = [
    "size_px",
    "aspect_ratio",
    "elongation",
    "compactness",
    "total_charge",
    "peak_value",
    "peak_snr",
    "sigma_x",
    "sigma_y",
    "axis_ratio",   # = sigma_x / sigma_y  – derived column added below
]

BUNDLE_VERSION = 1


def _default_font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _magma_like(values: np.ndarray) -> np.ndarray:
    """Small dependency-free black/purple/orange/yellow colormap."""
    stops = np.array([
        [0, 0, 4],
        [45, 17, 95],
        [120, 28, 109],
        [198, 64, 73],
        [249, 142, 8],
        [252, 253, 191],
    ], dtype=np.float32)
    values = np.clip(values, 0.0, 1.0)
    scaled = values * (len(stops) - 1)
    lo = np.floor(scaled).astype(int)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = (scaled - lo)[..., None]
    return (stops[lo] * (1.0 - frac) + stops[hi] * frac).astype(np.uint8)


def save_prediction_counts_plot(df: pd.DataFrame,
                                out_path: str = "particle_counts_by_type.png") -> None:
    """Save a direct count of detected clusters by predicted class."""
    counts = df["predicted_label"].value_counts().sort_index()
    width, height = 760, 460
    margin_l, margin_r, margin_t, margin_b = 80, 30, 58, 88
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)
    font = _default_font()
    draw.text((margin_l, 20), "Detected particles by predicted type",
              fill="#111111", font=font)

    labels = [str(k) for k in counts.index]
    values = [int(v) for v in counts.values]
    if not values:
        draw.text((margin_l, height // 2), "No predictions", fill="#666666", font=font)
        img.save(out_path)
        print(f"Prediction count plot saved to {out_path}")
        return

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_val = max(values) or 1
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="#333333")
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h),
              fill="#333333")
    bar_gap = 18
    bar_w = max(18, int((plot_w - bar_gap * (len(values) + 1)) / max(1, len(values))))
    for i, (label, value) in enumerate(zip(labels, values)):
        x0 = margin_l + bar_gap + i * (bar_w + bar_gap)
        x1 = x0 + bar_w
        h = int(plot_h * value / max_val)
        y0 = margin_t + plot_h - h
        y1 = margin_t + plot_h
        draw.rectangle((x0, y0, x1, y1), fill="#0071e3")
        draw.text((x0, max(margin_t, y0 - 16)), f"{value:,}", fill="#111111", font=font)
        draw.text((x0, y1 + 10), label[:18], fill="#111111", font=font)
    draw.text((12, margin_t + plot_h // 2), "Number", fill="#333333", font=font)
    img.save(out_path)
    print(f"Prediction count plot saved to {out_path}")


def save_ml_arrival_counts_plot(df: pd.DataFrame,
                                out_path: str = "ml_signal_arrival_counts.png") -> None:
    """Save a spatial arrival-density map from ML-classified detections.

    The C++ signal_arrival heatmap counts every detector cluster before the
    classifier has filtered it. This plot counts the centres of clusters after
    ML classification, preferring the "particle" class when it exists.
    """
    required = {"center_x", "center_y", "predicted_label"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        print(f"Skipping ML arrival count plot; missing columns: {missing}")
        return

    labels = df["predicted_label"].astype(str)
    if (labels == "particle").any():
        selected = df[labels == "particle"].copy()
        label_note = "predicted particle"
    else:
        selected = df.copy()
        label_note = "all predicted classes"

    x = pd.to_numeric(selected["center_x"], errors="coerce")
    y = pd.to_numeric(selected["center_y"], errors="coerce")
    valid = x.notna() & y.notna()
    x = x[valid].to_numpy()
    y = y[valid].to_numpy()

    width, height = 900, 560
    margin_l, margin_r, margin_t, margin_b = 70, 95, 58, 56
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)
    font = _default_font()
    if len(x) == 0:
        draw.text((width // 2 - 80, height // 2), "No ML-classified arrivals",
                  fill="#666666", font=font)
    else:
        max_x = pd.to_numeric(df.get("bbox_x", df["center_x"]), errors="coerce")
        max_y = pd.to_numeric(df.get("bbox_y", df["center_y"]), errors="coerce")
        if "bbox_w" in df.columns:
            max_x = max_x + pd.to_numeric(df["bbox_w"], errors="coerce").fillna(0)
        if "bbox_h" in df.columns:
            max_y = max_y + pd.to_numeric(df["bbox_h"], errors="coerce").fillna(0)
        sensor_w = max(1, int(np.ceil(np.nanmax(max_x.to_numpy()))))
        sensor_h = max(1, int(np.ceil(np.nanmax(max_y.to_numpy()))))
        plot_w = width - margin_l - margin_r
        plot_h = height - margin_t - margin_b
        bins_x = min(plot_w, sensor_w)
        bins_y = max(1, int(round(bins_x * sensor_h / sensor_w)))

        hist, x_edges, y_edges = np.histogram2d(
            x, y,
            bins=[bins_x, bins_y],
            range=[[0, sensor_w], [0, sensor_h]],
        )
        hist = hist.T
        max_count = float(hist.max()) or 1.0
        norm = hist / max_count
        heat = Image.fromarray(_magma_like(norm), mode="RGB")
        heat = heat.resize((plot_w, plot_h), Image.Resampling.NEAREST)
        img.paste(heat, (margin_l, margin_t))

        draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + plot_h),
                       outline="#222222")
        draw.text((margin_l, margin_t + plot_h + 24), "Sensor x (px)",
                  fill="#333333", font=font)
        draw.text((8, margin_t + plot_h // 2), "Sensor y", fill="#333333", font=font)
        draw.text((margin_l, margin_t + plot_h + 6), "0", fill="#333333", font=font)
        draw.text((margin_l + plot_w - 44, margin_t + plot_h + 6),
                  f"{sensor_w}", fill="#333333", font=font)
        draw.text((margin_l - 48, margin_t + 2), "0", fill="#333333", font=font)
        draw.text((margin_l - 58, margin_t + plot_h - 12),
                  f"{sensor_h}", fill="#333333", font=font)

        bar_x = width - margin_r + 30
        bar_y = margin_t
        bar_w = 18
        bar_h = plot_h
        grad = np.linspace(1.0, 0.0, bar_h)[:, None]
        grad_rgb = Image.fromarray(_magma_like(np.repeat(grad, bar_w, axis=1)), mode="RGB")
        img.paste(grad_rgb, (bar_x, bar_y))
        draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), outline="#222222")
        draw.text((bar_x + 25, bar_y), f"{int(max_count):,}", fill="#333333", font=font)
        draw.text((bar_x + 25, bar_y + bar_h - 12), "0", fill="#333333", font=font)
        draw.text((bar_x - 6, bar_y + bar_h + 8), "count/bin", fill="#333333", font=font)

    draw.text((margin_l, 20),
              f"Signal arrival counts from ML classifier ({label_note}, n={len(x):,})",
              fill="#111111", font=font)
    img.save(out_path)
    print(f"ML arrival count plot saved to {out_path}")
BUNDLE_PATH = "model_bundle.joblib"
ONNX_PATH = "model_bundle.onnx"

# Old two-file layout this script used to write. Kept only so a model
# trained before the bundle format existed can be migrated automatically
# the first time something tries to load it.
_LEGACY_MODEL_PATH = "rf_classifier.joblib"
_LEGACY_LABEL_ENC_PATH = "label_encoder.joblib"


# ─── Helpers ───────────────────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add any features that are derived from the base CSV columns."""
    df = df.copy()
    df["axis_ratio"] = df["sigma_x"] / (df["sigma_y"] + 1e-6)
    return df


def load_features(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = add_derived_features(df)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


# ─── Bundle load / save / cache ─────────────────────────────────────────────
#
# A bundle is a plain dict:
#   version, model, label_encoder, feature_cols, classes,
#   trained_at, source_csv, n_samples, cv_f1_macro_mean, cv_f1_macro_std,
#   sklearn_version, migrated_from_legacy

_bundle_cache: dict[str, dict[str, Any]] = {}
_bundle_cache_lock = threading.Lock()


def _migrate_legacy_pair(bundle_path: str) -> Optional[dict[str, Any]]:
    """If the old rf_classifier.joblib / label_encoder.joblib pair exists
    next to `bundle_path`, load them and write out a bundle so every
    future load goes through the single-file path from here on."""
    bundle = Path(bundle_path)
    base_dir = bundle.parent if str(bundle.parent) else Path(".")
    legacy_model = base_dir / _LEGACY_MODEL_PATH
    legacy_le = base_dir / _LEGACY_LABEL_ENC_PATH
    if not (legacy_model.is_file() and legacy_le.is_file()):
        return None

    print(f"[ml_classifier] No {bundle_path} found, but legacy "
          f"{_LEGACY_MODEL_PATH} + {_LEGACY_LABEL_ENC_PATH} exist — migrating.")
    clf = joblib.load(legacy_model)
    le = joblib.load(legacy_le)
    bundle = {
        "version": BUNDLE_VERSION,
        "model": clf,
        "label_encoder": le,
        "feature_cols": FEATURE_COLS,
        "classes": list(le.classes_),
        "trained_at": None,          # unknown for migrated models
        "source_csv": None,
        "n_samples": None,
        "cv_f1_macro_mean": None,
        "cv_f1_macro_std": None,
        "sklearn_version": None,
        "migrated_from_legacy": True,
    }
    Path(bundle_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    print(f"[ml_classifier] Migrated legacy model -> {bundle_path}")
    return bundle


def _load_bundle_from_disk(bundle_path: str) -> dict[str, Any]:
    path = Path(bundle_path)
    if path.is_file():
        bundle = joblib.load(path)
        if "feature_cols" not in bundle:  # defensive: not one of our bundles
            raise ValueError(f"{bundle_path} does not look like an ml_classifier bundle.")
        return bundle

    migrated = _migrate_legacy_pair(bundle_path)
    if migrated is not None:
        return migrated

    raise FileNotFoundError(
        f"No model bundle at {bundle_path} and no legacy "
        f"{_LEGACY_MODEL_PATH}/{_LEGACY_LABEL_ENC_PATH} pair to migrate from. "
        f"Run --train first."
    )


def load_bundle(bundle_path: str = BUNDLE_PATH) -> dict[str, Any]:
    """Load (and cache) a model bundle. Thread-safe: concurrent callers
    loading the same path block briefly on the first load, then share
    the cached object. Safe to call from a background worker thread."""
    key = str(bundle_path)
    with _bundle_cache_lock:
        cached = _bundle_cache.get(key)
        if cached is not None:
            return cached
        bundle = _load_bundle_from_disk(key)
        _bundle_cache[key] = bundle
        return bundle


def reload_bundle(bundle_path: str = BUNDLE_PATH) -> dict[str, Any]:
    """Force a fresh load, bypassing the cache — e.g. right after
    re-training, or when the user picks a different bundle in the
    Live Acquisition model-selector dropdown."""
    key = str(bundle_path)
    bundle = _load_bundle_from_disk(key)
    with _bundle_cache_lock:
        _bundle_cache[key] = bundle
    return bundle


def bundle_classes(bundle_path: str = BUNDLE_PATH) -> list[str]:
    """Class labels known to a bundle — used to populate the live-mode
    category checklist / bar chart without hardcoding label names."""
    return list(load_bundle(bundle_path)["classes"])


def bundle_info(bundle_path: str = BUNDLE_PATH) -> dict[str, Any]:
    """Small provenance summary for display in a model-selector dropdown."""
    b = load_bundle(bundle_path)
    return {
        "path": str(bundle_path),
        "classes": b["classes"],
        "trained_at": b.get("trained_at"),
        "n_samples": b.get("n_samples"),
        "cv_f1_macro_mean": b.get("cv_f1_macro_mean"),
        "migrated_from_legacy": b.get("migrated_from_legacy", False),
    }


# ─── Training ────────────────────────────────────────────────────────────────

def train(labeled_csv: str, bundle_path: str = BUNDLE_PATH) -> None:
    df = load_features(labeled_csv)

    if "true_label" not in df.columns:
        raise ValueError("The labeled CSV must have a 'true_label' column.")

    df["true_label"] = df["true_label"].fillna("").astype(str).str.strip()
    df = df[df["true_label"] != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError("The labeled CSV has no non-empty true_label rows.")

    X = df[FEATURE_COLS].values
    y_raw = df["true_label"].values

    # Fraction of training clusters whose peak pixel is clipped at 255. This
    # is the operating-point signature that matters for the classifier: a
    # model trained on 100%-saturated tracks has only ever seen clipped
    # peaks / compressed charge, and will not recognise the SAME particles
    # once you lower the gain to de-saturate. Stored in the .meta.json so the
    # live/predict side can warn when the data it's fed no longer matches the
    # saturation the model was trained at (see LiveDetectionEngine).
    train_saturation = (float((df["peak_value"] >= 254).mean())
                        if "peak_value" in df.columns else None)
    if train_saturation is not None:
        print(f"Training-set saturation (peak>=254): {100*train_saturation:.1f}%")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"Classes: {le.classes_}")
    print(f"Class distribution:\n{pd.Series(y_raw).value_counts()}\n")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",   # handles imbalance automatically
        n_jobs=-1,
        random_state=42,
    )

    # Cross-validation to estimate generalisation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
    print(f"Cross-val F1 (macro): {scores.mean():.3f} ± {scores.std():.3f}\n")

    # Final fit on all data
    clf.fit(X, y)

    # ── Feature importance plot ───────────────────────────────────────────
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS)
    importances.sort_values().plot(kind="barh", figsize=(7, 5))
    plt.title("Feature importances")
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=150)
    plt.close()
    print("Feature importance plot saved to feature_importances.png")

    # ── Confusion matrix on training data (sanity check) ─────────────────
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion matrix (training set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix saved to confusion_matrix.png")
    print("\n" + classification_report(y, y_pred, target_names=le.classes_))

    # ── Save unified bundle ────────────────────────────────────────────────
    bundle = {
        "version": BUNDLE_VERSION,
        "model": clf,
        "label_encoder": le,
        "feature_cols": FEATURE_COLS,
        "classes": list(le.classes_),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_csv": str(Path(labeled_csv).resolve()),
        "n_samples": int(len(df)),
        "cv_f1_macro_mean": float(scores.mean()),
        "cv_f1_macro_std": float(scores.std()),
        "sklearn_version": sklearn.__version__,
        "migrated_from_legacy": False,
    }

    Path(bundle_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    with _bundle_cache_lock:
        _bundle_cache[str(bundle_path)] = bundle   # a just-trained model is
                                                     # immediately usable, no reload
    print(f"Model bundle saved to {bundle_path}")

    export_onnx(clf, n_features=len(FEATURE_COLS), classes=list(le.classes_),
                onnx_path=str(Path(bundle_path).with_suffix(".onnx")),
                train_saturation=train_saturation)


def export_onnx(model, n_features: int, classes: list[str],
                 onnx_path: str = ONNX_PATH,
                 train_saturation: "float | None" = None) -> None:
    """Convert a fitted sklearn classifier to ONNX for fast runtime inference.

    Only the estimator is exported — ONNX has no slot for our bundle's
    extra metadata (label encoder, provenance, etc). `predict_single`'s
    label lookup replicates LabelEncoder.inverse_transform() using the
    `classes` list embedded in the .meta.json sidecar written here, so
    the ONNX file + sidecar together are a drop-in replacement for the
    joblib bundle at inference time.

    zipmap=False keeps the second output as a plain [n_samples, n_classes]
    float array (probabilities in class-index order) instead of a list of
    per-row dicts — much cheaper to consume in a tight batch-inference loop.
    """
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onx = to_onnx(
        model,
        initial_types=initial_type,
        target_opset=17,
        options={id(model): {"zipmap": False}},
    )
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())

    # Sidecar: class order + feature order, so the ONNX side never needs
    # joblib/sklearn installed to know how to interpret the raw arrays.
    meta_path = Path(onnx_path).with_suffix(".meta.json")
    import json
    meta_path.write_text(json.dumps({
        "classes": classes,
        "feature_cols": FEATURE_COLS,
        # Operating-point signature (None for legacy models): the saturation
        # this model was trained at. The runtime warns if live data diverges.
        "train_saturation": train_saturation,
    }, indent=2))
    print(f"ONNX model saved to {onnx_path} (+ {meta_path.name})")

# ─── Prediction — batch CSV (unchanged CLI behaviour) ─────────────────────────

def predict(input_csv: str, output_csv: str = "detections_classified.csv",
            bundle_path: str = BUNDLE_PATH) -> None:
    bundle = load_bundle(bundle_path)
    clf, le = bundle["model"], bundle["label_encoder"]

    df = load_features(input_csv)
    X = df[bundle["feature_cols"]].values

    proba = clf.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)
    pred_label = le.inverse_transform(pred_idx)
    confidence = proba[np.arange(len(pred_idx)), pred_idx]

    df["predicted_label"] = pred_label
    df["confidence"] = confidence

    df.to_csv(output_csv, index=False)
    print(f"Classified {len(df)} clusters → {output_csv}")

    print(df["predicted_label"].value_counts())
    save_prediction_counts_plot(df)
    save_ml_arrival_counts_plot(df)
    n_particles = (df["predicted_label"] == "particle").sum()
    print(f"\nParticle hits: {n_particles}")


# ─── Prediction — single frame (for live_detection_engine.py) ────────────────

def predict_single(features: dict[str, float],
                    bundle_path: str = BUNDLE_PATH) -> dict[str, Any]:
    """Classify ONE cluster's feature vector.

    `features` must contain every raw column FEATURE_COLS needs except the
    derived 'axis_ratio' (computed here): size_px, aspect_ratio, elongation,
    compactness, total_charge, peak_value, peak_snr, sigma_x, sigma_y.

    Returns {"label": str, "confidence": float, "proba": {class: prob, ...}}.
    """
    bundle = load_bundle(bundle_path)
    row = add_derived_features(pd.DataFrame([features]))
    X = row[bundle["feature_cols"]].values

    proba = bundle["model"].predict_proba(X)[0]
    idx = int(np.argmax(proba))
    label = bundle["label_encoder"].inverse_transform([idx])[0]
    return {
        "label": str(label),
        "confidence": float(proba[idx]),
        "proba": {cls: float(p) for cls, p in zip(bundle["classes"], proba)},
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster ML classifier")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--labeled", default="detections_labeled.csv",
                         help="CSV with true_label column (for training)")
    parser.add_argument("--input", default="detections.csv",
                         help="CSV to classify (for prediction)")
    parser.add_argument("--output", default="detections_classified.csv")
    parser.add_argument("--bundle", default=BUNDLE_PATH,
                         help="Model bundle path (default: model_bundle.joblib)")
    args = parser.parse_args()

    if args.train:
        train(args.labeled, bundle_path=args.bundle)
    if args.predict:
        predict(args.input, args.output, bundle_path=args.bundle)
    if not args.train and not args.predict:
        parser.print_help()


if __name__ == "__main__":
    main()
