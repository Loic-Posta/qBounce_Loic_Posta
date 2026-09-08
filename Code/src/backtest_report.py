"""
backtest_report.py — Sprint 4 model backtesting for the CMOS Particle Pipeline
================================================================================
Evaluates the current ML classifier (model_bundle.joblib) against a manually
verified "golden" dataset: a CSV carrying a `true_label` column that a human
has checked and is 100% sure of. Reports global accuracy, per-class
precision/recall/F1, and a confusion matrix — all printed to stdout (so
PipelineController's Console tab captures it verbatim) and optionally written
to a report file.

Meant to be called by PipelineController._step_backtest() (Step 7), but is a
plain standalone CLI tool otherwise.

Usage
-----
Recommended — run the bundle directly on the golden CSV's own feature columns
to (re)generate predictions, then score them against true_label. This is the
only mode that doesn't depend on some other file being row-aligned with the
golden set, so it's what PipelineController's Step 7 uses:

    python backtest_report.py --golden golden_detections.csv \
                               --bundle model_bundle.joblib \
                               --output backtest_report.txt

Alternative — reuse predictions already produced by Step 6
(detections_classified.csv), IF that file is row-aligned 1:1 with the golden
CSV (same rows, same order — e.g. the golden set. This restriction is because
detections_classified.csv is not guaranteed to be in the same order/subset as
whatever golden_detections.csv contains):

    python backtest_report.py --input detections_classified.csv \
                               --golden golden_detections.csv \
                               --output backtest_report.txt

Output
------
Always printed to stdout. If --output is given, the same text is written to
disk — as plain text, or as a minimal standalone HTML page if the path ends
in .html/.htm.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ml_classifier.py lives next to this script (Code/src/) — make sure it's
# importable regardless of the caller's cwd (PipelineController runs this
# with cwd=SRC_DIR, but keep this robust for standalone use too).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_classifier import add_derived_features, load_bundle, FEATURE_COLS  # noqa: E402

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



# ─── Data loading / alignment ──────────────────────────────────────────────

def load_golden(golden_csv: str) -> pd.DataFrame:
    """Load the golden CSV, keep only rows with a non-empty true_label."""
    df = pd.read_csv(golden_csv)
    if "true_label" not in df.columns:
        raise ValueError(f"{golden_csv} has no 'true_label' column.")
    df = df.copy()
    df["true_label"] = df["true_label"].fillna("").astype(str).str.strip()
    df = df[df["true_label"] != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{golden_csv} has no non-empty true_label rows.")
    return df


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same derived-feature step ml_classifier.load_features() applies,
    reused here so a golden CSV of raw detector columns can be fed straight
    into the model bundle."""
    df = add_derived_features(df)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Golden CSV is missing feature columns needed by the model: {missing}"
        )
    return df


def predict_on_golden(golden_df: pd.DataFrame, bundle_path: str) -> tuple[pd.DataFrame, dict]:
    """Run the model bundle directly on the golden CSV's feature columns."""
    bundle = load_bundle(bundle_path)
    clf, le = bundle["model"], bundle["label_encoder"]

    feat_df = _prepare_features(golden_df)
    X = feat_df[bundle["feature_cols"]].values

    proba = clf.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)
    pred_label = le.inverse_transform(pred_idx)
    confidence = proba[np.arange(len(pred_idx)), pred_idx]

    out = golden_df.copy()
    out["predicted_label"] = pred_label
    out["confidence"] = confidence

    model_info = {
        "trained_at": bundle.get("trained_at"),
        "n_samples": bundle.get("n_samples"),
        "cv_f1_macro_mean": bundle.get("cv_f1_macro_mean"),
    }
    return out, model_info


def align_from_input(golden_df: pd.DataFrame, input_csv: str) -> pd.DataFrame:
    """Reuse an already-classified CSV's predicted_label column instead of
    re-running inference — only valid if it's row-aligned 1:1 with golden_df."""
    input_df = pd.read_csv(input_csv)
    if len(input_df) != len(golden_df):
        raise ValueError(
            f"--input has {len(input_df)} rows but --golden has {len(golden_df)} "
            f"(after dropping empty true_label rows) — they must be row-aligned "
            f"1:1 to reuse predictions this way. Drop --input and pass --bundle "
            f"instead to run the model directly on the golden CSV's own features."
        )
    if "predicted_label" not in input_df.columns:
        raise ValueError(f"{input_csv} has no 'predicted_label' column.")

    out = golden_df.copy()
    out["predicted_label"] = input_df["predicted_label"].values
    if "confidence" in input_df.columns:
        out["confidence"] = input_df["confidence"].values
    return out


# ─── Report building ────────────────────────────────────────────────────────

def build_report(scored_df: pd.DataFrame, model_info: Optional[dict] = None) -> str:
    y_true = scored_df["true_label"].astype(str)
    y_pred = scored_df["predicted_label"].astype(str)
    n = len(scored_df)

    labels = sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("BACKTEST REPORT — qBOUNCE ML Classifier vs Golden Dataset")
    lines.append("=" * 72)
    lines.append(f"Generated:        {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    if model_info:
        trained_at = model_info.get("trained_at") or "unknown"
        lines.append(f"Model trained at: {trained_at}")
        n_samples = model_info.get("n_samples")
        if n_samples is not None:
            lines.append(f"Model train size: {n_samples:,} samples")
        cv_f1 = model_info.get("cv_f1_macro_mean")
        if cv_f1 is not None:
            lines.append(f"Model CV F1 (macro, on training data): {cv_f1:.3f}")
    lines.append(f"Golden set size:  {n:,} rows")
    lines.append("")
    n_correct = int(round(acc * n))
    lines.append(f"GLOBAL ACCURACY:  {acc:.4f}   ({n_correct:,}/{n:,} correct)")
    lines.append("")
    lines.append("Per-class precision / recall / F1")
    lines.append("-" * 72)
    lines.append(f"{'class':<18}{'precision':>12}{'recall':>12}{'f1':>10}{'support':>10}")
    for lbl, p, r, f, s in zip(labels, precision, recall, f1, support):
        lines.append(f"{lbl:<18}{p:>12.3f}{r:>12.3f}{f:>10.3f}{s:>10d}")
    lines.append("-" * 72)
    lines.append(f"{'macro avg':<18}{macro_p:>12.3f}{macro_r:>12.3f}{macro_f:>10.3f}{n:>10d}")
    lines.append("")
    lines.append("Confusion matrix (rows = true label, columns = predicted label)")
    lines.append("-" * 72)
    col_w = max(9, max(len(lbl) for lbl in labels) + 2)
    header = " " * col_w + "".join(f"{lbl[:col_w - 1]:>{col_w}}" for lbl in labels)
    lines.append(header)
    for i, lbl in enumerate(labels):
        row_str = f"{lbl:<{col_w}}" + "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(labels)))
        lines.append(row_str)
    lines.append("=" * 72)
    return "\n".join(lines)


def build_html_report(text_report: str) -> str:
    escaped = (
        text_report.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>qBOUNCE Backtest Report</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; background: #f5f5f7;
          margin: 0; padding: 24px; }}
  pre  {{ background: #ffffff; border: 1px solid #d2d2d7; border-radius: 8px;
          padding: 20px; font-family: "SF Mono", Menlo, Consolas, monospace;
          font-size: 13px; line-height: 1.5; overflow-x: auto; }}
</style>
</head>
<body>
<pre>{escaped}</pre>
</body>
</html>
"""


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the ML classifier against a manually verified golden dataset."
    )
    parser.add_argument("--golden", required=True,
                         help="Path to golden_detections.csv (must have a true_label column).")
    parser.add_argument("--input", default=None,
                         help="Optional: an already-classified CSV (predicted_label column) "
                              "that is row-aligned 1:1 with --golden. If omitted (recommended), "
                              "the model bundle is run directly on --golden's own features.")
    parser.add_argument("--bundle", default="model_bundle.joblib",
                         help="Path to model_bundle.joblib (ignored if --input already "
                              "supplies predicted_label).")
    parser.add_argument("--output", default=None,
                         help="Path to write the report (.txt or .html). Always printed "
                              "to stdout regardless.")
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.is_file():
        print(f"[backtest] ERROR: golden dataset not found: {golden_path}", file=sys.stderr)
        sys.exit(1)

    try:
        golden_df = load_golden(str(golden_path))
    except Exception as exc:
        print(f"[backtest] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    model_info: Optional[dict] = None
    try:
        if args.input:
            print(f"[backtest] Reusing predictions from {Path(args.input).name} "
                  f"(row-aligned mode)…")
            scored_df = align_from_input(golden_df, args.input)
        else:
            print(f"[backtest] Running model bundle '{args.bundle}' directly on "
                  f"{golden_path.name}'s features…")
            scored_df, model_info = predict_on_golden(golden_df, args.bundle)
    except Exception as exc:
        print(f"[backtest] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    report = build_report(scored_df, model_info)
    print()
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() in (".html", ".htm"):
            out_path.write_text(build_html_report(report), encoding="utf-8")
        else:
            out_path.write_text(report, encoding="utf-8")
        print(f"\n[backtest] Report written to {out_path}")


if __name__ == "__main__":
    main()