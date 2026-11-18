"""Audit collected feature CSVs and write a JSON report without altering data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml import FEATURES, GESTURES, EYE_STATES


def validate(path, kind):
    expected = GESTURES if kind == "gesture" else EYE_STATES
    feature_cols = [f"f{i}" for i in range(FEATURES[kind])]
    df = pd.read_csv(path)
    needed = ["timestamp", "session_id", "label", *feature_cols]
    absent = [col for col in needed if col not in df.columns]
    if absent: raise ValueError(f"Required CSV columns are missing: {absent}")
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    nonfinite = ~np.isfinite(numeric.to_numpy(dtype=float))
    label_counts = df.label.astype(str).value_counts().to_dict()
    missing_classes = sorted(set(expected)-set(label_counts))
    unknown_labels = sorted(set(df.label.dropna().astype(str))-set(expected))
    duplicate_features = int(df.duplicated(subset=feature_cols).sum())
    grouped = df.groupby("session_id", dropna=False).size()
    arr = numeric.to_numpy(dtype=float)
    finite_arr = arr[np.isfinite(arr)]
    report = {"kind": kind, "source": str(Path(path).resolve()), "rows": len(df), "feature_dimensions": len(feature_cols),
              "missing_values_by_column": {str(k): int(v) for k,v in df[needed].isna().sum().items() if v},
              "non_numeric_or_nonfinite_feature_values": int(nonfinite.sum()),
              "unknown_labels": unknown_labels, "missing_classes": missing_classes,
              "class_distribution": {label: int(label_counts.get(label,0)) for label in expected},
              "session_count": int(df.session_id.nunique(dropna=False)),
              "samples_per_session": {str(k): int(v) for k,v in grouped.items()},
              "duplicate_feature_rows": duplicate_features,
              "outlier_values_abs_gt_10": int((np.abs(finite_arr)>10).sum()) if finite_arr.size else 0,
              "feature_min": float(finite_arr.min()) if finite_arr.size else None,
              "feature_max": float(finite_arr.max()) if finite_arr.size else None}
    report["valid"] = not (report["missing_values_by_column"] or report["non_numeric_or_nonfinite_feature_values"] or unknown_labels or missing_classes)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("csv"); parser.add_argument("--kind",choices=["gesture","eye"],required=True); parser.add_argument("--output")
    args=parser.parse_args(); result=validate(args.csv,args.kind)
    output=Path(args.output) if args.output else Path("ml-engine/reports")/args.kind/"dataset_audit.json"
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if not result["valid"]: raise SystemExit(2)
