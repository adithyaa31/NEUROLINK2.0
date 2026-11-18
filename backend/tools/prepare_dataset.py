"""Validate and canonicalize a collected CSV without deleting observations."""
import argparse
import json
from pathlib import Path

import pandas as pd

from backend.app.ml import FEATURES
from backend.tools.validate_dataset import validate


def prepare(source, destination, kind):
    report=validate(source,kind)
    if not report["valid"]: raise ValueError(f"Dataset failed validation; see audit report: {json.dumps(report)}")
    df=pd.read_csv(source)
    cols=["timestamp","session_id","label"]
    if "handedness" in df.columns: cols.append("handedness")
    cols.extend(f"f{i}" for i in range(FEATURES[kind]))
    output=Path(destination);output.parent.mkdir(parents=True,exist_ok=True)
    df[cols].to_csv(output,index=False)
    print(f"Prepared {len(df)} rows at {output}; duplicate feature rows audited: {report['duplicate_feature_rows']} (retained, not silently removed).")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("csv");p.add_argument("output");p.add_argument("--kind",choices=["gesture","eye"],required=True);a=p.parse_args();prepare(a.csv,a.output,a.kind)
