"""Evaluate a saved checkpoint on its recorded held-out session list."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from backend.app.ml import FEATURES, GESTURES, EYE_STATES, FeatureClassifier, EyeFeatureClassifier


def evaluate(csv_path, checkpoint, kind):
    checkpoint=Path(checkpoint); metadata=json.loads((checkpoint.parent/"metadata.json").read_text(encoding="utf-8"))
    labels=GESTURES if kind=="gesture" else EYE_STATES
    if metadata.get("kind")!=kind or metadata.get("label_mapping")!={label:i for i,label in enumerate(labels)}: raise ValueError("Checkpoint metadata is incompatible with requested model")
    sessions=set(metadata["splits"]["test"]["sessions"])
    df=pd.read_csv(csv_path); df=df[df.session_id.astype(str).isin(sessions)]
    if not sessions or df.empty: raise ValueError("No held-out test sessions from checkpoint metadata were found in the dataset")
    columns=[f"f{i}" for i in range(FEATURES[kind])]; x=df[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all() or set(df.label.astype(str))!=set(labels): raise ValueError("Held-out sessions have invalid values or are missing classes")
    y=df.label.astype(str).map({label:i for i,label in enumerate(labels)}).to_numpy(dtype=np.int64)
    model=FeatureClassifier(len(labels)) if kind=="gesture" else EyeFeatureClassifier()
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if payload.get("architecture_version")!=metadata.get("architecture_version"): raise ValueError("Checkpoint version does not match metadata")
    model.load_state_dict(payload["state_dict"]);model.eval();start=time.perf_counter()
    with torch.inference_mode(): pred=model(torch.from_numpy(x)).argmax(1).numpy()
    latency=(time.perf_counter()-start)*1000/len(y)
    p,r,f,s=precision_recall_fscore_support(y,pred,labels=list(range(len(labels))),zero_division=0)
    result={"kind":kind,"evaluation":"held_out_test","samples":len(y),"accuracy":float(accuracy_score(y,pred)),"macro_precision":float(np.mean(p)),"macro_recall":float(np.mean(r)),"macro_f1":float(np.mean(f)),"per_class":{label:{"precision":float(p[i]),"recall":float(r[i]),"f1":float(f[i]),"support":int(s[i])} for i,label in enumerate(labels)},"confusion_matrix":confusion_matrix(y,pred,labels=list(range(len(labels)))).tolist(),"mean_batch_inference_latency_ms_per_sample":latency,"parameter_count":sum(v.numel() for v in model.parameters())}
    report=checkpoint.parent.parent/"reports"/kind;report.mkdir(parents=True,exist_ok=True);(report/"evaluation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2));return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("csv");p.add_argument("checkpoint");p.add_argument("--kind",choices=["gesture","eye"],required=True);a=p.parse_args();evaluate(a.csv,a.checkpoint,a.kind)
