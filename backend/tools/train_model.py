"""Train/evaluate one custom NEUROLINK model using session-isolated splits."""
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backend.app.ml import (ARCHITECTURE_VERSION, EYE_STATES, FEATURES, GESTURES,
                            EyeFeatureClassifier, FeatureClassifier, select_device)


def session_split(x, y, groups, seed=7, train_session=None, test_session=None,
                  validation_fraction=.2):
    """Split without test-session leakage.

    With explicit train/test sessions (or exactly two sessions), validation is
    stratified from the training session. With three or more sessions and no
    explicit pair, retain the original fully session-isolated 70/15/15 split.
    """
    sessions = list(dict.fromkeys(str(value) for value in groups))
    if train_session is not None or test_session is not None or len(sessions) == 2:
        if train_session is None and test_session is None:
            train_session, test_session = sessions
        elif train_session is None or test_session is None:
            raise ValueError("Specify both --train-session and --test-session")
        train_session, test_session = str(train_session), str(test_session)
        if train_session == test_session or train_session not in sessions or test_session not in sessions:
            raise ValueError("Training and test sessions must be distinct sessions present in the dataset")
        if set(sessions) != {train_session, test_session}:
            raise ValueError("This two-session protocol requires exactly the selected training and test sessions")
        train_pool = np.flatnonzero(np.asarray(groups, dtype=str) == train_session)
        test = np.flatnonzero(np.asarray(groups, dtype=str) == test_session)
        if set(np.unique(y[train_pool])) != set(np.unique(y)) or set(np.unique(y[test])) != set(np.unique(y)):
            raise ValueError("Every class must be present in both the training and held-out test sessions")
        train, val = train_test_split(train_pool, test_size=validation_fraction,
                                     random_state=seed, stratify=y[train_pool])
        assert not (set(groups[train]) & set(groups[test]))
        assert not (set(groups[val]) & set(groups[test]))
        return np.asarray(train), np.asarray(val), np.asarray(test)

    # Three-way session isolation remains available for datasets with three
    # or more recording sessions.
    labels = set(np.unique(y))
    outer = GroupShuffleSplit(n_splits=400, test_size=.15, random_state=seed)
    for train_val, test in outer.split(x, y, groups):
        if set(np.unique(y[test])) != labels: continue
        inner = GroupShuffleSplit(n_splits=250, test_size=.1765, random_state=seed + 1)
        for train_local, val_local in inner.split(x[train_val], y[train_val], groups[train_val]):
            train, val = train_val[train_local], train_val[val_local]
            if all(set(np.unique(y[idx])) == labels for idx in (train, val)):
                assert not (set(groups[train]) & set(groups[val]) or set(groups[train]) & set(groups[test]) or set(groups[val]) & set(groups[test]))
                return train, val, test
    raise ValueError("Could not create session-isolated train/validation/test splits containing every class. Collect more sessions for each label.")


def load_dataset(csv_path, kind):
    labels = GESTURES if kind == "gesture" else EYE_STATES
    n_features = FEATURES[kind]
    columns = [f"f{i}" for i in range(n_features)]
    df = pd.read_csv(csv_path)
    required = {"timestamp", "session_id", "label", *columns}
    if not required.issubset(df.columns): raise ValueError(f"CSV missing columns: {sorted(required-set(df.columns))}")
    if df[list(required)].isna().any().any(): raise ValueError("Dataset contains missing values; run validate_dataset.py and repair collection rows")
    unknown = set(df.label.astype(str)) - set(labels)
    missing = set(labels) - set(df.label.astype(str))
    if unknown or missing: raise ValueError(f"Invalid or missing classes. Unknown={sorted(unknown)}, missing={sorted(missing)}")
    x = df[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all(): raise ValueError("Features contain NaN or infinite values")
    y = df.label.astype(str).map({label: i for i, label in enumerate(labels)}).to_numpy(dtype=np.int64)
    groups = df.session_id.astype(str).to_numpy()
    return df, x, y, groups, labels


def audit_dataset(csv_path, kind, output=None):
    df, x, y, groups, labels = load_dataset(csv_path, kind)
    report = {"kind": kind, "rows": len(df), "features": x.shape[1], "classes": labels,
              "class_counts": {label: int((df.label == label).sum()) for label in labels},
              "sessions": int(df.session_id.nunique()),
              "samples_per_session": {str(k): int(v) for k, v in df.session_id.value_counts().sort_index().items()},
              "duplicate_feature_rows": int(df.duplicated(subset=[f"f{i}" for i in range(x.shape[1])]).sum()),
              "invalid_values": int((~np.isfinite(x)).sum()),
              "feature_min": float(x.min()), "feature_max": float(x.max())}
    if output: Path(output).parent.mkdir(parents=True, exist_ok=True); Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train(csv_path, output, epochs=50, model_kind="gesture", batch_size=64, lr=1e-3,
          patience=8, seed=7, report_dir=None, train_session=None,
          test_session=None, validation_fraction=.2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"): torch.backends.cudnn.benchmark = False
    device = select_device()
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"Training device: {device}")
    df, x, y, groups, labels = load_dataset(csv_path, model_kind)
    sessions = list(dict.fromkeys(groups.tolist()))
    holdout_protocol = train_session is not None or test_session is not None or len(sessions) == 2
    tr, va, te = session_split(x, y, groups, seed, train_session, test_session, validation_fraction)
    print(f"Dataset size: {len(df)}; classes: {len(labels)}")
    print("Split sessions:", {name: sorted(set(groups[idx])) for name, idx in (("train", tr), ("validation", va), ("test", te))})
    mean = x[tr].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x[tr].std(axis=0, dtype=np.float64).astype(np.float32)
    zero_variance = np.flatnonzero(scale < 1e-8).tolist()
    scale[scale < 1e-8] = 1.0
    x_train = ((x[tr] - mean) / scale).astype(np.float32)
    x_val = ((x[va] - mean) / scale).astype(np.float32)
    model = (FeatureClassifier(len(labels)) if model_kind == "gesture" else EyeFeatureClassifier(len(labels))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y[tr])), batch_size=batch_size, shuffle=True, drop_last=(len(tr) > batch_size and len(tr) % batch_size == 1), generator=torch.Generator().manual_seed(seed))
    xv, yv = torch.from_numpy(x_val).to(device), torch.from_numpy(y[va]).to(device)
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    protected_paths = [out, out.with_name(out.stem + "_best" + out.suffix),
                       out.with_name(out.stem + "_last" + out.suffix), out.parent / "metadata.json"]
    occupied = [str(path) for path in protected_paths if path.exists()]
    if occupied: raise FileExistsError(f"Refusing to overwrite existing model artifacts: {occupied}")
    reports_root = Path(report_dir or Path(__file__).resolve().parents[2] / "ml-engine" / "reports" / model_kind)
    reports_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reports = reports_root / run_id
    reports.mkdir()
    history = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        model.train(); total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()
            total += float(loss.item()) * len(xb)
        model.eval()
        with torch.inference_mode():
            logits = model(xv); val_loss = float(criterion(logits, yv).item()); val_acc = float((logits.argmax(1) == yv).float().mean().item())
        train_loss = total / len(tr)
        history["train_loss"].append(train_loss); history["validation_loss"].append(val_loss); history["validation_accuracy"].append(val_acc)
        print(f"Epoch {epoch+1:03d}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_acc:.4f}")
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"architecture_version": ARCHITECTURE_VERSION, "features": FEATURES[model_kind], "labels": labels,
                        "state_dict": best_state}, out.with_name(out.stem + "_best" + out.suffix))
        else:
            stale += 1
            if stale >= patience: print("Early stopping: validation loss did not improve."); break
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"architecture_version": ARCHITECTURE_VERSION, "features": FEATURES[model_kind], "labels": labels,
                "checkpoint_role": "last_epoch", "state_dict": last_state}, out.with_name(out.stem + "_last" + out.suffix))
    model.load_state_dict(best_state); model.to(device).eval()
    torch.save({"architecture_version": ARCHITECTURE_VERSION, "state_dict": model.state_dict(),
                "labels": labels, "features": FEATURES[model_kind]}, out)
    # Held-out test evaluation occurs exactly once after model selection.
    xt, yt = torch.from_numpy(((x[te] - mean) / scale).astype(np.float32)).to(device), y[te]
    start = time.perf_counter()
    with torch.inference_mode(): predicted = model(xt).argmax(1).cpu().numpy()
    latency_ms = (time.perf_counter() - start) * 1000 / len(te)
    precision, recall, f1, support = precision_recall_fscore_support(yt, predicted, labels=list(range(len(labels))), zero_division=0)
    metrics = {"accuracy": float(accuracy_score(yt, predicted)),
               "macro_precision": float(precision_recall_fscore_support(yt, predicted, average="macro", zero_division=0)[0]),
               "macro_recall": float(precision_recall_fscore_support(yt, predicted, average="macro", zero_division=0)[1]),
               "macro_f1": float(precision_recall_fscore_support(yt, predicted, average="macro", zero_division=0)[2]),
               "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i,label in enumerate(labels)},
               "confusion_matrix": confusion_matrix(yt, predicted, labels=list(range(len(labels)))).tolist(),
               "mean_batch_inference_latency_ms_per_sample": latency_ms, "parameter_count": sum(p.numel() for p in model.parameters()),
               "test_samples": len(te), "validation_samples": len(va)}
    metrics["classification_report"] = classification_report(yt, predicted, target_names=labels, output_dict=True, zero_division=0)
    stamp = datetime.now(timezone.utc).isoformat()
    metadata = {"architecture_version": ARCHITECTURE_VERSION, "kind": model_kind, "feature_dimensions": FEATURES[model_kind],
                "feature_ordering": "MediaPipe hand xyz landmarks 0..20, wrist-relative then max-absolute scaled" if model_kind == "gesture" else "left then right eye; six contour points (outer, upper1, upper2, inner, lower1, lower2), xy eye-relative",
                "normalization": "wrist origin; canonical right-hand; max-absolute scale" if model_kind == "gesture" else "subtract eye outer corner; divide by outer-inner corner distance",
                "labels": labels, "label_mapping": {label: i for i, label in enumerate(labels)},
                "preprocessing": {"method": "standardize", "fit_scope": "training_split_only", "fit_sessions": sorted(set(groups[tr])), "mean": mean.tolist(), "scale": scale.tolist(), "zero_variance_features": zero_variance},
                "split_protocol": "training_session_internal_validation_and_heldout_session_test" if holdout_protocol else "three_way_session_isolated",
                "training": {"run_id": run_id, "epochs_requested": epochs, "epochs_completed": len(history["train_loss"]), "batch_size": batch_size, "optimizer": "AdamW", "learning_rate": lr, "seed": seed, "device": str(device), "early_stopping_patience": patience, "validation_fraction_of_training_session": validation_fraction if holdout_protocol else None},
                "dataset": {"path": str(Path(csv_path)), "rows": len(df), "version": 1}, "splits": {name: {"rows": len(idx), "sessions": sorted(set(groups[idx]))} for name,idx in (("train",tr),("validation",va),("test",te))},
                "torch_version": torch.__version__, "metrics": metrics, "created_at": stamp}
    (out.parent / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (reports / "evaluation.json").write_text(json.dumps({"kind": model_kind, "metrics": metrics, "metadata": metadata}, indent=2), encoding="utf-8")
    (reports / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(); ax.plot(history["train_loss"], label="train"); ax.plot(history["validation_loss"], label="validation"); ax.set(xlabel="Epoch", ylabel="Cross entropy", title=f"{model_kind} training loss"); ax.legend(); fig.tight_layout(); fig.savefig(reports / "loss.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(); ax.imshow(metrics["confusion_matrix"], cmap="Blues"); ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="Actual", title=f"{model_kind} test confusion matrix"); plt.setp(ax.get_xticklabels(), rotation=35, ha="right"); fig.tight_layout(); fig.savefig(reports / "confusion_matrix.png", dpi=150); plt.close(fig)
    except ImportError: print("Matplotlib unavailable; JSON evaluation reports were written without PNG plots.")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv"); parser.add_argument("output", nargs="?")
    parser.add_argument("--kind", choices=["gesture", "eye"], default="gesture")
    parser.add_argument("--epochs", type=int, default=50); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7); parser.add_argument("--audit", action="store_true")
    parser.add_argument("--train-session"); parser.add_argument("--test-session")
    parser.add_argument("--validation-fraction", type=float, default=.2)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit_dataset(args.csv, args.kind, Path("ml-engine/reports") / args.kind / "dataset_audit.json"), indent=2))
    else:
        target = args.output or (f"ml-engine/models/{args.kind}/{args.kind}_model.pth")
        print(json.dumps(train(args.csv, target, args.epochs, args.kind, args.batch_size, args.lr, args.patience, args.seed,
                               train_session=args.train_session, test_session=args.test_session,
                               validation_fraction=args.validation_fraction), indent=2))
