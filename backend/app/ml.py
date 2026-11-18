"""Custom classifiers and strict, one-time checkpoint loading for the API."""
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from .config import MODEL_DIR, ROOT

GESTURES = ["open_palm", "fist", "thumbs_up", "peace", "pinch", "pointing"]
EYE_STATES = ["open", "closed"]
FEATURES = {"gesture": 63, "eye": 24}
ARCHITECTURE_VERSION = "1.0"


class FeatureClassifier(nn.Module):
    """Gesture classifier: Linear(63,128)-BN-ReLU-Dropout-64-6."""
    def __init__(self, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(.2),
            nn.Linear(64, output_size),
        )
    def forward(self, x): return self.network(x)


class EyeFeatureClassifier(nn.Module):
    def __init__(self, output_size=2):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(24, 64), nn.ReLU(), nn.Dropout(.25),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_size))
    def forward(self, x): return self.network(x)


def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelRegistry:
    def __init__(self):
        self.models = {key: None for key in FEATURES}
        self.states = {key: "model_not_trained" for key in FEATURES}
        self.errors = {}
        self.preprocessing = {key: None for key in FEATURES}
        self.evaluations = {key: None for key in FEATURES}

    def load(self):
        for key in FEATURES:
            self.models[key] = None
            self.states[key] = "model_not_trained"
            self.errors.pop(key, None)
            self.preprocessing[key] = None
            self.evaluations[key] = None
            checkpoint_names = ("eye_model_v2.pth", "eye_model.pth", "eye.pth") if key == "eye" else ("gesture_model.pth", "gesture.pth")
            candidates = [ROOT / "ml-engine" / "models" / key / name for name in checkpoint_names]
            candidates.extend((MODEL_DIR / f"{key}_model.pth", MODEL_DIR / f"{key}.pth"))
            path = next((p for p in candidates if p.exists()), None)
            if path is None: continue
            self.states[key] = "model_loading"
            try:
                model = FeatureClassifier(len(GESTURES)) if key == "gesture" else EyeFeatureClassifier()
                payload = torch.load(path, map_location="cpu", weights_only=True)
                labels = GESTURES if key == "gesture" else EYE_STATES
                metadata_path = path.parent / "metadata.json"
                if not metadata_path.exists():
                    raise ValueError("Evaluation metadata is required before a checkpoint can be marked ready")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("kind") != key or metadata.get("architecture_version") != ARCHITECTURE_VERSION:
                    raise ValueError("Evaluation metadata kind or architecture version is incompatible")
                if metadata.get("feature_dimensions") != FEATURES[key] or metadata.get("label_mapping") != {label: i for i, label in enumerate(labels)}:
                    raise ValueError("Evaluation metadata feature dimensions or class mapping are incompatible")
                splits = metadata.get("splits", {})
                split_sessions = [set(splits.get(name, {}).get("sessions", [])) for name in ("train", "validation", "test")]
                protocol = metadata.get("split_protocol")
                if protocol == "training_session_internal_validation_and_heldout_session_test":
                    split_safe = not (split_sessions[0] & split_sessions[2] or split_sessions[1] & split_sessions[2])
                elif protocol == "three_way_session_isolated":
                    split_safe = not any(split_sessions[i] & split_sessions[j] for i in range(3) for j in range(i+1, 3))
                else:
                    split_safe = False
                metrics = metadata.get("metrics", {})
                scores = [metrics.get(name) for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1")]
                split_rows = [int(splits.get(name, {}).get("rows", 0)) for name in ("train", "validation", "test")]
                dataset_rows = int(metadata.get("dataset", {}).get("rows", 0))
                if (not all(split_sessions) or not split_safe
                        or any(rows < 1 for rows in split_rows) or sum(split_rows) != dataset_rows
                        or int(metrics.get("test_samples", 0)) != split_rows[2]
                        or any(not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1 for score in scores)
                        or set(metrics.get("per_class", {})) != set(labels)
                        or len(metrics.get("confusion_matrix", [])) != len(labels)):
                    raise ValueError("Checkpoint has no complete, session-isolated held-out evaluation")
                preprocessing = metadata.get("preprocessing", {})
                mean, scale = preprocessing.get("mean"), preprocessing.get("scale")
                if (preprocessing.get("method") != "standardize"
                        or preprocessing.get("fit_scope") != "training_split_only"
                        or set(preprocessing.get("fit_sessions", [])) != split_sessions[0]
                        or not isinstance(mean, list) or not isinstance(scale, list)
                        or len(mean) != FEATURES[key] or len(scale) != FEATURES[key]
                        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in mean + scale)
                        or any(v <= 0 for v in scale)):
                    raise ValueError("Checkpoint preprocessing metadata is missing or invalid")
                self.preprocessing[key] = (torch.tensor(mean, dtype=torch.float32),
                                           torch.tensor(scale, dtype=torch.float32))
                if not isinstance(payload, dict) or "state_dict" not in payload:
                    raise ValueError("Checkpoint is missing required version and class metadata")
                if payload.get("architecture_version") != ARCHITECTURE_VERSION:
                    raise ValueError("Checkpoint architecture version is incompatible")
                if payload.get("features") != FEATURES[key] or payload.get("labels") != labels:
                    raise ValueError("Checkpoint feature dimensions or class ordering are incompatible")
                payload = payload["state_dict"]
                model.load_state_dict(payload, strict=True)
                model.eval()
                self.models[key] = model
                self.states[key] = "ready"
                self.evaluations[key] = {"protocol": protocol, "accuracy": metrics.get("accuracy"),
                                         "per_class": metrics.get("per_class", {})}
            except Exception as exc:
                self.preprocessing[key] = None
                self.errors[key] = str(exc)
                self.states[key] = "model_error"

    def status(self):
        labels = {"gesture": GESTURES, "eye": EYE_STATES}
        result = {}
        for key in FEATURES:
            item = {"available": self.models[key] is not None, "status": self.states[key],
                    "labels": labels[key], "features": FEATURES[key],
                    **({"error": self.errors[key]} if key in self.errors else {})}
            if self.models[key] is not None:
                item["evaluation"] = self.evaluations[key]
            result[key] = item
        return result

    def predict(self, key, features):
        if key not in FEATURES: raise ValueError("Unknown model kind")
        if not isinstance(features, (list, tuple)) or len(features) != FEATURES[key]:
            raise ValueError(f"Expected {FEATURES[key]} features")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in features):
            raise ValueError("Features must be finite numbers")
        model = self.models.get(key)
        if model is None:
            state = self.states.get(key, "model_not_trained")
            return {"available": False, "status": state, "prediction": None, "confidence": None, "latency_ms": None}
        start = time.perf_counter()
        tensor = torch.tensor(features, dtype=torch.float32)
        if self.preprocessing.get(key) is not None:
            mean, scale = self.preprocessing[key]
            tensor = (tensor - mean) / scale
        with torch.inference_mode():
            probs = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0]
        labels = GESTURES if key == "gesture" else EYE_STATES
        confidence, idx = torch.max(probs, dim=0)
        return {"available": True, "status": "ready", "prediction": labels[idx.item()],
                "confidence": float(confidence), "latency_ms": (time.perf_counter()-start)*1000}


registry = ModelRegistry()
