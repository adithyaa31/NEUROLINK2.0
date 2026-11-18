"""Append labeled landmark vectors to a CSV. Feed in MediaPipe-extracted vectors from the client."""
import argparse, csv, math
from datetime import datetime, timezone
from pathlib import Path
from backend.app.features import normalize_landmarks

def append_sample(path, session_id, label, features):
    if len(features) != 63: raise ValueError("Expected 63 normalized landmark features")
    if not all(math.isfinite(float(value)) for value in features): raise ValueError("Features must be finite numbers")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file: writer.writerow(["timestamp", "session_id", "label", *[f"f{i}" for i in range(63)]])
        writer.writerow([datetime.now(timezone.utc).isoformat(), session_id, label, *features])

def append_landmarks(path, session_id, label, points, handedness="Right"):
    """Normalize raw 21x3 MediaPipe landmarks identically to live browser inference."""
    append_sample(path, session_id, label, normalize_landmarks(points, handedness))

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("csv_path"); parser.add_argument("session_id"); parser.add_argument("label"); parser.add_argument("features", nargs=63, type=float)
    args=parser.parse_args(); append_sample(args.csv_path,args.session_id,args.label,args.features)
