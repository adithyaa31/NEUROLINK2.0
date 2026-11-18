"""Local-only webcam landmark collector. Raw frames are never written to disk."""
import argparse
import csv
import hashlib
import json
import math
import os
import urllib.request
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp

from backend.app.features import normalize_eye_landmarks, normalize_landmarks
from backend.app.ml import EYE_STATES, FEATURES, GESTURES

ROOT = Path(__file__).resolve().parents[2]
ASSET_MANIFEST = ROOT / "ml-engine/assets/manifest.json"
TASK_ASSETS = {
    "gesture": ROOT / "ml-engine/assets/hand_landmarker.task",
    "eye": ROOT / "ml-engine/assets/face_landmarker.task",
}

def _asset_matches(path, expected):
    if not path.is_file() or path.stat().st_size != expected["size_bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper() == expected["sha256"].upper()

def ensure_task_asset(kind):
    if kind not in TASK_ASSETS:
        raise ValueError("kind must be gesture or eye")
    record = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))["assets"][kind]
    path = TASK_ASSETS[kind]
    if _asset_matches(path, record):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".download")
    try:
        print(f"Downloading MediaPipe {kind} landmark asset {record['file']} from Google's model bucket.")
        urllib.request.urlretrieve(record["url"], partial)
        if not _asset_matches(partial, record):
            raise RuntimeError(f"Downloaded {record['file']} failed size/SHA-256 verification")
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
    return path


def append_collected_sample(path, session_id, label, features, kind, handedness="unknown", timestamp=None):
    """Append one validated normalized observation using the collector CSV schema."""
    if kind not in FEATURES:
        raise ValueError("kind must be gesture or eye")
    labels = GESTURES if kind == "gesture" else EYE_STATES
    if label not in labels:
        raise ValueError(f"label must be one of {labels}")
    if not session_id:
        raise ValueError("session_id is required")
    if len(features) != FEATURES[kind]:
        raise ValueError(f"Expected {FEATURES[kind]} {kind} features")
    values = [float(value) for value in features]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("features must be finite numbers")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    when = timestamp or datetime.now(timezone.utc).isoformat()
    columns = [f"f{i}" for i in range(FEATURES[kind])]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "session_id", "label", "handedness", *columns])
        writer.writerow([when, session_id, label, handedness, *values])
    return when


def build_landmarker_options(kind, asset_path):
    """Build the same MediaPipe Tasks options used by live collection."""
    if kind not in FEATURES:
        raise ValueError("kind must be gesture or eye")
    vision = mp.tasks.vision
    common = dict(base_options=mp.tasks.BaseOptions(model_asset_path=str(asset_path)),
                  running_mode=vision.RunningMode.VIDEO)
    if kind == "gesture":
        return vision.HandLandmarker, vision.HandLandmarkerOptions(**common, num_hands=1,
            min_hand_detection_confidence=.6, min_hand_presence_confidence=.5, min_tracking_confidence=.5)
    return vision.FaceLandmarker, vision.FaceLandmarkerOptions(**common, num_faces=1,
        min_face_detection_confidence=.6, min_face_presence_confidence=.5, min_tracking_confidence=.5,
        output_face_blendshapes=False, output_facial_transformation_matrixes=False)


def open_webcam(camera_index):
    """Open a webcam with a verified frame using Windows capture backends."""
    backends = ((cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation"))
    for backend, name in backends:
        cap = None
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    print(f"Webcam {camera_index} opened with {name}.")
                    return cap
        except Exception:
            pass
        if cap is not None:
            cap.release()
    raise RuntimeError(
        f"Could not open webcam {camera_index} with DirectShow or Media Foundation "
        "(both failed to provide a valid frame)."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["gesture", "eye"])
    parser.add_argument("--output", help="CSV destination (defaults under ml-engine/data/raw)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--session", default=None, help="Session ID; generated when omitted")
    args = parser.parse_args()
    kind = args.kind
    labels = GESTURES if kind == "gesture" else EYE_STATES
    count = FEATURES[kind]
    output = Path(args.output or f"ml-engine/data/raw/{'gestures' if kind == 'gesture' else 'eyes'}/{kind}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    session = args.session or str(uuid.uuid4())
    exists = output.exists() and output.stat().st_size > 0
    cap = open_webcam(args.camera)
    try:
        asset = ensure_task_asset(kind)
        detector_type, options = build_landmarker_options(kind, asset)
        tracker = detector_type.create_from_options(options)
    except Exception:
        cap.release()
        raise
    current, recording, previous, last_saved = labels[0], False, None, 0.0
    counts = {label: 0 for label in labels}
    try:
        if exists:
            with output.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("label") in counts: counts[row["label"]] += 1
        key_labels = {ord(str(i+1)): label for i, label in enumerate(labels)}
        frame_no = 0
        print(f"Session {session}; output {output.resolve()}")
        print("Keys: " + ", ".join(f"{key_labels[k]}={chr(k)}" for k in key_labels) + "; SPACE record/pause; Q quit")
        while True:
            ok, frame = cap.read()
            if not ok: print("Webcam frame unavailable."); break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_no += 1
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = tracker.detect_for_video(image, frame_no * 33)
            feature, handedness = None, "unknown"
            if kind == "gesture" and result.hand_landmarks:
                hand = result.hand_landmarks[0]
                connections = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17))
                for a,b in connections:
                    cv2.line(frame,(int(hand[a].x*frame.shape[1]),int(hand[a].y*frame.shape[0])),(int(hand[b].x*frame.shape[1]),int(hand[b].y*frame.shape[0])),(0,230,255),2)
                for p in hand:
                    cv2.circle(frame,(int(p.x*frame.shape[1]),int(p.y*frame.shape[0])),3,(245,245,245),-1)
                handedness = result.handedness[0][0].category_name
                # Process raw camera orientation to match the browser feature extractor.
                feature = normalize_landmarks([[p.x, p.y, p.z] for p in hand], handedness)
            elif kind == "eye" and result.face_landmarks:
                face = result.face_landmarks[0]
                from backend.app.features import EYE_LANDMARKS
                selected = sorted(set(i for contour in EYE_LANDMARKS.values() for i in contour))
                for idx in selected:
                    p = face[idx]
                    cv2.circle(frame, (int(p.x*frame.shape[1]), int(p.y*frame.shape[0])), 2, (0, 230, 255), -1)
                feature = normalize_eye_landmarks([[p.x, p.y, p.z] for p in face])
            if feature is not None:
                cv2.putText(frame, "LANDMARKS DETECTED", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, .65, (50, 230, 160), 2)
            else:
                cv2.putText(frame, "NO HAND" if kind == "gesture" else "NO FACE", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, .65, (80, 120, 255), 2)
            cv2.putText(frame, f"{kind.upper()} / {current} / {'RECORDING' if recording else 'PAUSED'}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 230, 255), 2)
            cv2.putText(frame, "  ".join(f"{i+1}:{label}={counts[label]}" for i,label in enumerate(labels)), (12, 54), cv2.FONT_HERSHEY_SIMPLEX, .42, (235,235,235), 1)
            cv2.imshow("NEUROLINK local dataset collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27): break
            if key == ord(" "): recording = not recording; previous = None
            if key in key_labels: current = key_labels[key]; previous = None
            now = time.monotonic()
            if recording and feature is not None and now-last_saved >= .12 and (previous is None or max(abs(a-b) for a,b in zip(feature, previous)) > .003):
                append_collected_sample(output, session, current, feature, kind, handedness)
                exists = True
                counts[current] += 1; previous = feature; last_saved = now
    finally:
        try:
            tracker.close()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Collection stopped; webcam released. Samples:", counts)


if __name__ == "__main__": main()
