"""Feature helpers for landmark models. No raw image data is persisted."""
import numpy as np

EYE_LANDMARKS = {
    # MediaPipe Face Mesh: outer corner, upper contour, inner corner, lower contour.
    "left": (33, 160, 158, 133, 153, 144),
    "right": (362, 385, 387, 263, 373, 380),
}

def normalize_landmarks(points, handedness="Right"):
    """Translate to wrist origin, mirror left hands to right-hand orientation, then scale."""
    points=np.asarray(points,dtype=np.float32)
    if points.shape != (21,3): raise ValueError("Expected 21 hand landmarks with x, y, z coordinates")
    if not np.isfinite(points).all(): raise ValueError("Hand landmarks must be finite numbers")
    points=points-points[0]
    if str(handedness).lower() == "left": points[:,0] *= -1
    scale=float(np.max(np.abs(points)))
    if scale > 0: points=points/scale
    return points.reshape(-1).tolist()

def eye_aspect_ratio(eye_points):
    """Compute a standard 2D eye aspect ratio from six ordered eye landmarks."""
    p=np.asarray(eye_points,dtype=np.float32)
    if p.shape != (6,2): raise ValueError("Expected six 2D eye landmarks")
    horizontal=float(np.linalg.norm(p[0]-p[3]))
    if horizontal == 0: return 0.0
    return float((np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2*horizontal))

def normalize_eye_landmarks(face_points):
    """Return 24 values: six ordered (x,y) contour points per eye, eye-relative.

    For each eye, subtract its first (outer-corner) point and divide by the
    outer-to-inner corner distance. This is invariant to image position/scale.
    """
    points = np.asarray(face_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] <= max(max(x) for x in EYE_LANDMARKS.values()):
        raise ValueError("Expected MediaPipe Face Mesh landmarks containing both eyes")
    if not np.isfinite(points).all():
        raise ValueError("Face landmarks must be finite numbers")
    result = []
    for indices in EYE_LANDMARKS.values():
        eye = points[list(indices), :2].copy()
        scale = float(np.linalg.norm(eye[3] - eye[0]))
        if scale <= 1e-8:
            raise ValueError("Eye contour is degenerate")
        eye = (eye - eye[0]) / scale
        result.extend(eye.reshape(-1).tolist())
    return result
