# MediaPipe detector assets

These are landmark detectors only. Gesture and eye-state classification remain custom PyTorch models.

| Pipeline | Asset | MediaPipe Tasks model variant | Size |
|---|---|---|---:|
| Hand landmarks | `hand_landmarker.task` | `hand_landmarker/float16/1` | 7,819,105 bytes |
| Face/eye landmarks | `face_landmarker.task` | `face_landmarker/float16/1` | 3,758,596 bytes |

Both files were downloaded from the exact Google MediaPipe model-storage object URLs in `manifest.json`. Their byte lengths and MD5 values were checked against the Google Cloud Storage object metadata; SHA-256 values are recorded there for local verification. The object generation IDs are also recorded. The configured Python runtime reports `mediapipe 1.2.3` and exposes both Task APIs.

Official task references: [Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) and [Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker).
