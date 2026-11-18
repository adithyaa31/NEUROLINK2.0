# NEUROLINK local ML workflow

This workspace trains two independent custom PyTorch classifiers from local webcam landmarks. Raw images are not written by default. Collection targets (6,000 hand rows and 2,000 eye rows) are goals. This repository currently includes gesture and eye datasets, checkpoints, metadata, and evaluation reports under `ml-engine/data/`, `ml-engine/models/`, and `ml-engine/reports/`. Review the report for per-class weaknesses before relying on a model. `ml-engine/requirements.txt` includes backend/test dependencies; install the CUDA-matched PyTorch package separately so its selected build is preserved.

## Environment and GPU

Use Python 3.11 on Windows. From the repository root in PowerShell:

```powershell
py -3.11 -m venv ml-engine/.venv
ml-engine/.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
```

Install the PyTorch build that matches the laptop and Python environment by choosing Windows, Pip, Python, and the compatible CUDA option at the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Run the generated command, then install the remaining libraries:

```powershell
python -m pip install -r ml-engine/requirements.txt
python -c "import torch; print('PyTorch',torch.__version__); print('CUDA',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
```

The trainer selects CUDA when `torch.cuda.is_available()` is true and prints the selected device. Otherwise it prints and uses CPU.

## Collect real observations

Run multiple distinct sessions across days, camera positions, lighting, and (for hands) both handednesses. Record every class during each session where practical; this lets the session-isolated split keep each class represented in train, validation, and test. Each collector shows the local preview, landmark overlay, counters, label controls, and missing-hand/face state. Press Space to start/pause, number keys to choose labels, and Q to stop. Frames stay in memory and are never saved.

```powershell
python -m backend.tools.collect_webcam gesture
python -m backend.tools.collect_webcam eye
```

Gesture labels use keys 1–6 in this order: `open_palm`, `fist`, `thumbs_up`, `peace`, `pinch`, `pointing`. Eye uses 1=`open`, 2=`closed`. Default CSVs are `ml-engine/data/raw/gestures/gesture.csv` and `ml-engine/data/raw/eyes/eye.csv`. Pass `--output`, `--camera`, or `--session` to change those options.

Gesture rows contain 63 values from 21 MediaPipe hand landmarks (xyz), wrist-centered, left-hand mirrored, and max-absolute scaled. Eye rows contain 24 values: left then right eye, six ordered contour points per eye (outer, upper contour 1, upper contour 2, inner, lower contour 1, lower contour 2), xy relative to the outer corner and divided by eye width. Face Mesh indices are documented in `backend/app/features.py` and shared with the browser code.

## Audit, train, evaluate

Audit does not modify the source CSV. It reports class counts, missing/nonfinite values, duplicates, sessions and per-session counts, and large outliers. Training requires every class and enough distinct sessions to put every class in train, validation, and test without session overlap.

```powershell
python -m backend.tools.validate_dataset ml-engine/data/raw/gestures/gesture.csv --kind gesture
python -m backend.tools.validate_dataset ml-engine/data/raw/eyes/eye.csv --kind eye
python -m backend.tools.prepare_dataset ml-engine/data/raw/gestures/gesture.csv ml-engine/data/processed/gestures/gesture.csv --kind gesture
python -m backend.tools.prepare_dataset ml-engine/data/raw/eyes/eye.csv ml-engine/data/processed/eyes/eye.csv --kind eye
python -m backend.tools.train_model ml-engine/data/raw/gestures/gesture.csv --kind gesture
python -m backend.tools.train_model ml-engine/data/raw/eyes/eye.csv --kind eye
python -m backend.tools.evaluate_model ml-engine/data/raw/gestures/gesture.csv ml-engine/models/gesture/gesture_model.pth --kind gesture
python -m backend.tools.evaluate_model ml-engine/data/raw/eyes/eye.csv ml-engine/models/eye/eye_model.pth --kind eye
```

For a local smoke inference once weights exist, pass a single normalized vector to `python -m backend.tools.predict_model --kind eye <24 floats>` or `--kind gesture <63 floats>`.

Training defaults: 50 epochs, batch 64, AdamW, cross entropy, early stopping on validation loss, and fixed seed 7. Flags `--epochs`, `--batch-size`, `--lr`, `--patience`, and `--seed` override defaults. The split groups by session at 70/15/15 and fails clearly if every class cannot be represented in all partitions. The test set is used only after model selection. Checkpoints, metadata, split session IDs, training history, JSON metrics, and loss/confusion PNGs are written under `ml-engine/models/` and `ml-engine/reports/`. Evaluation values are generated only from collected observations.

The FastAPI service loads `ml-engine/models/{gesture,eye}/{kind}_model.pth` once at startup. A model is reported ready only when its checkpoint has matching architecture/class metadata and a complete held-out evaluation sidecar with disjoint train/validation/test sessions. Restart it after training. To override model lookup, set `MODEL_DIR` to an alternate weights directory. Status distinguishes missing, ready, and invalid checkpoint states.

## Start the app and smoke-test inference

In separate PowerShell terminals, activate the environment and start the API, then start the frontend from the repository root:

```powershell
uvicorn backend.app.main:app --reload --port 8000
npm run dev
```

After training and restarting FastAPI, open Gesture Lab or Eye & Voice Lab and click **Start camera**. The browser extracts landmarks locally; the app sends normalized features over `WS /ws/vision`. For a local CLI check using a stored row:

```powershell
$row = Import-Csv ml-engine/data/raw/eyes/eye.csv | Select-Object -First 1
$features = 0..23 | ForEach-Object { [double]$row.PSObject.Properties["f$_"].Value }
python -m backend.tools.predict_model --kind eye $features
```

Use `0..62` and `--kind gesture` for the hand model. This CLI invokes the local checkpoint without camera access.

## Inference contracts

- Gesture: 63 values; custom 63→128→64→6 network.
- Eye: 24 values; custom 24→64→32→2 network.
- Blink selection uses the eye model's temporal OPEN→CLOSED→OPEN output, a 50–500 ms closed interval, and an 800 ms cooldown. Eye state alone never fires repeated selections.
- Eye landmark inference happens in the browser after Start camera; normalized features only are sent over the local FastAPI WebSocket.

The included model artifacts are trained checkpoints with saved held-out reports. Those recorded evaluations do not guarantee live webcam accuracy. The gesture report has zero held-out recall for `open_palm`; this class should be treated as unreliable until new session-diverse data has been audited and an independent evaluation supports a replacement.
