# NEUROLINK 2.0

**Where Biology Meets Technology** — an educational human–machine interface prototype. It is not a medical diagnostic product, neural implant, or physiological simulator.

## Architecture

- React 18 + Vite 6 frontend with seven React Router routes.
- FastAPI/Pydantic REST API and `/ws/vision` WebSocket.
- SQLite event log (`neurolink.db` by default).
- Browser-side MediaPipe Tasks Vision for hand/face landmarks. Camera frames remain in the browser; normalized landmark features are sent to the local API.
- PyTorch classifiers: gesture (63 features, six classes) and eye (24 features, `open`/`closed`). Model readiness is validated against checkpoint and evaluation metadata at API startup.
- Browser Web Speech API for voice navigation where supported.

## Local setup

Requires Windows 10/11, Node.js 20+, Python 3.11 (recommended by the ML tooling), webcam for camera features.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
npm install
```

Start the backend in one terminal:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Start the frontend in another terminal:

```powershell
npm run dev
```

Open `http://localhost:5173`. Development and preview servers bind to loopback by default. The API is unauthenticated and intended for a local single-user demonstration only. Do not expose it to a LAN or public network. A public deployment needs authentication/authorization, rate limiting, secure transport, operational configuration, and an explicit data policy before exposure; CORS is not authentication.

The frontend API defaults to `http://127.0.0.1:8000`; configure `VITE_API_URL` at build time to change it. `FRONTEND_ORIGINS` is a comma-separated exact origin allowlist. `DATABASE_PATH` and `MODEL_DIR` may be set to override their defaults. See `.env.example` for backend variables.

## Features and limitations

Routes: Overview, Gesture Lab, Eye & Voice Lab, Neural Simulator, Neuron Comparison, Performance Analytics, and About.

- Camera access begins only after the user selects Start. Raw webcam images are not stored by application code.
- Gesture features are 21 xyz points normalized wrist-relative and max-absolute scaled. Classes: `open_palm`, `fist`, `thumbs_up`, `peace`, `pinch`, `pointing`.
- The checked-in gesture report records 75.40% held-out accuracy; `open_palm` recall is 0% (all 152 held-out open-palm examples were classified as fist). Treat open palm as unreliable; do not use this prototype for safety-critical control. Check `ml-engine/reports/gesture/session1-train_session2-test-20261114/evaluation.json`.
- Eye checkpoint and metadata are included at `ml-engine/models/eye/`; its saved held-out accuracy is 76.32%. An independent report records 89.25%. These are recorded-data evaluations, not live-camera performance guarantees.
- Cursor is an in-app pointer only. Pinch is geometric. Voice commands are deterministic and depend on browser support; browsers may use their own speech recognition service.
- MediaPipe runtime assets download from public CDNs after camera activation, so first camera startup requires network access.

Model weights, metadata, and datasets are present under `ml-engine/`. No training runs automatically. See [ml-engine/README.md](ml-engine/README.md) before collecting or training; preserve existing artifacts and use a new evaluation run when investigating performance.

## Event storage and cleanup

SQLite stores interaction kind/label, confidence, latency, optional session ID, metadata, and timestamp. It does not store camera frames or voice recordings. Events currently have no automatic retention schedule. Review local event retention and prune intentionally with an explicit UTC cutoff:

```powershell
python -m backend.tools.prune_events --before 2026-01-01T00:00:00
```

This permanently removes matching event rows from the local SQLite database. Back up the database first if those analytics must be retained. Indexes support event kind/order and timestamp queries.

## Verification

```powershell
npm test
npm run build
python -m pytest backend/tests -q
python -m pip check
```

The frontend tests cover feature normalization, blink timing, interaction coordination, and voice parsing. Backend tests cover API contracts, features, checkpoint validation, and pipelines. Camera, microphone, real browser rendering, and real-user behavior still require manual verification on target devices. Use [docs/ACCEPTANCE_TESTING.md](docs/ACCEPTANCE_TESTING.md) to record the browser/device checks and guided five-user review; those checks are not represented as complete until results are recorded.

## Security and release boundary

This repository is configured as a local prototype, not a hosted service. FastAPI routes and WebSocket are unauthenticated; event metadata is size-bounded, but there is no rate limiting or user authorization. Bind to `127.0.0.1` for demonstrations. Before network deployment, add an authentication/authorization design, WebSocket access control, rate limiting/connection limits, HTTPS, production server configuration, monitoring, and retention policy. Do not run Uvicorn with `--reload` in production.
