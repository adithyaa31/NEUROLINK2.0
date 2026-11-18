# NEUROLINK 2.0 acceptance and release checklist

## Current verification record

Automated checks must be run in the release environment and recorded below. No live camera or microphone session, five-person usability session, or cross-browser hardware matrix has been performed as part of this code change. Do not mark those items passed without recording the actual browser, device, participant task and observed result.

| Check | Status | Evidence |
|---|---|---|
| Frontend unit tests | Passed: 15 | `npm test` · 2026-11-18 |
| Backend tests | Passed: 24, skipped: 1 | `python -m pytest backend/tests -q` · 2026-11-18 |
| Production build | Passed | `npm run build` · Vite 6.4.6 · 2026-11-18 |
| Python dependency consistency | Passed | `python -m pip check` · 2026-11-18 |
| Camera permission/start/stop and camera loss | Manual required | Browser/device and observed result |
| Hand tracking, six class outputs and in-app cursor | Manual required | Browser/device; model limitations below |
| Eye state/blink selection | Manual required | Browser/device and observed result |
| Voice support/unsupported/permission handling | Manual required | Browser/device and observed result |
| Five-user guided usability tasks | Not performed | Recruit five users and complete the script below |

## Guided user tasks

For each participant, ask them to:

1. Find and open Gesture Lab from the dashboard.
2. Explain the camera permission prompt before starting, then start and stop the camera.
3. Locate the hand detection and model status, and explain what “model ready” does and does not mean.
4. Adjust cursor sensitivity, calibrate, and move the in-app cursor; try a pinch selection.
5. Open Eye & Voice Lab, start and stop eye tracking, and identify the blink status.
6. If supported, activate voice navigation and say “open analytics”; otherwise find the same page with keyboard navigation.
7. Find the accessibility controls and clear the local event log, confirming the destructive action.
8. Report any confusing status, inaccessible control, unexpected permission request, or failure to recover after stopping.

Record browser/version, OS/device, whether camera/microphone permissions were granted or denied, task completion, time-to-first-success, errors, recovery success, and participant comments. Keep personal or biometric data out of notes. Do not record audio/video.

## Known release limitations

- The stored gesture evaluation has 0% recall for `open_palm` (152 held-out examples were classified as fist). Do not rely on open palm activation as a reliable control or safety mechanism.
- Saved eye evaluation metrics are from recorded data and do not establish live-camera accuracy.
- MediaPipe model assets load from public CDNs on first camera use; a network connection is needed then.
- Browser speech recognition availability and audio processing vary by browser; some browsers may use their own remote recognition service.
- The API has no user authentication and is designed for a single-user local demonstration bound to loopback. It is not ready for public or LAN exposure.
- Local interaction event rows have no automatic retention schedule. Users can explicitly clear them in Analytics or prune by UTC cutoff from the CLI.

## Per-run record

- Date/time and code revision:
- Tester/operator:
- Browser/version and OS/device:
- Commands and results (`npm test`, `npm run build`, `python -m pytest backend/tests -q`, `python -m pip check`):
- Manual camera/eye/voice results:
- Failures and reproducible steps:
- Release decision and owner:
