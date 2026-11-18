export function createInteractionCoordinator({ cooldownMs = 700 } = {}) {
  let mode = 'paused';
  let lastSelection = -Infinity;
  let lastAction = null;
  return {
    get mode() { return mode; },
    pause() { mode = 'paused'; lastAction = 'pause'; return mode; },
    stop() { mode = 'stopped'; lastAction = 'stop'; return mode; },
    activate() { if (mode !== 'stopped') mode = 'active'; return mode; },
    handle(action, now = Date.now()) {
      if (action === 'fist' || action === 'stop') { this.pause(); return { accepted: true, mode, action }; }
      if (action === 'open_palm') { this.activate(); return { accepted: true, mode, action }; }
      if (mode !== 'active') return { accepted: false, mode, reason: 'paused' };
      if (action === 'pinch' || action === 'blink') {
        if (lastAction === 'select' || now - lastSelection < cooldownMs) return { accepted: false, mode, reason: 'cooldown' };
        lastSelection = now; lastAction = 'select'; return { accepted: true, mode, action: 'select' };
      }
      if (action === 'pointing') return { accepted: true, mode, action: 'move-cursor' };
      if (action === 'peace') return { accepted: true, mode, action: 'switch-mode' };
      if (action === 'thumbs_up') return { accepted: true, mode, action: 'confirm' };
      return { accepted: false, mode, reason: 'unknown' };
    },
  };
}

export function stablePrediction(history, prediction, confidence, { threshold = 0.8, frames = 3 } = {}) {
  if (!prediction || !Number.isFinite(confidence) || confidence < threshold) { history.length = 0; return null; }
  history.push(prediction);
  if (history.length > frames) history.shift();
  return history.length === frames && history.every(label => label === prediction) ? prediction : null;
}
