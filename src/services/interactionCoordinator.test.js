import test from 'node:test';
import assert from 'node:assert/strict';
import { createInteractionCoordinator, stablePrediction } from './interactionCoordinator.js';
import { smoothCursor } from './gestureFeatures.js';

test('pause has priority and selection is gated, debounced and serialized', () => {
  const c = createInteractionCoordinator({ cooldownMs: 700 });
  assert.equal(c.handle('pinch', 1000).accepted, false);
  c.handle('open_palm', 1000);
  assert.equal(c.handle('pinch', 1100).accepted, true);
  assert.equal(c.handle('blink', 1200).accepted, false);
  assert.equal(c.handle('fist', 1250).mode, 'paused');
  assert.equal(c.handle('blink', 2000).accepted, false);
});

test('requires repeated high-confidence model predictions', () => {
  const h = [];
  assert.equal(stablePrediction(h, 'fist', .95), null);
  assert.equal(stablePrediction(h, 'fist', .95), null);
  assert.equal(stablePrediction(h, 'fist', .95), 'fist');
  assert.equal(stablePrediction(h, 'fist', .3), null);
});

test('cursor coordinate smoothing works without classifier state', () => {
  assert.deepEqual(smoothCursor(null, { x: .4, y: .6 }), { x: .4, y: .6 });
});
