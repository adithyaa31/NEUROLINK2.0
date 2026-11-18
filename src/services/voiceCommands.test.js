import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeSpeech, parseVoiceCommand } from './voiceCommands.js';

test('normalizes case, punctuation and excess whitespace', () => {
  assert.equal(normalizeSpeech('  OPEN   Gesture Lab! '), 'open gesture lab');
});

test('matches supported routes and controls deterministically', () => {
  const cases = [['go home', 'home'], ['open gesture recognition lab', 'gesture'], ['show eye lab', 'eye'], ['open neural simulator', 'simulator'], ['open neuron comparison', 'comparison'], ['show performance analytics', 'analytics'], ['open about page', 'about'], ['enable accessibility mode', 'accessibility-on'], ['disable accessibility', 'accessibility-off'], ['start listening', 'start-listening'], ['stop listening', 'stop-listening'], ['help', 'help']];
  for (const [phrase, id] of cases) assert.equal(parseVoiceCommand(phrase)?.id, id, phrase);
});

test('does not map unknown speech to a route', () => {
  assert.equal(parseVoiceCommand('tell me a joke'), null);
});
