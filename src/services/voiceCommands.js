const intents = [
  { id: 'home', path: '/', patterns: [/^(?:open|go to|navigate to|show) (?:the )?(?:home|dashboard|overview)(?: page)?$/, /^(?:go )?home$/] },
  { id: 'gesture', path: '/gesture', patterns: [/^(?:open|go to|navigate to|show|start) (?:the )?(?:gesture|hand)(?: recognition)?(?: lab)?(?: page)?$/] },
  { id: 'eye', path: '/eye-voice', patterns: [/^(?:open|go to|navigate to|show) (?:the )?(?:eye|eye interaction|eye and voice|voice)(?: lab| page)?$/] },
  { id: 'simulator', path: '/simulator', patterns: [/^(?:open|go to|navigate to|show) (?:the )?(?:neural |neuron )?(?:pathway )?simulator(?: page)?$/] },
  { id: 'comparison', path: '/comparison', patterns: [/^(?:open|go to|navigate to|show) (?:the )?(?:neuron )?comparison(?: page)?$/] },
  { id: 'analytics', path: '/analytics', patterns: [/^(?:open|go to|navigate to|show) (?:the )?(?:performance )?analytics(?: page)?$/] },
  { id: 'about', path: '/about', patterns: [/^(?:open|go to|navigate to|show) (?:the )?about(?: page)?$/] },
  { id: 'accessibility-on', patterns: [/^(?:enable|turn on|activate) accessibility(?: mode)?$/] },
  { id: 'accessibility-off', patterns: [/^(?:disable|turn off|deactivate) accessibility(?: mode)?$/] },
  { id: 'start-listening', patterns: [/^(?:start|begin|resume) listening$/] },
  { id: 'stop-listening', patterns: [/^(?:stop|end) listening$/] },
  { id: 'help', patterns: [/^help$/, /^(?:show|list) (?:me )?(?:the )?(?:voice )?commands$/, /^what can i say$/] },
];

export function normalizeSpeech(value) {
  return String(value ?? '').toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim();
}

export function parseVoiceCommand(value) {
  const transcript = normalizeSpeech(value);
  if (!transcript) return null;
  return intents.find(intent => intent.patterns.some(pattern => pattern.test(transcript))) || null;
}

export const voiceCommandLabels = [
  'Open home / go home', 'Open gesture lab', 'Open eye lab', 'Open neural simulator',
  'Open neuron comparison', 'Open analytics', 'Open about page', 'Enable accessibility',
  'Disable accessibility', 'Start listening', 'Stop listening', 'Help',
];
