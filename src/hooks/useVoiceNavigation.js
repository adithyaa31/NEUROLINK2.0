import { useCallback, useEffect, useRef, useState } from 'react';
import { parseVoiceCommand } from '../services/voiceCommands.js';

export function useVoiceNavigation({ navigate, accessibility, setAccessibility }) {
  const recognitionRef = useRef(null);
  const activeRef = useRef(false);
  const seenRef = useRef(new Set());
  const navigateRef = useRef(navigate);
  const accessRef = useRef({ accessibility, setAccessibility });
  const [supported, setSupported] = useState(false);
  const [state, setState] = useState('idle');
  const [transcript, setTranscript] = useState('');
  const [lastCommand, setLastCommand] = useState('');
  const [message, setMessage] = useState('Activate the microphone to begin.');
  useEffect(() => { navigateRef.current = navigate; }, [navigate]);
  useEffect(() => { accessRef.current = { accessibility, setAccessibility }; }, [accessibility, setAccessibility]);

  const stop = useCallback(() => {
    activeRef.current = false;
    if (recognitionRef.current) {
      try { recognitionRef.current.abort(); } catch { /* already stopped */ }
    }
    setState(current => current === 'idle' ? current : 'idle');
    setMessage('Listening stopped.');
  }, []);

  const start = useCallback(() => {
    if (!recognitionRef.current) {
      setState('unsupported');
      setMessage('Speech recognition is not supported in this browser. Try a current version of Chrome or Edge.');
      return;
    }
    seenRef.current.clear();
    activeRef.current = true;
    setTranscript('');
    setState('starting');
    setMessage('Requesting microphone access and starting speech recognition…');
    try { recognitionRef.current.start(); }
    catch (error) {
      activeRef.current = false;
      setState('error');
      setMessage(error?.name === 'NotAllowedError' ? 'Microphone permission was denied. Allow microphone access in browser settings and try again.' : 'Could not start speech recognition. Check microphone permission and try again.');
    }
  }, []);

  useEffect(() => {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) { setSupported(false); setState('unsupported'); setMessage('Speech recognition is not supported in this browser.'); return; }
    setSupported(true);
    let recognition;
    try { recognition = new Recognition(); } catch { setState('unsupported'); setMessage('Speech recognition could not be initialized in this browser.'); return; }
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognitionRef.current = recognition;
    recognition.onstart = () => {
      if (!activeRef.current) { try { recognition.abort(); } catch {} return; }
      setState('listening');
      setMessage('Listening. Speak a command, then use Stop listening at any time.');
    };
    recognition.onresult = event => {
      setState('processing');
      const result = event.results[event.resultIndex];
      if (!result) return;
      const spoken = result[0]?.transcript?.trim() || '';
      if (spoken) setTranscript(spoken);
      if (!result.isFinal) { setState('listening'); return; }
      const key = spoken.toLowerCase().replace(/\s+/g, ' ').trim();
      if (!key || seenRef.current.has(key)) { setState('listening'); return; }
      seenRef.current.add(key);
      const intent = parseVoiceCommand(spoken);
      if (!intent) { setMessage(`I didn't recognize “${spoken}”. Say “help” to hear supported commands.`); setState('listening'); return; }
      setLastCommand(spoken);
      if (intent.path) { navigateRef.current(intent.path); setMessage(`Navigating to ${intent.id === 'home' ? 'home' : intent.id}.`); }
      else if (intent.id === 'accessibility-on' || intent.id === 'accessibility-off') {
        const enabled = intent.id === 'accessibility-on';
        accessRef.current.setAccessibility(enabled);
        setMessage(`Accessibility mode ${enabled ? 'enabled' : 'disabled'}.`);
      } else if (intent.id === 'stop-listening') { setMessage('Listening stopped by voice command.'); activeRef.current = false; try { recognitionRef.current?.stop(); } catch {} setState('idle'); return; }
      else if (intent.id === 'start-listening') { setMessage('Already listening.'); }
      else if (intent.id === 'help') { setMessage('Supported commands are listed below.'); }
      setState(activeRef.current?'listening':'idle');
    };
    recognition.onerror = event => {
      activeRef.current = false;
      setState('error');
      const errors = { 'not-allowed': 'Microphone permission was denied. Allow access in browser settings and try again.', 'service-not-allowed': 'The browser blocked its speech recognition service.', 'no-speech': 'No speech was detected. Activate the microphone and try again.', 'audio-capture': 'No microphone is available. Connect or enable a microphone and try again.', network: 'The browser speech recognition service is unavailable. Check your connection and try again.' };
      setMessage(errors[event.error] || `Speech recognition error: ${event.error}. You can retry.`);
    };
    recognition.onend = () => {
      if (activeRef.current) { activeRef.current = false; setState('error'); setMessage('Speech recognition disconnected. Activate the microphone to reconnect.'); }
    };
    return () => { activeRef.current = false; recognition.onstart = null; recognition.onresult = null; recognition.onerror = null; recognition.onend = null; try { recognition.abort(); } catch {} recognitionRef.current = null; };
  }, []);
  return { supported, state, transcript, lastCommand, message, start, stop };
}
