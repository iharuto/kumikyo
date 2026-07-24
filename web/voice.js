"use strict";

/* ============================================================
 * voice.js — thin wrapper around the vendored klattsch engine.
 * Offline Klatt formant synthesis of ARPABET strings, with prosody
 * (pitch / speed / vibrato) control. Renders to a Float32Array, then plays
 * via Web Audio and/or exports a WAV. No AudioWorklet needed. ES module.
 *
 * klattsch compact directives used here (see sequencer.js):
 *   b<Hz>  base pitch (F0)      r<ms>  rate = ms per phoneme
 *   p<ms>  pause                v<Hz>  vibrato depth   w<Hz> vibrato rate
 *   [bank=name] voice bank
 * ========================================================== */

import { compileString, FormantSynth, encodeWav, banks } from "./vendor/klattsch/engine/index.js";

const SR = 48000;
export const DEFAULT_RATE = 110; // klattsch default ms/phoneme
export const VOICE_BANKS = banks.list(); // ['ja-hecko-2026','ja-mokhtari-2000','klatt1980-en']

const PUNCT_MS = { ",": 100, ";": 200, ".": 300 };
const isPhonemeTok = (t) => /^[A-Z]/.test(t); // phonemes are uppercase-leading

let _ctx = null;
function ctx() {
  if (!_ctx) _ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (_ctx.state === "suspended") _ctx.resume();
  return _ctx;
}

/**
 * Linear time-stretch: scale every duration directive by `factor` so speech and
 * pauses stretch together (klattsch does NOT scale pauses with rate on its own).
 * Scales inline r<ms>, p<ms>, and punctuation pauses (,;.).
 */
export function timeStretch(arpa, factor) {
  if (!factor || factor === 1) return arpa;
  return arpa
    .split(/\s+/)
    .map((tok) => {
      let m;
      if ((m = tok.match(/^r(\d+(?:\.\d+)?)$/))) return "r" + Math.round(+m[1] * factor);
      if ((m = tok.match(/^p(\d+(?:\.\d+)?)$/))) return "p" + Math.round(+m[1] * factor);
      if (tok in PUNCT_MS) return "p" + Math.round(PUNCT_MS[tok] * factor);
      return tok;
    })
    .join(" ");
}

/** Insert a directive token (e.g. "v12") right before the n-th phoneme (0-based). */
export function insertAtPhoneme(arpa, n, directive) {
  const toks = arpa.split(/\s+/).filter(Boolean);
  const out = [];
  let count = 0;
  for (const tok of toks) {
    if (isPhonemeTok(tok)) {
      if (count === n) out.push(directive);
      count++;
    }
    out.push(tok);
  }
  if (n >= count) out.push(directive);
  return out.join(" ");
}

const countPhonemes = (arpa) => arpa.split(/\s+/).filter(isPhonemeTok).length;

/**
 * Compose a klattsch source string: prosody directives + ARPABET carrier.
 * opts:
 *   bank        voice bank name
 *   pitch       base F0 (Hz)
 *   speedFactor time-stretch multiplier (1 = normal, >1 slower, <1 faster);
 *               scales rate AND pauses together
 *   rate        explicit ms/phoneme (overrides speedFactor's rate prefix)
 *   vibratoDepth / vibratoRate     whole-sentence vibrato
 *   vibratoMid  { depth, atFraction } insert vibrato partway (mid-sentence toggle)
 *   pitchMid    { hz, atFraction }    insert a pitch shift (b<Hz>) partway
 */
export function buildUtterance(arpa, opts = {}) {
  let carrier = arpa;
  const factor = opts.speedFactor || 1;
  if (factor !== 1) carrier = timeStretch(carrier, factor);

  // Mid-sentence inserts. Inserted directives are not phonemes, so they don't
  // shift each other's phoneme indices — order is irrelevant.
  if (opts.vibratoMid && opts.vibratoMid.depth) {
    const n = countPhonemes(carrier);
    const frac = opts.vibratoMid.atFraction ?? 0.5;
    const at = Math.max(0, Math.min(n - 1, Math.round(frac * n)));
    carrier = insertAtPhoneme(carrier, at, `v${opts.vibratoMid.depth}`);
  }
  if (opts.pitchMid && opts.pitchMid.hz) {
    const n = countPhonemes(carrier);
    const frac = opts.pitchMid.atFraction ?? 0.5;
    const at = Math.max(0, Math.min(n - 1, Math.round(frac * n)));
    carrier = insertAtPhoneme(carrier, at, `b${Math.round(opts.pitchMid.hz)}`);
  }

  const parts = [];
  if (opts.bank) parts.push(`[bank=${opts.bank}]`);
  if (opts.pitch) parts.push(`b${Math.round(opts.pitch)}`);
  if (opts.rate) parts.push(`r${Math.round(opts.rate)}`);
  else if (factor !== 1) parts.push(`r${Math.round(DEFAULT_RATE * factor)}`);
  if (opts.vibratoDepth) parts.push(`v${opts.vibratoDepth}`);
  if (opts.vibratoRate) parts.push(`w${opts.vibratoRate}`);
  parts.push(carrier);
  return parts.join(" ");
}

/** Render to a mono Float32Array. Returns { buf, totalMs, sampleRate, source, warnings }. */
export function renderFloat(arpa, opts = {}) {
  const source = buildUtterance(arpa, opts);
  const { schedule, totalMs, warnings } = compileString(source);
  const synth = new FormantSynth({ sampleRate: SR, schedule });
  const buf = new Float32Array(Math.ceil((totalMs + 200) * SR / 1000));
  synth.process(buf);
  return { buf, totalMs, sampleRate: SR, source, warnings };
}

// ---- Playback (tracked so it can be stopped on screen changes) ----
let _srcNode = null;
export function stopVoice() {
  if (_srcNode) {
    try { _srcNode.onended = null; _srcNode.stop(); } catch (e) {}
    _srcNode = null;
  }
}

/** Play an utterance. onDone fires when playback ends. Returns approx duration (ms). */
export function playVoice(arpa, opts = {}, onDone) {
  stopVoice();
  const c = ctx();
  const { buf, totalMs } = renderFloat(arpa, opts);
  const ab = c.createBuffer(1, buf.length, SR);
  ab.copyToChannel(buf, 0);
  const src = c.createBufferSource();
  src.buffer = ab;
  src.connect(c.destination);
  src.onended = () => { if (_srcNode === src) _srcNode = null; if (onDone) onDone(); };
  _srcNode = src;
  src.start();
  return totalMs;
}

/** Encode an utterance to a WAV Blob (embeds the source string as a comment). */
export function voiceWavBlob(arpa, opts = {}) {
  const { buf, source } = renderFloat(arpa, opts);
  const { bytes } = encodeWav(buf, SR, { metadata: { software: "Kumikyo", comment: source } });
  return new Blob([bytes], { type: "audio/wav" });
}
