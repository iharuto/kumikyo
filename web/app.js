"use strict";

/* ============================================================
 * Kumikyo 組响 — Web edition (ES module)
 * Sound sources: tones (abstract sine) and voice (klattsch, JA).
 * Modes: genjiko (match the 5-group pattern), taketori (same/different survival).
 * Ported from the v0 desktop app (script/kumikyo.py) to Web Audio.
 * No build step; plain HTML/JS, deployable on GitHub Pages.
 * ========================================================== */

import { playVoice, stopVoice, voiceWavBlob, sequenceWavBlob, sequenceWavBlobStrings } from "./voice.js";

// ---- Domain constants (same as v0) ----
const NOTE_FREQS = {
  C4: 261.626, "C#4": 277.183,
  D4: 293.665, "D#4": 311.127,
  E4: 329.628,
  F4: 349.228, "F#4": 369.994,
  G4: 391.995, "G#4": 415.305,
  A4: 440.0, "A#4": 466.164,
  B4: 493.883,
};
const NOTE_LIST = Object.keys(NOTE_FREQS);

// Constrained major scale (C-D-E-F-G) for subtle discrimination
const MAJOR_SCALE = [0, 2, 4, 5, 7];

const DIFFICULTY_LEVELS = {
  easy:      { n_positions: 3, max_edit_distance: 3, label: "Easy",      hint: "patterns differ noticeably" },
  normal:    { n_positions: 4, max_edit_distance: 2, label: "Normal",    hint: "patterns are somewhat similar" },
  hard:      { n_positions: 5, max_edit_distance: 1, label: "Hard",      hint: "patterns are very similar" },
  very_hard: { n_positions: 5, max_edit_distance: 1, label: "Very Hard", hint: "extremely similar; shown as text, not images" },
};

const NOTE_DURATION = 0.5;   // seconds per note
const NOTE_GAP = 0.05;       // gap between notes within a melody
const MELODY_GAP = 1.5;      // gap between successive melodies
const SAMPLE_RATE = 44100;

// ---- Genji-kō patterns (loaded from data/genji_ko.csv) ----
let GENJI_PATTERNS = {};   // rgs -> slug
let PATTERN_NAMES = [];    // list of rgs

async function loadGenjiPatterns() {
  const res = await fetch("data/genji_ko.csv");
  const text = await res.text();
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",");
  const iRgs = header.indexOf("rgs");
  const iSlug = header.indexOf("slug");
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",");
    const rgs = cols[iRgs];
    const slug = cols[iSlug];
    if (!rgs) continue;
    GENJI_PATTERNS[rgs] = slug;
    PATTERN_NAMES.push(rgs);
  }
}

// ---- Voice carriers (JA only, from data/phrases.json) ----
let VOICE_CARRIERS = []; // [{ arpa, text, label }]
const VOICE_BANK = "ja-hecko-2026";

async function loadVoiceCarriers() {
  try {
    const d = await (await fetch("data/phrases.json")).json();
    for (const p of d.phrases) {
      if (p.langs && p.langs.ja) VOICE_CARRIERS.push({ arpa: p.langs.ja.arpa, text: p.langs.ja.text, label: p.meaning });
    }
    for (const x of d.dishes || []) {
      if (x.lang === "ja") VOICE_CARRIERS.push({ arpa: x.arpa, text: x.text, label: x.dish_en });
    }
  } catch (e) {
    console.error("Failed to load voice carriers:", e);
  }
}

// ---- Music melodies (klattsch, from data/music.json) ----
let MUSIC = []; // [{ id, title, klatt }]
async function loadMusic() {
  try {
    const d = await (await fetch("data/music.json")).json();
    MUSIC = (d.melodies || []).map((m) => ({
      id: m.id, title: m.title, composer: m.composer || "", reference: m.reference || "", klatt: m.lines.join("\n"),
    }));
  } catch (e) {
    console.error("Failed to load music melodies:", e);
  }
}

const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// Credit line (title, composer, clickable reference link) for a music melody.
function musicCreditHtml(tune) {
  if (!tune) return "";
  let s = `♪ “${escapeHtml(tune.title)}”`;
  if (tune.composer) s += ` — ${escapeHtml(tune.composer)}`;
  if (tune.reference) s += ` · <a href="${escapeHtml(tune.reference)}" target="_blank" rel="noopener">reference</a>`;
  return s;
}

// ---- Seeded PRNG (mulberry32) ----
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const randInt = (rng, lo, hi) => lo + Math.floor(rng() * (hi - lo + 1)); // inclusive
const choice = (rng, arr) => arr[Math.floor(rng() * arr.length)];
function sample(rng, arr, k) {
  const pool = arr.slice();
  const out = [];
  for (let i = 0; i < k && pool.length; i++) {
    out.push(pool.splice(Math.floor(rng() * pool.length), 1)[0]);
  }
  return out;
}
function shuffle(rng, arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
const randomSeed = () => (Math.floor(Math.random() * 0x7fffffff)) >>> 0;

// ---- Melody generation (v0's generate_difficulty_melody) ----
function genBaseMelody(seed, nNotes) {
  const rng = mulberry32(seed);
  const melody = [];
  for (let i = 0; i < nNotes; i++) melody.push(choice(rng, MAJOR_SCALE));
  return melody;
}

// Variation for a group: change exactly one note to an adjacent scale step
function genVariationMelody(baseMelody, seed) {
  const rng = mulberry32(seed);
  const melody = baseMelody.slice();
  const n = melody.length;
  if (n === 0) return melody;
  const pos = randInt(rng, 0, n - 1);
  const original = melody[pos];
  const idx = MAJOR_SCALE.indexOf(original);
  let options = [];
  if (idx > 0) options.push(MAJOR_SCALE[idx - 1]);
  if (idx >= 0 && idx < MAJOR_SCALE.length - 1) options.push(MAJOR_SCALE[idx + 1]);
  if (options.length === 0) options = MAJOR_SCALE.filter((x) => x !== original);
  if (options.length) melody[pos] = choice(rng, options);
  return melody;
}

// Substitution distance between two patterns
function patternDistance(p1, p2) {
  if (p1.length !== p2.length) return Math.max(p1.length, p2.length);
  let d = 0;
  for (let i = 0; i < p1.length; i++) if (p1[i] !== p2[i]) d++;
  return d;
}

// ---- Stimulus for genjiko mode (v0's Stimulus) ----
function buildStimulus(difficulty, seed) {
  if (!DIFFICULTY_LEVELS[difficulty]) difficulty = "normal";
  const cfg = DIFFICULTY_LEVELS[difficulty];
  const nNotes = cfg.n_positions;
  const rng = mulberry32(seed);

  // Pick target
  const target = choice(rng, PATTERN_NAMES);
  const targetSlug = GENJI_PATTERNS[target];

  // Distractor candidates within the difficulty's max edit distance
  let compatible = PATTERN_NAMES.filter(
    (p) => p !== target && patternDistance(target, p) <= cfg.max_edit_distance
  );
  if (compatible.length < 5) {
    const extra = PATTERN_NAMES.filter((p) => p !== target && !compatible.includes(p));
    compatible = compatible.concat(extra);
  }
  let distractors = sample(rng, compatible, Math.min(5, compatible.length));
  while (distractors.length < 5) {
    const r = choice(rng, PATTERN_NAMES.filter((p) => p !== target));
    if (!distractors.includes(r)) distractors.push(r);
  }

  // 6 choices (target + 5 distractors), shuffled
  const allPatterns = shuffle(rng, [target, ...distractors]);
  const correctPosition = allPatterns.indexOf(target);

  const rootNote = choice(rng, NOTE_LIST);

  // Map each digit of the target to a group id in first-seen order
  const digits = target.split("");
  const digitToGroup = {};
  let g = 0;
  for (const d of digits) if (!(d in digitToGroup)) digitToGroup[d] = g++;
  const positionGroups = digits.map((d) => digitToGroup[d]);

  // Generate one melody per group (group 0 = base, others = 1-note variation)
  const uniqueGroups = [...new Set(positionGroups)];
  const base = genBaseMelody(seed + 0, nNotes);
  const groupMelodies = { 0: base };
  for (const gid of uniqueGroups) {
    if (gid !== 0) groupMelodies[gid] = genVariationMelody(base, seed + gid);
  }

  // Per-position melodies (semitone arrays)
  const positionMelodies = positionGroups.map((gid) => groupMelodies[gid]);

  return {
    difficulty, seed, nNotes,
    target, targetSlug,
    distractors, allPatterns, correctPosition,
    rootNote, positionGroups, positionMelodies,
  };
}

// ---- Image paths ----
const imagePath = (rgs) => `fig_genjiko/${rgs}_${GENJI_PATTERNS[rgs]}.png`;
const slugTitle = (rgs) =>
  (GENJI_PATTERNS[rgs] || "unknown").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

/* ============================================================
 * Audio: Web Audio API
 * All oscillators and timers are tracked so playback can be stopped
 * immediately on screen changes (fixes "audio keeps playing after Quit").
 * ========================================================== */
let audioCtx = null;
let activeOscillators = [];
let audioTimers = [];

function ac() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}
function trackTimer(fn, ms) {
  const id = setTimeout(() => {
    audioTimers = audioTimers.filter((t) => t !== id);
    fn();
  }, ms);
  audioTimers.push(id);
  return id;
}
function stopAllAudio() {
  audioTimers.forEach(clearTimeout);
  audioTimers = [];
  for (const osc of activeOscillators) {
    try { osc.onended = null; osc.stop(); } catch (e) { /* already stopped */ }
    try { osc.disconnect(); } catch (e) {}
  }
  activeOscillators = [];
  stopVoice(); // stop any klattsch voice playback too
  // Clear any "playing" highlights left on replay buttons
  document.querySelectorAll(".replay-positions button.playing").forEach((b) => b.classList.remove("playing"));
}

const semitoneFreq = (rootNote, semitone) =>
  (NOTE_FREQS[rootNote] || NOTE_FREQS.C4) * Math.pow(2, semitone / 12);

// Schedule one melody (semitone array) starting at ctx time. Returns end time.
function scheduleMelody(ctx, rootNote, melody, startTime, track) {
  let t = startTime;
  const fade = 0.02;
  for (const semi of melody) {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = semitoneFreq(rootNote, semi);
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.9, t + fade);
    gain.gain.setValueAtTime(0.9, t + NOTE_DURATION - fade);
    gain.gain.linearRampToValueAtTime(0, t + NOTE_DURATION);
    osc.connect(gain).connect(ctx.destination);
    osc.start(t);
    osc.stop(t + NOTE_DURATION);
    if (track) activeOscillators.push(osc);
    t += NOTE_DURATION + NOTE_GAP;
  }
  return t - NOTE_GAP;
}

const melodyDuration = (melody) =>
  melody.length * NOTE_DURATION + Math.max(0, melody.length - 1) * NOTE_GAP;

// Play a single melody live. onDone callback when finished.
function playSingleMelody(rootNote, melody, onDone) {
  stopAllAudio();
  const ctx = ac();
  const end = scheduleMelody(ctx, rootNote, melody, ctx.currentTime + 0.05, true);
  if (onDone) trackTimer(onDone, (end - ctx.currentTime) * 1000 + 60);
}

// Play several melodies back-to-back with MELODY_GAP between them.
function playMelodySequence(rootNote, melodies, onDone) {
  stopAllAudio();
  const ctx = ac();
  let t = ctx.currentTime + 0.1;
  for (let i = 0; i < melodies.length; i++) {
    const end = scheduleMelody(ctx, rootNote, melodies[i], t, true);
    t = end + MELODY_GAP;
  }
  const total = t - MELODY_GAP - ctx.currentTime;
  if (onDone) trackTimer(onDone, total * 1000 + 100);
}

// ---- Offline render to WAV ----
async function renderSequenceWav(rootNote, melodies) {
  let totalDur = 0.1;
  for (let i = 0; i < melodies.length; i++) {
    totalDur += melodyDuration(melodies[i]);
    if (i < melodies.length - 1) totalDur += MELODY_GAP;
  }
  totalDur += 0.2;
  const octx = new OfflineAudioContext(1, Math.ceil(SAMPLE_RATE * totalDur), SAMPLE_RATE);
  let t = 0.1;
  for (let i = 0; i < melodies.length; i++) {
    const end = scheduleMelody(octx, rootNote, melodies[i], t, false);
    t = end + MELODY_GAP;
  }
  const buffer = await octx.startRendering();
  return audioBufferToWav(buffer);
}

function audioBufferToWav(buffer) {
  const ch = buffer.getChannelData(0);
  const len = ch.length;
  const out = new DataView(new ArrayBuffer(44 + len * 2));
  const wr = (o, s) => { for (let i = 0; i < s.length; i++) out.setUint8(o + i, s.charCodeAt(i)); };
  wr(0, "RIFF"); out.setUint32(4, 36 + len * 2, true); wr(8, "WAVE");
  wr(12, "fmt "); out.setUint32(16, 16, true); out.setUint16(20, 1, true);
  out.setUint16(22, 1, true); out.setUint32(24, SAMPLE_RATE, true);
  out.setUint32(28, SAMPLE_RATE * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  wr(36, "data"); out.setUint32(40, len * 2, true);
  let o = 44;
  for (let i = 0; i < len; i++) {
    const s = Math.max(-1, Math.min(1, ch[i]));
    out.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    o += 2;
  }
  return new Blob([out.buffer], { type: "audio/wav" });
}

/* ============================================================
 * Stats (localStorage)
 * ========================================================== */
const STATS_KEY = "kumikyo_stats_v1";
const BEST_STREAK_KEY = "kumikyo_best_streak_v1";

function loadStats() {
  try { return JSON.parse(localStorage.getItem(STATS_KEY)) || []; }
  catch { return []; }
}
function recordTrial(rec) {
  const stats = loadStats();
  stats.push(rec);
  if (stats.length > 2000) stats.splice(0, stats.length - 2000);
  localStorage.setItem(STATS_KEY, JSON.stringify(stats));
}
function clearStats() {
  localStorage.removeItem(STATS_KEY);
  localStorage.removeItem(BEST_STREAK_KEY);
}
function loadBestStreak() { return parseInt(localStorage.getItem(BEST_STREAK_KEY) || "0", 10) || 0; }
function saveBestStreak(n) { localStorage.setItem(BEST_STREAK_KEY, String(n)); }

function aggregateStats(days = 14) {
  const stats = loadStats();
  const cutoff = Date.now() - days * 86400000;
  const byDay = {};
  for (const r of stats) {
    if (r.ts < cutoff) continue;
    const d = new Date(r.ts).toISOString().slice(0, 10);
    if (!byDay[d]) byDay[d] = { ok: 0, total: 0 };
    byDay[d].total++;
    if (r.correct) byDay[d].ok++;
  }
  return Object.entries(byDay).sort((a, b) => b[0].localeCompare(a[0]));
}

/* ============================================================
 * UI state & control
 * ========================================================== */
const state = {
  sound: "tones", // 'tones' | 'voice' | 'music'
  mode: "genjiko", // 'genjiko' | 'taketoriko' | 'survival'
  difficulty: "normal",
  stim: null,
  selectedIndex: null,
  startTs: null,
  taketori: null, // survival state
  match: null,    // taketori-kō state
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function showScreen(id) {
  stopAllAudio(); // stop any playback when leaving/entering a screen
  $$(".screen").forEach((s) => s.classList.remove("active"));
  $("#" + id).classList.add("active");
}

const SOUND_HINTS = {
  tones: "Abstract pure-sine melodies. Discriminate pitch patterns.",
  voice: "Robotic Klatt speech (Japanese). Discriminate speaking style — pitch, speed, vibrato, intonation.",
  music: "A sung Klatt melody. Detect when some sung vowels have been swapped (pitch & rhythm stay fixed).",
};
const MODE_HINTS = {
  genjiko: "Hear 5 sounds; identify which positions share a pattern and match the Genji-kō symbol.",
  taketoriko: "Hear one reference, then pick the one of 5 candidates that matches it.",
  survival: "Memorize the reference, then judge each following sound: same or different. One mistake ends the run.",
};
const PROSODY_HINTS = {
  easy: "large prosody difference",
  normal: "moderate prosody difference",
  hard: "subtle prosody difference",
  very_hard: "very subtle prosody difference",
};
const MUSIC_HINTS = {
  easy: "3 vowels swapped",
  normal: "2 vowels swapped",
  hard: "1 vowel swapped",
  very_hard: "1 vowel swapped to a near vowel",
};
function updateSoundHint() { $("#sound-hint").textContent = SOUND_HINTS[state.sound]; }
function updateModeHint() { $("#mode-hint").textContent = MODE_HINTS[state.mode]; }
function updateDifficultyHint() {
  if (state.sound === "music") {
    $("#difficulty-hint").textContent = `Difference: ${MUSIC_HINTS[state.difficulty]}`;
  } else if (state.sound === "voice") {
    $("#difficulty-hint").textContent = `Prosody: ${PROSODY_HINTS[state.difficulty]}`;
  } else {
    const cfg = DIFFICULTY_LEVELS[state.difficulty];
    $("#difficulty-hint").textContent = `Each melody: ${cfg.n_positions} notes / ${cfg.hint}`;
  }
}

function selectChip(groupSel, value, attr) {
  $$(groupSel + " .chip").forEach((b) => b.classList.toggle("active", b.dataset[attr] === value));
}

// ---- Home ----
function initHome() {
  $$("#sound-buttons .chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      selectChip("#sound-buttons", btn.dataset.sound, "sound");
      state.sound = btn.dataset.sound;
      updateSoundHint(); updateModeHint(); updateDifficultyHint();
    });
  });
  $$("#mode-buttons .chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      selectChip("#mode-buttons", btn.dataset.mode, "mode");
      state.mode = btn.dataset.mode;
      updateModeHint();
    });
  });
  $$("#difficulty-buttons .chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      selectChip("#difficulty-buttons", btn.dataset.difficulty, "difficulty");
      state.difficulty = btn.dataset.difficulty;
      updateDifficultyHint();
    });
  });
  updateSoundHint();
  updateModeHint();
  updateDifficultyHint();

  $("#btn-start").addEventListener("click", startSession);
  $("#btn-show-stats").addEventListener("click", showStats);
}

// ---- Start (dispatch by sound + mode) ----
function startSession() {
  ac(); // wake AudioContext on user gesture
  if (state.mode === "taketoriko") { startTaketoriKo(state.sound); return; }
  if (state.mode === "survival") { startTaketori(state.sound); return; } // 'tones' | 'voice' | 'music'
  startGenjiko(); // genji-kō (branches on state.sound inside)
}

/* ---------- Genji-kō mode ---------- */
// Distinct prosody preset per group (group 0 = base, others = palette entries).
// Difficulty controls how different the presets are.
function groupProsodyPresets(rng, difficulty, n) {
  const band = PROSODY_BANDS[difficulty] || PROSODY_BANDS.normal;
  const palette = [
    { pitch: 120 + band.pitch }, { pitch: 120 - band.pitch },
    { speedFactor: 1 + band.speed }, { speedFactor: 1 - band.speed },
    { vibratoMid: { depth: band.vibrato, atFraction: 0.5 } },
    { pitchMid: { hz: 120 + band.pitchMid, atFraction: 0.5 } },
    { pitchMid: { hz: 120 - band.pitchMid, atFraction: 0.5 } },
  ];
  const shuffled = shuffle(rng, palette);
  const presets = [{ ...VOICE_BASE_OPTS }];
  for (let i = 0; i < n - 1; i++) presets.push({ ...VOICE_BASE_OPTS, ...shuffled[i] });
  return presets;
}

// Voice Genji-kō: reuse the pattern selection, but each group is a prosody preset
// applied to one fixed JA carrier phrase (spoken 5 times).
function buildVoiceStimulus(difficulty, seed) {
  const stim = buildStimulus(difficulty, seed);
  const rng = mulberry32((seed ^ 0x5bd1e995) >>> 0);
  stim.isVoice = true;
  stim.carrier = choice(rng, VOICE_CARRIERS);
  const uniqueGroups = [...new Set(stim.positionGroups)];
  const presets = groupProsodyPresets(rng, difficulty, uniqueGroups.length);
  const groupToOpts = {};
  uniqueGroups.forEach((g, i) => { groupToOpts[g] = presets[i]; });
  stim.positionOpts = stim.positionGroups.map((g) => groupToOpts[g]);
  return stim;
}

// Music Genji-kō: play one JA melody 5 times; each group is a distinct
// vowel-swap variant (group 0 = original). Same group = identical rendition.
function buildMusicStimulus(difficulty, seed) {
  const stim = buildStimulus(difficulty, seed);
  const rng = mulberry32((seed ^ 0x27d4eb2f) >>> 0);
  stim.isMusic = true;
  stim.tune = choice(rng, MUSIC);
  const band = MUSIC_BANDS[difficulty] || MUSIC_BANDS.normal;
  const uniqueGroups = [...new Set(stim.positionGroups)];
  const groupToKlatt = {};
  uniqueGroups.forEach((g, i) => {
    groupToKlatt[g] = i === 0 ? stim.tune.klatt : swapVowels(stim.tune.klatt, band.count, band.near, seed + g * 101 + i).klatt;
  });
  stim.positionKlatt = stim.positionGroups.map((g) => groupToKlatt[g]);
  return stim;
}

// Play a list of items (opts for voice, full strings for music) back-to-back.
function playSeq(playItem, count, gapMs, onDone) {
  stopAllAudio();
  let i = 0;
  const step = () => {
    if (i >= count) { if (onDone) onDone(); return; }
    const idx = i++;
    playItem(idx, () => trackTimer(step, i < count ? gapMs : 0));
  };
  step();
}

// Genji-kō playback that branches on sound source.
function playGenjikoSequence(onDone) {
  const stim = state.stim;
  if (stim.isMusic) playSeq((i, cb) => playVoice(stim.positionKlatt[i], {}, cb), stim.positionKlatt.length, MELODY_GAP * 1000, onDone);
  else if (stim.isVoice) playSeq((i, cb) => playVoice(stim.carrier.arpa, stim.positionOpts[i], cb), stim.positionOpts.length, MELODY_GAP * 1000, onDone);
  else playMelodySequence(stim.rootNote, stim.positionMelodies, onDone);
}
function playGenjikoPosition(i, onDone) {
  const stim = state.stim;
  if (stim.isMusic) playVoice(stim.positionKlatt[i], {}, onDone);
  else if (stim.isVoice) playVoice(stim.carrier.arpa, stim.positionOpts[i], onDone);
  else playSingleMelody(stim.rootNote, stim.positionMelodies[i], onDone);
}

function startGenjiko() {
  const seed = randomSeed();
  state.stim =
    state.sound === "music" ? buildMusicStimulus(state.difficulty, seed)
    : state.sound === "voice" ? buildVoiceStimulus(state.difficulty, seed)
    : buildStimulus(state.difficulty, seed);
  state.selectedIndex = null;
  state.startTs = null;

  buildGrid();
  $("#btn-submit").disabled = true;
  const play = $("#btn-play-seq");
  play.disabled = false;
  const stim = state.stim;
  $("#play-credit").innerHTML = stim.isMusic ? musicCreditHtml(stim.tune) : "";
  if (stim.isMusic) {
    play.textContent = "▶ Play the melody 5×";
    $("#play-instruction").textContent = "One melody plays 5 times; some renditions swap sung vowels. Find which positions share the same rendition (same group), then pick the matching Genji-kō symbol.";
    $("#play-status").textContent = "Press Play.";
  } else if (stim.isVoice) {
    play.textContent = "▶ Play the 5 utterances";
    $("#play-instruction").textContent = "The same phrase is spoken 5 times. Find which positions share the same speaking style (same group), then pick the matching Genji-kō symbol.";
    $("#play-status").textContent = `Phrase: “${stim.carrier.text}”. Press Play to hear it.`;
  } else {
    play.textContent = "▶ Play the 5 melodies";
    $("#play-instruction").textContent = "Listen for which positions share the same melody (same group), then pick the matching Genji-kō symbol.";
    $("#play-status").textContent = "Press Play to hear the melodies.";
  }
  showScreen("screen-play");
}

function buildGrid() {
  const grid = $("#pattern-grid");
  grid.innerHTML = "";
  const textMode = state.difficulty === "very_hard";
  state.stim.allPatterns.forEach((rgs, i) => {
    const cell = document.createElement("div");
    cell.className = "cell" + (textMode ? " text-mode" : "");
    if (textMode) {
      cell.textContent = slugTitle(rgs);
    } else {
      const img = document.createElement("img");
      img.src = imagePath(rgs);
      img.alt = slugTitle(rgs);
      img.onerror = () => { cell.classList.add("text-mode"); cell.textContent = `${rgs}\n${slugTitle(rgs)}`; };
      cell.appendChild(img);
    }
    cell.addEventListener("click", () => selectCell(i, cell));
    grid.appendChild(cell);
  });
}

function selectCell(index, cell) {
  if (state.startTs === null) return; // can't choose before playback finishes
  $$("#pattern-grid .cell").forEach((c) => c.classList.remove("selected"));
  cell.classList.add("selected");
  state.selectedIndex = index;
  $("#btn-submit").disabled = false;
}

function initPlay() {
  $("#btn-play-seq").addEventListener("click", () => {
    const btn = $("#btn-play-seq");
    btn.disabled = true;
    $("#play-status").textContent = "♪ Playing… listen carefully.";
    playGenjikoSequence(() => {
      btn.disabled = false;
      state.startTs = performance.now();
      $("#play-status").textContent = "Done. Pick the symbol that matches the grouping.";
    });
  });
  $("#btn-submit").addEventListener("click", submitAnswer);
  $("#btn-quit").addEventListener("click", () => showScreen("screen-home"));
}

function submitAnswer() {
  if (state.selectedIndex === null) return;
  const stim = state.stim;
  const correct = state.selectedIndex === stim.correctPosition;
  const rtMs = state.startTs ? Math.round(performance.now() - state.startTs) : 0;

  recordTrial({
    ts: Date.now(), mode: stim.isMusic ? "music_genjiko" : stim.isVoice ? "voice_genjiko" : "genjiko",
    difficulty: stim.difficulty, seed: stim.seed,
    target: stim.target, choice: stim.allPatterns[state.selectedIndex], correct, rt_ms: rtMs,
  });

  showResult(correct);
}

function showResult(correct) {
  const stim = state.stim;
  const banner = $("#result-banner");
  banner.className = "banner " + (correct ? "ok" : "ng");
  banner.textContent = correct ? "✅ Correct!" : "❌ Incorrect";

  $("#result-correct").innerHTML = answerCardHtml(stim.target);
  const yoursWrap = $("#result-yours-wrap");
  if (correct) {
    yoursWrap.style.display = "none";
  } else {
    yoursWrap.style.display = "";
    $("#result-yours").innerHTML = answerCardHtml(stim.allPatterns[state.selectedIndex]);
  }

  // Reflect correctness on the (still-present) grid cells
  const cells = $$("#pattern-grid .cell");
  cells[stim.correctPosition]?.classList.add("correct");
  if (!correct) cells[state.selectedIndex]?.classList.add("wrong");

  buildReplay();
  showScreen("screen-result");
}

function answerCardHtml(rgs) {
  const textMode = state.difficulty === "very_hard";
  const inner = textMode
    ? ""
    : `<img src="${imagePath(rgs)}" alt="${slugTitle(rgs)}" onerror="this.style.display='none'">`;
  return `${inner}<div class="slug">${slugTitle(rgs)}</div><div class="rgs">${rgs}</div>`;
}

function buildReplay() {
  const stim = state.stim;
  const box = $("#replay-positions");
  box.innerHTML = "";
  stim.positionGroups.forEach((grp, i) => {
    const btn = document.createElement("button");
    btn.innerHTML = `Pos ${i + 1} <span class="grp">grp ${grp + 1}</span>`;
    btn.addEventListener("click", () => {
      $$("#replay-positions button").forEach((b) => b.classList.remove("playing"));
      btn.classList.add("playing");
      playGenjikoPosition(i, () => btn.classList.remove("playing"));
    });
    box.appendChild(btn);
  });
}

function initResult() {
  $("#btn-replay-all").addEventListener("click", () => {
    const btn = $("#btn-replay-all");
    btn.disabled = true;
    playGenjikoSequence(() => (btn.disabled = false));
  });
  $("#btn-download-wav").addEventListener("click", async () => {
    const btn = $("#btn-download-wav");
    const stim = state.stim;
    btn.disabled = true;
    btn.textContent = "Rendering…";
    try {
      const blob = stim.isMusic
        ? sequenceWavBlobStrings(stim.positionKlatt, MELODY_GAP * 1000)
        : stim.isVoice
        ? sequenceWavBlob(stim.carrier.arpa, stim.positionOpts, MELODY_GAP * 1000)
        : await renderSequenceWav(stim.rootNote, stim.positionMelodies);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `kumikyo_${stim.isVoice ? "voice_" : ""}${stim.target}_${stim.seed}.wav`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      btn.disabled = false;
      btn.textContent = "Download WAV";
    }
  });
  $("#btn-next").addEventListener("click", startGenjiko);
  $("#btn-home").addEventListener("click", () => showScreen("screen-home"));
}

/* ---------- Taketori mode (tones or voice) ----------
 * A reference sound plays once at the start. Then single sounds play one at a
 * time; for each, the player decides whether it is the SAME as the reference or
 * DIFFERENT. P(different) = 0.8. First mistake ends the run.
 *
 * kind 'tones': reference/test are sine melodies (1-note variation).
 * kind 'voice': reference/test are klattsch utterances of a fixed JA carrier;
 *   "different" applies a random prosody change (pitch / speed / mid vibrato /
 *   mid pitch), magnitude scaled by difficulty, at a random (seeded) position.
 */
const TAKETORI_P_DIFFERENT = 0.8;

// Prosody difference magnitudes per difficulty
const PROSODY_BANDS = {
  easy:      { pitch: 35, speed: 0.25, vibrato: 16, pitchMid: 45 },
  normal:    { pitch: 22, speed: 0.16, vibrato: 12, pitchMid: 30 },
  hard:      { pitch: 13, speed: 0.10, vibrato: 8,  pitchMid: 20 },
  very_hard: { pitch: 8,  speed: 0.06, vibrato: 5,  pitchMid: 13 },
};
const VOICE_BASE_OPTS = { bank: VOICE_BANK, pitch: 120, speedFactor: 1 };

// ---- Music mode: swap sung vowels to adjust difficulty ----
const ARPA_VOWELS = ["AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"];
// Acoustically-near vowels (for the subtle very_hard swaps)
const VOWEL_NEAR = {
  IY:["IH","EY"], IH:["IY","EH"], EH:["IH","AE"], AE:["EH","AH"],
  AH:["AE","AO","ER"], AO:["AH","OW"], OW:["AO","UH"], UH:["OW","UW"],
  UW:["UH","OW"], ER:["AH","UH"], EY:["IY","EH"], AY:["AA","EH"],
  AW:["AA","AO"], OY:["AO","OW"], AA:["AH","AO"],
};
// count = how many vowels to change; near = swap to an acoustically similar vowel
const MUSIC_BANDS = {
  easy:      { count: 3, near: false },
  normal:    { count: 2, near: false },
  hard:      { count: 1, near: false },
  very_hard: { count: 1, near: true },
};
function pickSwapVowel(rng, cur, near) {
  if (near && VOWEL_NEAR[cur]) return choice(rng, VOWEL_NEAR[cur]);
  return choice(rng, ARPA_VOWELS.filter((v) => v !== cur));
}
// Swap `count` vowels in a klattsch melody string (preserving all whitespace,
// pitch and rhythm directives). Returns { klatt, changes }.
function swapVowels(klatt, count, near, seed) {
  const rng = mulberry32(seed);
  const toks = klatt.split(/(\s+)/); // keep whitespace tokens
  const vIdx = [];
  toks.forEach((t, i) => { if (ARPA_VOWELS.includes(t)) vIdx.push(i); });
  if (!vIdx.length) return { klatt, changes: 0 };
  const chosen = shuffle(rng, vIdx).slice(0, Math.min(count, vIdx.length));
  for (const i of chosen) toks[i] = pickSwapVowel(rng, toks[i], near);
  return { klatt: toks.join(""), changes: chosen.length };
}

// Build a random prosody variant (the "different" case). Returns { opts, label }.
function makeProsodyVariant(rng, difficulty) {
  const band = PROSODY_BANDS[difficulty] || PROSODY_BANDS.normal;
  const axis = choice(rng, ["globalPitch", "speed", "vibratoMid", "pitchMid"]);
  const sign = rng() < 0.5 ? -1 : 1;
  const pos = 0.2 + rng() * 0.6; // random insert position 0.2..0.8
  switch (axis) {
    case "globalPitch":
      return { opts: { pitch: 120 + sign * band.pitch }, label: `overall pitch ${sign > 0 ? "higher" : "lower"}` };
    case "speed": {
      const factor = 1 + sign * band.speed;
      return { opts: { speedFactor: factor }, label: `${factor > 1 ? "slower" : "faster"} speed` };
    }
    case "vibratoMid":
      return { opts: { vibratoMid: { depth: band.vibrato, atFraction: pos } }, label: "mid-sentence vibrato" };
    case "pitchMid":
    default:
      return { opts: { pitchMid: { hz: 120 + sign * band.pitchMid, atFraction: pos } }, label: `mid-sentence pitch ${sign > 0 ? "up" : "down"}` };
  }
}

function startTaketori(kind) {
  const t = { kind, phase: "reference", round: 0, streak: 0, best: loadBestStreak(), answered: false, isSame: null, variantLabel: null };
  if (kind === "music") {
    if (!MUSIC.length) {
      alert("Music data failed to load.");
      showScreen("screen-home");
      return;
    }
    t.tune = choice(mulberry32(randomSeed()), MUSIC);
    t.melody = t.tune.klatt;
    t.testMelody = null;
  } else if (kind === "voice") {
    if (!VOICE_CARRIERS.length) {
      alert("Voice data failed to load.");
      showScreen("screen-home");
      return;
    }
    t.carrier = choice(mulberry32(randomSeed()), VOICE_CARRIERS);
    t.baseOpts = { ...VOICE_BASE_OPTS };
    t.refOpts = { ...VOICE_BASE_OPTS };
    t.testOpts = null;
  } else {
    const nNotes = DIFFICULTY_LEVELS[state.difficulty].n_positions;
    const refSeed = randomSeed();
    t.nNotes = nNotes;
    t.reference = genBaseMelody(refSeed, nNotes);
    t.rootNote = choice(mulberry32(refSeed + 777), NOTE_LIST);
    t.test = null;
  }
  state.taketori = t;

  $("#tk-best").textContent = t.best;
  $("#tk-round").textContent = 0;
  $("#tk-streak").textContent = 0;
  $("#tk-feedback").style.display = "none";
  $("#btn-tk-same").disabled = true;
  $("#btn-tk-diff").disabled = true;
  const play = $("#btn-tk-play");
  play.disabled = false;
  play.textContent = "▶ Play reference";
  $("#tk-credit").innerHTML = kind === "music" ? musicCreditHtml(t.tune) : "";
  $("#tk-status").textContent =
    kind === "music"
      ? "Listen to the reference melody and memorize its sung vowels."
      : kind === "voice"
      ? `Reference phrase: “${t.carrier.text}”. Listen and memorize how it is spoken.`
      : "Listen to the reference melody and memorize it.";
  showScreen("screen-taketori");
}

function newTaketoriRound() {
  const t = state.taketori;
  t.phase = "round";
  t.round += 1;
  t.answered = false;

  const seed = randomSeed();
  const rng = mulberry32(seed);
  const isDifferent = rng() < TAKETORI_P_DIFFERENT;
  t.isSame = !isDifferent;

  if (t.kind === "music") {
    if (isDifferent) {
      const band = MUSIC_BANDS[state.difficulty] || MUSIC_BANDS.normal;
      const { klatt, changes } = swapVowels(t.melody, band.count, band.near, seed);
      t.testMelody = klatt;
      t.variantLabel = `${changes} vowel${changes === 1 ? "" : "s"} changed`;
    } else {
      t.testMelody = t.melody;
      t.variantLabel = null;
    }
  } else if (t.kind === "voice") {
    if (isDifferent) {
      const variant = makeProsodyVariant(rng, state.difficulty);
      t.testOpts = { ...t.baseOpts, ...variant.opts };
      t.variantLabel = variant.label;
    } else {
      t.testOpts = { ...t.baseOpts };
      t.variantLabel = null;
    }
  } else {
    t.test = isDifferent ? genVariationMelody(t.reference, seed) : t.reference.slice();
  }

  $("#tk-round").textContent = t.round;
  $("#tk-streak").textContent = t.streak;
  $("#tk-feedback").style.display = "none";
  $("#btn-tk-same").disabled = true;
  $("#btn-tk-diff").disabled = true;
  const play = $("#btn-tk-play");
  play.disabled = false;
  play.textContent = "▶ Play sound";
  $("#tk-status").textContent = "Play the sound — same as the reference, or different?";
}

// Play the reference or the test item for the current taketori state.
function taketoriPlay(which, onDone) {
  const t = state.taketori;
  if (t.kind === "music") {
    const klatt = which === "reference" ? t.melody : t.testMelody;
    playVoice(klatt, {}, onDone);
  } else if (t.kind === "voice") {
    const opts = which === "reference" ? t.refOpts : t.testOpts;
    playVoice(t.carrier.arpa, opts, onDone);
  } else {
    const mel = which === "reference" ? t.reference : t.test;
    playSingleMelody(t.rootNote, mel, onDone);
  }
}

function playTaketori() {
  const t = state.taketori;
  const btn = $("#btn-tk-play");
  btn.disabled = true;
  if (t.phase === "reference") {
    $("#tk-status").textContent = "♪ Reference…";
    taketoriPlay("reference", () => newTaketoriRound());
  } else {
    $("#tk-status").textContent = "♪ …";
    taketoriPlay("test", () => {
      btn.disabled = false;
      $("#tk-status").textContent = "Same as the reference, or different?";
      if (!t.answered) {
        $("#btn-tk-same").disabled = false;
        $("#btn-tk-diff").disabled = false;
      }
    });
  }
}

function answerTaketori(userSaysSame) {
  const t = state.taketori;
  if (t.answered) return;
  t.answered = true;
  $("#btn-tk-same").disabled = true;
  $("#btn-tk-diff").disabled = true;

  const correct = userSaysSame === t.isSame;
  const modeName = { music: "music_taketori", voice: "voice_taketori", tones: "taketori" }[t.kind];
  recordTrial({
    ts: Date.now(), mode: modeName, difficulty: state.difficulty,
    round: t.round, isSame: t.isSame, answer: userSaysSame ? "same" : "different", correct,
  });

  if (correct) {
    t.streak += 1;
    $("#tk-streak").textContent = t.streak;
    const fb = $("#tk-feedback");
    fb.className = "banner ok";
    fb.textContent = "✅ Correct!";
    fb.style.display = "";
    trackTimer(newTaketoriRound, 900);
  } else {
    taketoriGameOver();
  }
}

function taketoriGameOver() {
  const t = state.taketori;
  const best = Math.max(t.streak, loadBestStreak());
  saveBestStreak(best);
  t.best = best;
  $("#go-streak").textContent = t.streak;
  $("#go-best").textContent = best;
  let reveal = `That sound was ${t.isSame ? "the SAME as" : "DIFFERENT from"} the reference.`;
  if (!t.isSame && t.variantLabel) reveal += ` (${t.variantLabel})`;
  $("#go-reveal").textContent = reveal;
  buildTaketoriReplay();
  showScreen("screen-gameover");
}

function buildTaketoriReplay() {
  const box = $("#go-replay");
  box.innerHTML = "";
  [["Reference", "reference"], ["That sound", "test"]].forEach(([label, which]) => {
    const btn = document.createElement("button");
    btn.textContent = "▶ " + label;
    btn.addEventListener("click", () => {
      $$("#go-replay button").forEach((b) => b.classList.remove("playing"));
      btn.classList.add("playing");
      taketoriPlay(which, () => btn.classList.remove("playing"));
    });
    box.appendChild(btn);
  });
}

function initTaketori() {
  $("#btn-tk-play").addEventListener("click", playTaketori);
  $("#btn-tk-same").addEventListener("click", () => answerTaketori(true));
  $("#btn-tk-diff").addEventListener("click", () => answerTaketori(false));
  $("#btn-tk-quit").addEventListener("click", () => showScreen("screen-home"));
  $("#btn-go-again").addEventListener("click", () => startTaketori(state.taketori ? state.taketori.kind : "tones"));
  $("#btn-go-home").addEventListener("click", () => showScreen("screen-home"));
}

/* ---------- Taketori-kō mode (match-to-reference) ----------
 * One reference plays; 5 candidates are shown (one identical to the reference,
 * 4 variations). Pick the matching candidate. Works for tones / voice / music.
 */
function buildMatchStimulus(kind, difficulty, seed) {
  const rng = mulberry32(seed);
  const st = { kind, selectedIndex: null, submitted: false };
  let distractors, matchItem;

  if (kind === "music") {
    if (!MUSIC.length) return null;
    st.tune = choice(rng, MUSIC);
    st.reference = st.tune.klatt;
    const band = MUSIC_BANDS[difficulty] || MUSIC_BANDS.normal;
    distractors = Array.from({ length: 4 }, () => swapVowels(st.reference, band.count, band.near, (Math.floor(rng() * 1e9)) >>> 0).klatt);
    matchItem = st.reference;
  } else if (kind === "voice") {
    if (!VOICE_CARRIERS.length) return null;
    st.carrier = choice(rng, VOICE_CARRIERS);
    st.baseOpts = { ...VOICE_BASE_OPTS };
    st.reference = { ...VOICE_BASE_OPTS };
    distractors = Array.from({ length: 4 }, () => {
      const v = makeProsodyVariant(rng, difficulty);
      return { ...st.baseOpts, ...v.opts };
    });
    matchItem = { ...st.baseOpts };
  } else {
    const nNotes = DIFFICULTY_LEVELS[difficulty].n_positions;
    st.rootNote = choice(rng, NOTE_LIST);
    st.reference = genBaseMelody((Math.floor(rng() * 1e9)) >>> 0, nNotes);
    distractors = Array.from({ length: 4 }, () => genVariationMelody(st.reference, (Math.floor(rng() * 1e9)) >>> 0));
    matchItem = st.reference.slice();
  }

  const all = distractors.concat([matchItem]); // match is at index 4
  const order = shuffle(rng, [0, 1, 2, 3, 4]);
  st.candidates = order.map((i) => all[i]);
  st.correctIndex = order.indexOf(4);
  return st;
}

function playMatchItem(item, onDone) {
  const s = state.match;
  if (s.kind === "music") playVoice(item, {}, onDone);
  else if (s.kind === "voice") playVoice(s.carrier.arpa, item, onDone);
  else playSingleMelody(s.rootNote, item, onDone);
}

function buildCandidates() {
  const box = $("#tko-candidates");
  box.innerHTML = "";
  state.match.candidates.forEach((item, i) => {
    const btn = document.createElement("button");
    btn.className = "cand";
    btn.textContent = `▶ Candidate ${i + 1}`;
    btn.addEventListener("click", () => {
      if (!state.match.submitted) {
        $$("#tko-candidates .cand").forEach((b) => b.classList.remove("selected"));
        btn.classList.add("selected");
        state.match.selectedIndex = i;
        $("#btn-tko-submit").disabled = false;
      }
      $$("#tko-candidates .cand").forEach((b) => b.classList.remove("playing"));
      btn.classList.add("playing");
      playMatchItem(item, () => btn.classList.remove("playing"));
    });
    box.appendChild(btn);
  });
}

function startTaketoriKo(kind) {
  const s = buildMatchStimulus(kind, state.difficulty, randomSeed());
  if (!s) { alert(`${kind} data failed to load.`); showScreen("screen-home"); return; }
  state.match = s;

  $("#tko-credit").innerHTML = kind === "music" ? musicCreditHtml(s.tune) : "";
  const refLabel =
    kind === "music" ? `Reference melody: “${s.tune.title}”. `
    : kind === "voice" ? `Reference phrase: “${s.carrier.text}”. `
    : "";
  $("#tko-status").textContent = refLabel + "Play the reference, then pick the matching candidate.";
  buildCandidates();
  $("#btn-tko-ref").disabled = false;
  $("#btn-tko-submit").disabled = true;
  $("#tko-feedback").style.display = "none";
  $("#tko-after").style.display = "none";
  showScreen("screen-taketoriko");
}

function submitMatch() {
  const s = state.match;
  if (s.selectedIndex === null || s.submitted) return;
  s.submitted = true;
  const correct = s.selectedIndex === s.correctIndex;
  const modeName = { music: "music_taketoriko", voice: "voice_taketoriko", tones: "taketoriko" }[s.kind];
  recordTrial({ ts: Date.now(), mode: modeName, difficulty: state.difficulty, correct });

  const cands = $$("#tko-candidates .cand");
  cands[s.correctIndex]?.classList.add("correct");
  if (!correct) cands[s.selectedIndex]?.classList.add("wrong");
  const fb = $("#tko-feedback");
  fb.className = "banner " + (correct ? "ok" : "ng");
  fb.textContent = correct ? "✅ Correct!" : "❌ Incorrect — the highlighted candidate matched the reference.";
  fb.style.display = "";
  $("#btn-tko-submit").disabled = true;
  $("#tko-after").style.display = "flex";
}

function initTaketoriKo() {
  $("#btn-tko-ref").addEventListener("click", () => {
    const b = $("#btn-tko-ref");
    b.disabled = true;
    playMatchItem(state.match.reference, () => (b.disabled = false));
  });
  $("#btn-tko-submit").addEventListener("click", submitMatch);
  $("#btn-tko-quit").addEventListener("click", () => showScreen("screen-home"));
  $("#btn-tko-next").addEventListener("click", () => startTaketoriKo(state.sound));
  $("#btn-tko-home").addEventListener("click", () => showScreen("screen-home"));
}

// ---- Stats dialog ----
function showStats() {
  const rows = aggregateStats(14);
  const body = $("#stats-body");
  const best = loadBestStreak();
  let html = `<p class="hint">Taketori best streak: <strong>${best}</strong></p>`;
  if (rows.length === 0) {
    html += `<p class="hint">No records yet. Start a session!</p>`;
  } else {
    html += rows
      .map(([d, s]) => {
        const pct = s.total ? Math.round((s.ok / s.total) * 100) : 0;
        return `<div class="stat-row"><span>${d}</span><span>${s.ok}/${s.total} (${pct}%)</span></div>
                <div class="stat-bar" style="width:${pct}%"></div>`;
      })
      .join("");
  }
  body.innerHTML = html;
  $("#stats-dialog").showModal();
}

function initStatsDialog() {
  $("#btn-close-stats").addEventListener("click", () => $("#stats-dialog").close());
  $("#btn-clear-stats").addEventListener("click", () => {
    if (confirm("Clear all stats history?")) {
      clearStats();
      $("#stats-dialog").close();
    }
  });
}

// ---- Boot ----
async function main() {
  try {
    await loadGenjiPatterns();
    $("#pattern-count").textContent = PATTERN_NAMES.length;
  } catch (e) {
    $("#pattern-count").textContent = "load failed";
    console.error("Failed to load Genji-kō patterns:", e);
  }
  await loadVoiceCarriers();
  await loadMusic();
  initHome();
  initPlay();
  initResult();
  initTaketori();
  initTaketoriKo();
  initStatsDialog();
}

main();
