"use strict";

/* ============================================================
 * Kumikyo 組响 — Web edition
 * Modes: genjiko (match the 5-group pattern), taketori (same/different survival)
 * Ported from the v0 desktop app (script/kumikyo.py) to Web Audio.
 * No build step; plain HTML/JS, deployable on GitHub Pages.
 * ========================================================== */

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
  mode: "genjiko",
  difficulty: "normal",
  stim: null,
  selectedIndex: null,
  startTs: null,
  taketori: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function showScreen(id) {
  stopAllAudio(); // stop any playback when leaving/entering a screen
  $$(".screen").forEach((s) => s.classList.remove("active"));
  $("#" + id).classList.add("active");
}

const MODE_HINTS = {
  genjiko: "Hear 5 melodies; identify which positions share a melody and match the Genji-kō symbol.",
  taketori: "Compare two melodies and answer Same or Different. Keep your streak alive — one mistake ends the run.",
};
function updateModeHint() { $("#mode-hint").textContent = MODE_HINTS[state.mode]; }
function updateDifficultyHint() {
  const cfg = DIFFICULTY_LEVELS[state.difficulty];
  $("#difficulty-hint").textContent = `Each melody: ${cfg.n_positions} notes / ${cfg.hint}`;
}

// ---- Home ----
function initHome() {
  $$("#mode-buttons .chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      $$("#mode-buttons .chip").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      updateModeHint();
    });
  });
  $$("#difficulty-buttons .chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$("#difficulty-buttons .chip").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.difficulty = btn.dataset.difficulty;
      updateDifficultyHint();
    });
  });
  updateModeHint();
  updateDifficultyHint();

  $("#btn-start").addEventListener("click", startSession);
  $("#btn-show-stats").addEventListener("click", showStats);
}

// ---- Start (dispatch by mode) ----
function startSession() {
  ac(); // wake AudioContext on user gesture
  if (state.mode === "taketori") startTaketori();
  else startGenjiko();
}

/* ---------- Genji-kō mode ---------- */
function startGenjiko() {
  state.stim = buildStimulus(state.difficulty, randomSeed());
  state.selectedIndex = null;
  state.startTs = null;

  buildGrid();
  $("#btn-submit").disabled = true;
  $("#btn-play-seq").disabled = false;
  $("#play-status").textContent = "Press Play to hear the melodies.";
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
    playMelodySequence(state.stim.rootNote, state.stim.positionMelodies, () => {
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
    ts: Date.now(), mode: "genjiko", difficulty: stim.difficulty, seed: stim.seed,
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
  stim.positionMelodies.forEach((melody, i) => {
    const btn = document.createElement("button");
    const grp = stim.positionGroups[i];
    btn.innerHTML = `Pos ${i + 1} <span class="grp">grp ${grp + 1}</span>`;
    btn.addEventListener("click", () => {
      $$("#replay-positions button").forEach((b) => b.classList.remove("playing"));
      btn.classList.add("playing");
      playSingleMelody(stim.rootNote, melody, () => btn.classList.remove("playing"));
    });
    box.appendChild(btn);
  });
}

function initResult() {
  $("#btn-replay-all").addEventListener("click", () => {
    const btn = $("#btn-replay-all");
    btn.disabled = true;
    playMelodySequence(state.stim.rootNote, state.stim.positionMelodies, () => (btn.disabled = false));
  });
  $("#btn-download-wav").addEventListener("click", async () => {
    const btn = $("#btn-download-wav");
    btn.disabled = true;
    btn.textContent = "Rendering…";
    try {
      const blob = await renderSequenceWav(state.stim.rootNote, state.stim.positionMelodies);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `kumikyo_${state.stim.target}_${state.stim.seed}.wav`;
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

/* ---------- Taketori mode ---------- */
function startTaketori() {
  state.taketori = { round: 0, streak: 0, best: loadBestStreak(), reference: null, test: null, isSame: null, rootNote: null, answered: false };
  $("#tk-best").textContent = state.taketori.best;
  showScreen("screen-taketori");
  newTaketoriRound();
}

function newTaketoriRound() {
  const t = state.taketori;
  t.round += 1;
  t.answered = false;

  const cfg = DIFFICULTY_LEVELS[state.difficulty];
  const nNotes = cfg.n_positions;
  const seed = randomSeed();
  const rng = mulberry32(seed);

  t.reference = genBaseMelody(seed, nNotes);
  t.isSame = rng() < 0.5;
  t.rootNote = choice(rng, NOTE_LIST);
  t.test = t.isSame ? t.reference.slice() : genVariationMelody(t.reference, seed + 1);

  $("#tk-round").textContent = t.round;
  $("#tk-streak").textContent = t.streak;
  $("#tk-feedback").style.display = "none";
  $("#btn-tk-same").disabled = true;
  $("#btn-tk-diff").disabled = true;
  $("#btn-tk-play").disabled = false;
  $("#tk-status").textContent = "Press Play to hear the two melodies.";
}

function playTaketori() {
  const t = state.taketori;
  const btn = $("#btn-tk-play");
  btn.disabled = true;
  $("#tk-status").textContent = "♪ Reference, then comparison…";
  playMelodySequence(t.rootNote, [t.reference, t.test], () => {
    btn.disabled = false;
    $("#tk-status").textContent = "Same or different?";
    if (!t.answered) {
      $("#btn-tk-same").disabled = false;
      $("#btn-tk-diff").disabled = false;
    }
  });
}

function answerTaketori(userSaysSame) {
  const t = state.taketori;
  if (t.answered) return;
  t.answered = true;
  $("#btn-tk-same").disabled = true;
  $("#btn-tk-diff").disabled = true;

  const correct = userSaysSame === t.isSame;
  recordTrial({
    ts: Date.now(), mode: "taketori", difficulty: state.difficulty,
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
  $("#go-reveal").textContent = `The two melodies were ${t.isSame ? "the SAME" : "DIFFERENT"}.`;
  buildTaketoriReplay();
  showScreen("screen-gameover");
}

function buildTaketoriReplay() {
  const t = state.taketori;
  const box = $("#go-replay");
  box.innerHTML = "";
  const items = [
    { label: "Reference", melody: t.reference },
    { label: "Comparison", melody: t.test },
  ];
  items.forEach(({ label, melody }) => {
    const btn = document.createElement("button");
    btn.textContent = "▶ " + label;
    btn.addEventListener("click", () => {
      $$("#go-replay button").forEach((b) => b.classList.remove("playing"));
      btn.classList.add("playing");
      playSingleMelody(t.rootNote, melody, () => btn.classList.remove("playing"));
    });
    box.appendChild(btn);
  });
}

function initTaketori() {
  $("#btn-tk-play").addEventListener("click", playTaketori);
  $("#btn-tk-same").addEventListener("click", () => answerTaketori(true));
  $("#btn-tk-diff").addEventListener("click", () => answerTaketori(false));
  $("#btn-tk-quit").addEventListener("click", () => showScreen("screen-home"));
  $("#btn-go-again").addEventListener("click", startTaketori);
  $("#btn-go-home").addEventListener("click", () => showScreen("screen-home"));
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
  initHome();
  initPlay();
  initResult();
  initTaketori();
  initStatsDialog();
}

main();
