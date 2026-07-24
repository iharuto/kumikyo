"use strict";

/* ============================================================
 * Kumikyo 組响 — Web版 (フェーズ1: genjiko モード)
 * v0 の script/kumikyo.py の刺激生成ロジックを Web Audio に移植。
 * ビルド不要の素の HTML/JS。GitHub Pages で配信可能。
 * ========================================================== */

// ---- ドメイン定数（v0 と同じ）----
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

// 拘束された長音階 (C-D-E-F-G) — 半音差の微妙な識別のため
const MAJOR_SCALE = [0, 2, 4, 5, 7];

const DIFFICULTY_LEVELS = {
  easy:      { n_positions: 3, max_edit_distance: 3, label: "やさしい", hint: "パターンの違いが大きめ" },
  normal:    { n_positions: 4, max_edit_distance: 2, label: "ふつう",   hint: "パターンはやや似ている" },
  hard:      { n_positions: 5, max_edit_distance: 1, label: "むずかしい", hint: "パターンは非常に似ている" },
  very_hard: { n_positions: 5, max_edit_distance: 1, label: "激ムズ",   hint: "極めて似ている・図ではなくテキスト表示" },
};

const NOTE_DURATION = 0.5;   // 1音あたり秒
const NOTE_GAP = 0.05;       // メロディ内の音間ギャップ
const MELODY_GAP = 1.5;      // 5メロディ間のギャップ
const SAMPLE_RATE = 44100;

// ---- 源氏香パターン (data/genji_ko.csv から読み込み) ----
let GENJI_PATTERNS = {};   // rgs -> slug
let PATTERN_NAMES = [];    // rgs のリスト

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

// ---- seed 付き PRNG (mulberry32) ----
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const randInt = (rng, lo, hi) => lo + Math.floor(rng() * (hi - lo + 1)); // 両端含む
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

// ---- メロディ生成 (v0 の generate_difficulty_melody 相当) ----
function genBaseMelody(seed, nNotes) {
  const rng = mulberry32(seed);
  const melody = [];
  for (let i = 0; i < nNotes; i++) melody.push(choice(rng, MAJOR_SCALE));
  return melody;
}

// 組 groupId 用のメロディ: 基準から1音だけ隣接ステップに変える
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

// パターン間の置換距離
function patternDistance(p1, p2) {
  if (p1.length !== p2.length) return Math.max(p1.length, p2.length);
  let d = 0;
  for (let i = 0; i < p1.length; i++) if (p1[i] !== p2[i]) d++;
  return d;
}

// ---- 刺激生成 (v0 の Stimulus 相当) ----
function buildStimulus(difficulty, seed) {
  if (!DIFFICULTY_LEVELS[difficulty]) difficulty = "normal";
  const cfg = DIFFICULTY_LEVELS[difficulty];
  const nNotes = cfg.n_positions;
  const rng = mulberry32(seed);

  // ターゲット選択
  const target = choice(rng, PATTERN_NAMES);
  const targetSlug = GENJI_PATTERNS[target];

  // 難易度制約内の distractor 候補
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

  // 6択 (ターゲット + distractor 5) をシャッフル
  const allPatterns = shuffle(rng, [target, ...distractors]);
  const correctPosition = allPatterns.indexOf(target);

  const rootNote = choice(rng, NOTE_LIST);

  // ターゲットの各桁 -> 組ID (出現順に 0,1,2,...)
  const digits = target.split("");
  const digitToGroup = {};
  let g = 0;
  for (const d of digits) if (!(d in digitToGroup)) digitToGroup[d] = g++;
  const positionGroups = digits.map((d) => digitToGroup[d]);

  // 各組のメロディを生成 (組0=基準, 他=1音違い)
  const uniqueGroups = [...new Set(positionGroups)];
  const groupMelodies = {};
  for (const gid of uniqueGroups) {
    if (gid === 0) groupMelodies[gid] = genBaseMelody(seed + 0, nNotes);
  }
  const base = groupMelodies[0] || genBaseMelody(seed + 0, nNotes);
  for (const gid of uniqueGroups) {
    if (gid !== 0) groupMelodies[gid] = genVariationMelody(base, seed + gid);
  }

  // 位置ごとの実メロディ (半音配列)
  const positionMelodies = positionGroups.map((gid) => groupMelodies[gid]);

  return {
    difficulty, seed, nNotes,
    target, targetSlug,
    distractors, allPatterns, correctPosition,
    rootNote, positionGroups, positionMelodies,
  };
}

// ---- 画像パス ----
const imagePath = (rgs) => `fig_genjiko/${rgs}_${GENJI_PATTERNS[rgs]}.png`;
const slugTitle = (rgs) =>
  (GENJI_PATTERNS[rgs] || "unknown").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

/* ============================================================
 * Audio: Web Audio API
 * ========================================================== */
let audioCtx = null;
function ac() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}
const semitoneFreq = (rootNote, semitone) =>
  (NOTE_FREQS[rootNote] || NOTE_FREQS.C4) * Math.pow(2, semitone / 12);

// 1つのメロディ(半音配列)を live 再生。startTime は AudioContext 時刻。返り値=終了時刻
function scheduleMelody(ctx, rootNote, melody, startTime) {
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
    t += NOTE_DURATION + NOTE_GAP;
  }
  return t - NOTE_GAP;
}

// メロディの実時間(秒)
const melodyDuration = (melody) =>
  melody.length * NOTE_DURATION + Math.max(0, melody.length - 1) * NOTE_GAP;

// 単一位置メロディの再生。onDone コールバック
function playSingleMelody(stim, melody, onDone) {
  const ctx = ac();
  const end = scheduleMelody(ctx, stim.rootNote, melody, ctx.currentTime + 0.05);
  if (onDone) setTimeout(onDone, (end - ctx.currentTime) * 1000 + 60);
}

// 5メロディを順に再生。onDone は最後に呼ばれる
function playFullSequence(stim, onDone) {
  const ctx = ac();
  let t = ctx.currentTime + 0.1;
  for (let i = 0; i < stim.positionMelodies.length; i++) {
    const end = scheduleMelody(ctx, stim.rootNote, stim.positionMelodies[i], t);
    t = end + MELODY_GAP;
  }
  const total = t - MELODY_GAP - ctx.currentTime;
  if (onDone) setTimeout(onDone, total * 1000 + 100);
}

// ---- オフラインレンダリングして WAV 生成 ----
async function renderSequenceWav(stim) {
  // 全体長を計算
  let totalDur = 0.1;
  for (let i = 0; i < stim.positionMelodies.length; i++) {
    totalDur += melodyDuration(stim.positionMelodies[i]);
    if (i < stim.positionMelodies.length - 1) totalDur += MELODY_GAP;
  }
  totalDur += 0.2;
  const octx = new OfflineAudioContext(1, Math.ceil(SAMPLE_RATE * totalDur), SAMPLE_RATE);
  let t = 0.1;
  for (let i = 0; i < stim.positionMelodies.length; i++) {
    const end = scheduleMelody(octx, stim.rootNote, stim.positionMelodies[i], t);
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
 * 成績 (localStorage)
 * ========================================================== */
const STATS_KEY = "kumikyo_stats_v1";
function loadStats() {
  try { return JSON.parse(localStorage.getItem(STATS_KEY)) || []; }
  catch { return []; }
}
function recordTrial(rec) {
  const stats = loadStats();
  stats.push(rec);
  // 直近1000件のみ保持
  if (stats.length > 1000) stats.splice(0, stats.length - 1000);
  localStorage.setItem(STATS_KEY, JSON.stringify(stats));
}
function clearStats() { localStorage.removeItem(STATS_KEY); }

// 日付ごとに集計
function aggregateStats(days = 14) {
  const stats = loadStats();
  const now = Date.now();
  const cutoff = now - days * 86400000;
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
 * UI 状態 & 制御
 * ========================================================== */
const state = {
  mode: "genjiko",
  difficulty: "normal",
  stim: null,
  selectedIndex: null,
  startTs: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function showScreen(id) {
  $$(".screen").forEach((s) => s.classList.remove("active"));
  $("#" + id).classList.add("active");
}

function updateDifficultyHint() {
  const cfg = DIFFICULTY_LEVELS[state.difficulty];
  $("#difficulty-hint").textContent = `各メロディ ${cfg.n_positions} 音 / ${cfg.hint}`;
}

// ---- ホーム画面 ----
function initHome() {
  $$("#mode-buttons .chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      $$("#mode-buttons .chip").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
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
  updateDifficultyHint();

  $("#btn-start").addEventListener("click", startSession);
  $("#btn-show-stats").addEventListener("click", showStats);
}

// ---- セッション開始 ----
function startSession() {
  const seed = (Math.floor(Math.random() * 0x7fffffff)) >>> 0;
  state.stim = buildStimulus(state.difficulty, seed);
  state.selectedIndex = null;
  state.startTs = null;

  buildGrid();
  $("#btn-submit").disabled = true;
  $("#play-status").textContent = "「再生」を押してメロディを聞いてください。";
  showScreen("screen-play");
  ac(); // ユーザー操作でAudioContextを起こす
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
  if (state.startTs === null) return; // 再生前は選べない
  $$("#pattern-grid .cell").forEach((c) => c.classList.remove("selected"));
  cell.classList.add("selected");
  state.selectedIndex = index;
  $("#btn-submit").disabled = false;
}

function initPlay() {
  $("#btn-play-seq").addEventListener("click", () => {
    const btn = $("#btn-play-seq");
    btn.disabled = true;
    $("#play-status").textContent = "♪ 再生中… よく聞いてください。";
    playFullSequence(state.stim, () => {
      btn.disabled = false;
      state.startTs = performance.now();
      $("#play-status").textContent = "再生完了。同じ組の並びを図から選んでください。";
    });
  });
  $("#btn-submit").addEventListener("click", submitAnswer);
  $("#btn-quit").addEventListener("click", () => showScreen("screen-home"));
}

// ---- 回答 ----
function submitAnswer() {
  if (state.selectedIndex === null) return;
  const stim = state.stim;
  const correct = state.selectedIndex === stim.correctPosition;
  const rtMs = state.startTs ? Math.round(performance.now() - state.startTs) : 0;

  recordTrial({
    ts: Date.now(),
    mode: state.mode,
    difficulty: stim.difficulty,
    seed: stim.seed,
    target: stim.target,
    choice: stim.allPatterns[state.selectedIndex],
    correct,
    rt_ms: rtMs,
  });

  showResult(correct);
}

// ---- 答え合わせ画面 ----
function showResult(correct) {
  const stim = state.stim;
  const banner = $("#result-banner");
  banner.className = "banner " + (correct ? "ok" : "ng");
  banner.textContent = correct ? "✅ 正解！" : "❌ 不正解";

  $("#result-correct").innerHTML = answerCardHtml(stim.target);
  const yoursWrap = $("#result-yours-wrap");
  if (correct) {
    yoursWrap.style.display = "none";
  } else {
    yoursWrap.style.display = "";
    $("#result-yours").innerHTML = answerCardHtml(stim.allPatterns[state.selectedIndex]);
  }

  // グリッドにも正誤を反映（プレイ画面のグリッドは残っている）
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
    btn.innerHTML = `位置${i + 1} <span class="grp">組${grp + 1}</span>`;
    btn.addEventListener("click", () => {
      $$("#replay-positions button").forEach((b) => b.classList.remove("playing"));
      btn.classList.add("playing");
      playSingleMelody(stim, melody, () => btn.classList.remove("playing"));
    });
    box.appendChild(btn);
  });
}

function initResult() {
  $("#btn-replay-all").addEventListener("click", () => {
    const btn = $("#btn-replay-all");
    btn.disabled = true;
    playFullSequence(state.stim, () => (btn.disabled = false));
  });
  $("#btn-download-wav").addEventListener("click", async () => {
    const btn = $("#btn-download-wav");
    btn.disabled = true;
    btn.textContent = "生成中…";
    try {
      const blob = await renderSequenceWav(state.stim);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `kumikyo_${state.stim.target}_${state.stim.seed}.wav`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      btn.disabled = false;
      btn.textContent = "WAVをダウンロード";
    }
  });
  $("#btn-next").addEventListener("click", startSession);
  $("#btn-home").addEventListener("click", () => showScreen("screen-home"));
}

// ---- 成績ダイアログ ----
function showStats() {
  const rows = aggregateStats(14);
  const body = $("#stats-body");
  if (rows.length === 0) {
    body.innerHTML = `<p class="hint">まだ記録がありません。セッションを始めましょう！</p>`;
  } else {
    body.innerHTML = rows
      .map(([d, s]) => {
        const pct = s.total ? Math.round((s.ok / s.total) * 100) : 0;
        return `<div class="stat-row"><span>${d}</span><span>${s.ok}/${s.total} (${pct}%)</span></div>
                <div class="stat-bar" style="width:${pct}%"></div>`;
      })
      .join("");
  }
  $("#stats-dialog").showModal();
}

function initStatsDialog() {
  $("#btn-close-stats").addEventListener("click", () => $("#stats-dialog").close());
  $("#btn-clear-stats").addEventListener("click", () => {
    if (confirm("成績履歴を消去しますか？")) {
      clearStats();
      $("#stats-dialog").close();
    }
  });
}

// ---- 起動 ----
async function main() {
  try {
    await loadGenjiPatterns();
    $("#pattern-count").textContent = PATTERN_NAMES.length;
  } catch (e) {
    $("#pattern-count").textContent = "読み込み失敗";
    console.error("源氏香パターンの読み込みに失敗:", e);
  }
  initHome();
  initPlay();
  initResult();
  initStatsDialog();
}

main();
