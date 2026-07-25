# 🎵 Kumikyo — 組响 — Auditory Pattern Trainer

An auditory pattern-recognition trainer inspired by an incense-matching game, *Kumikō* (組香).
Listen to short sounds and figure out which ones share the same pattern.

> 🌐 **Playable in the browser** — a static site with no build step, deployed via [GitHub Pages](https://iharuto.github.io/kumikyo/).

## 🎮 How to play

Pick a **Sound source**, a **Mode**, and a **Difficulty**, then start.

### Sound sources

- **🎵 Tones** — abstract pure-sine melodies. Discriminate pitch patterns.
- **🗣️ Voice** — robotic Klatt speech (Japanese, via klattsch). Discriminate *speaking style*: pitch, speed, vibrato, mid-sentence intonation.
- **🎼 Music** — a sung Klatt melody (arrangements of songs by Susumu Hirasawa). Detect swapped sung vowels; pitch & rhythm stay fixed.

### Modes

- **Genji-kō**: Five sounds play in sequence. Work out which positions share the same sound (the same "group") and pick the matching Genji-kō symbol from 6 choices.
- **Taketori-kō**: One **reference** plays, then **5 candidates** are shown — exactly one is identical to the reference, the other four are variations. Pick the matching candidate.
- **Survival**: A reference plays once; memorize it. Then single sounds play one at a time — for each, decide whether it is the **same** as the reference or **different**. Keep going while you're correct; a single mistake ends the run. Different sounds appear ~80% of the time; your streak and all-time best are tracked.

All three modes work with all three sound sources.

<details><summary>Genji-kō flow</summary>

1. Choose a **Sound**, set **Mode → Genji-kō**, and a **Difficulty**
2. Press **▶ Play** — the 5 sounds play in sequence
3. Identify which of the 5 positions share the same sound (same group)
4. Pick the Genji-kō symbol whose grouping matches, then **Submit**
5. On the result screen you can **replay each position (1–5) or the full sequence**, and **download the audio as a WAV**

</details>

<details><summary>Taketori-kō flow</summary>

1. Choose a **Sound**, set **Mode → Taketori-kō**, and a **Difficulty**
2. Press **▶ Play reference**
3. Play each of the **5 candidates** and pick the one identical to the reference, then **Submit**
4. The candidate that matched is highlighted

</details>

<details><summary>Survival flow</summary>

1. Choose a **Sound**, set **Mode → Survival**, and a **Difficulty**
2. Press **▶ Play reference** once and memorize it
3. Each round, press **▶ Play sound** and answer **Same** / **Different**
4. Correct → continue (streak +1). Wrong → Game Over (best streak saved). Roughly 80% of rounds are different.

</details>

### Difficulty

Difficulty controls how subtle the difference is, per sound source:

| Sound | Easy → Very Hard |
|-------|------------------|
| Tones | 3 / 4 / 5 notes per melody; groups differ by a single adjacent scale step |
| Voice | prosody difference: large → very subtle (pitch / speed / vibrato / intonation) |
| Music | swapped vowels: 3 → 2 → 1 → 1 (to an acoustically near vowel) |

Headphones recommended.

Results are stored locally (`localStorage`); "View recent stats" shows a per-day summary and your Survival best streak.

## 🚀 Deployment (GitHub Pages)

Pushing to `main` triggers a GitHub Actions workflow that builds and deploys automatically.

One-time setup: in the repo, go to **Settings → Pages → Source: GitHub Actions**.
After that, every push updates the public URL (e.g. `https://<user>.github.io/<repo>/`).

## 🖥️ Run locally

`file://` won't work because of `fetch`, so serve over a local HTTP server:

```bash
# from the repo root
python3 -m http.server 8000
# → open http://localhost:8000/
# voice tuning tool: http://localhost:8000/voice-lab.html
```

## 🏗️ Structure (plain HTML/JS, no dependencies)

```
index.html                     # UI (Setup → Genji-kō / Taketori-kō / Survival screens)
voice-lab.html                 # prosody tuning tool for klattsch (pitch/speed/vibrato/bank)
web/style.css                  # styles
web/app.js                     # game logic + Web Audio synthesis + stats
web/voice.js                   # klattsch wrapper (offline synth, prosody, WAV export)
web/vendor/klattsch/           # vendored klattsch engine subset (MIT)
data/genji_ko.csv              # 52 Genji-kō patterns (rgs / slug / heights)
data/phrases.json              # ARPABET phrases (Voice-mode carriers)
data/music.json                # sung melodies for Music mode (with references)
fig_genjiko/                   # symbol images (52 PNGs)
.github/workflows/pages.yml    # GitHub Pages auto-deploy
.nojekyll                      # disable Jekyll processing
```

### How it works

- **Grouping (Genji-kō)**: Each digit of a Genji-kō pattern is a "group" (e.g. `12121` → groups `01010`). Positions in the same group get an identical sound; other groups get a variation.
- **Tones**: Web Audio `OscillatorNode` (pure sine) on a constrained C-D-E-F-G scale; groups differ by a single adjacent step.
- **Voice / Music**: the vendored [klattsch](https://github.com/tgies/klattsch) Klatt synthesizer renders ARPABET offline to a buffer. Voice varies prosody via directives (pitch `b`, rate `r`, vibrato `v`, mid-sentence inserts); Music swaps sung vowels while keeping pitch/rhythm.
- **Reproducibility**: a seeded PRNG (mulberry32) regenerates the same puzzle and audio from the same seed.
- **Stats**: stored in `localStorage` (no server required).
- **WAV export** is rendered via `OfflineAudioContext` (tones) or klattsch offline render (voice/music).

## 📄 Contact

haruka.ij [at] gmail.com

## 🙏 Acknowledgments

- **Genji-kō** and **Taketori-kō** — incense-comparison games of *Kumikō* (組香) that inspired the modes
   - Genji-kō symbols from the chapters of *The Tale of Genji*
   - Taketori-kō symbols from the chapters of *The Tale of Taketori*
- Melodies in Music mode composed by **Susumu Hirasawa** (user-transcribed as klattsch arrangements for auditory training; see `data/music.json` for per-track references)
- [klattsch](https://github.com/tgies/klattsch) — Tony Gies (MIT)
- Web Audio API
