# 🎵 Kumikyo — 組响 — Auditory Pattern Trainer

An auditory pattern-recognition trainer inspired by *Genji-kō* (源氏香), the incense-matching game of *Kumikō* (組香).
Listen to short melodies and figure out which ones share the same pattern.

> 🌐 **Playable in the browser** — a static site with no build step, deployed via [GitHub Pages](https://iharuto.github.io/kumikyo/).

## 🎮 How to play

### Modes

- **Genji-kō**: Five melodies play in sequence. Work out which positions share a melody (the same "group") and pick the matching Genji-kō symbol from 6 choices.
- **Taketori**: A reference melody plays once at the start. Then single sounds play one at a time — for each, decide whether it is **the same** as the reference or **different**. Keep going as long as you're correct; a single mistake ends the run. Different sounds appear 80% of the time. Your streak and all-time best are tracked.

### Genji-kō flow

1. Choose a **mode** (Genji-kō) and a **difficulty**
2. Press **▶ Play the 5 melodies** and listen
3. Identify which of the 5 positions play the **same melody** (same group)
4. Pick the Genji-kō symbol whose grouping matches, then **Submit**
5. On the result screen you can:
   - **Replay each position (1–5) or the full sequence** (with group labels)
   - **Download the generated audio as a WAV**
6. Results are stored locally (`localStorage`); "View recent stats" shows a per-day summary

### Difficulty

| Difficulty | Notes per melody | Notes |
|------------|------------------|-------|
| Easy | 3 | Patterns differ noticeably |
| Normal | 4 | Somewhat similar |
| Hard | 5 | Very similar |
| Very Hard | 5 | Extremely similar; symbols shown as text |

Melodies use a constrained C-D-E-F-G major scale of pure sine tones, and groups differ by **a single adjacent scale step**. Headphones recommended.

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
```

## 🏗️ Structure (plain HTML/JS, no dependencies)

```
index.html                   # UI (Setup → Play → Result, plus Taketori / Game Over)
web/style.css                # styles
web/app.js                   # stimulus generation + Web Audio synthesis + stats
data/genji_ko.csv            # 52 Genji-kō patterns (rgs / slug / heights)
fig_genjiko/                 # symbol images (52 PNGs)
.github/workflows/pages.yml  # GitHub Pages auto-deploy
.nojekyll                    # disable Jekyll processing
```

### How it works

- **Stimulus generation**: Each digit of a Genji-kō pattern is read as a "group" (e.g. `12121` → groups `01010`). Positions in the same group get an identical melody; other groups get a one-note variation.
- **Audio synthesis**: Web Audio API `OscillatorNode` (sine) with a gain envelope. WAV export is rendered via `OfflineAudioContext`.
- **Reproducibility**: A seeded PRNG (mulberry32) regenerates the same puzzle and audio from the same seed.
- **Stats**: Stored in `localStorage` (no server required).



## 📄 Contact

haruka.ij [at] gmail.com

## 🙏 Acknowledgments

- **Genji-kō** and **Taketori-kō** — incense-comparison games of *Kumikō* (組香) that inspired the modes
- Genji-kō symbols from the chapters of *The Tale of Genji*
- Melodies in Music mode composed by **Susumu Hirasawa** (user-transcribed as klattsch arrangements for auditory training; see `data/music.json` for per-track references)
- [klattsch](https://github.com/tgies/klattsch) — Tony Gies (MIT)
- Web Audio API

---

<div align="center">

**🎵 Train your ear, expand your mind 🧠**

</div>
