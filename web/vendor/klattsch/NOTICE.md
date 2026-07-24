# Vendored: klattsch

This directory contains a subset of [klattsch](https://github.com/tgies/klattsch)
v0.7.0 by Tony Gies, used under the MIT License (see LICENSE).

Only the offline synthesis engine is vendored (no AudioWorklet, no
English/Japanese text-to-phoneme helpers, no bank JSON files — bank data is
already inlined in `engine/banks/bundled.js`). We feed ARPABET directly, so
`pronounce.js` / `kana.js` and their `cmu-pronouncing-dictionary` dependency are
not needed.

Files:
  engine/index.js         public surface (re-exports)
  engine/dsp.js           low-level DSP
  engine/synth-core.js    FormantSynth (offline render)
  engine/phonemes.js      ARPABET parameter table
  engine/sequencer.js     phoneme-string -> schedule compiler (+ prosody directives)
  engine/wav.js           WAV encoder
  engine/banks/index.js   voice-bank registry
  engine/banks/bundled.js inlined voice-bank data (en + ja)
