# 🎵 Kumikyo — 組响 — 音のパターン認識トレーナー

源氏香（源氏香 / 組香）に着想を得た、聴覚パターン認識のトレーニングアプリです。
5つのメロディを聞き、**同じ組（＝同じメロディ）の並び**を聞き分けて、対応する源氏香の図を当てます。

> 🌐 **ブラウザで遊べます** — ビルド不要の静的サイトで、GitHub Pages で配信しています。

## 🎮 遊び方

### モード

- **源氏香 (Genji-kō)**: 5つのメロディを聞き、同じ組の並びを当てて源氏香の図を6択から選ぶ
- **竹取 (Taketori)**: reference メロディと比較メロディを聞き、**Same / Different** を答え続けるサバイバル。1問でも間違えると終了。連続正解(streak)と最高記録(best)を記録

### 源氏香モードの流れ

1. **モード**（源氏香）と**難易度**を選ぶ
2. 「セッション開始」→「▶ 5つのメロディを再生」で音を聞く
3. 5つのメロディのうち、**同じ音**が鳴る位置の組み合わせを聞き取る
4. その並びに対応する源氏香の図を **6択**から選んで「回答する」
5. 答え合わせ画面で：
   - **各位置(位置1〜5)／全体のメロディを聴き直し**（組ラベル付き）
   - 生成された音を **WAV でダウンロード**
6. 成績は端末内（`localStorage`）に保存され、「最近の成績を見る」で日別集計を確認できます

### 難易度

| 難易度 | 1メロディの音数 | 内容 |
|--------|----------------|------|
| やさしい | 3音 | パターンの違いが大きめ |
| ふつう | 4音 | やや似ている |
| むずかしい | 5音 | 非常に似ている |
| 激ムズ | 5音 | 極めて似ている・図ではなくテキスト表示 |

各メロディは C-D-E-F-G の長音階に限定した純正弦波で、組の違いは **1音だけ隣接ステップに変化**します。ヘッドフォン推奨。

## 🚀 デプロイ（GitHub Pages）

`main` へ push すると GitHub Actions が自動でビルド・デプロイします。

初回のみ、リポジトリの **Settings → Pages → Source: GitHub Actions** を選択してください。
以降は push だけで公開URL（例: `https://<user>.github.io/<repo>/`）が更新されます。

## 🖥️ ローカルで動かす

`file://` では `fetch` が動かないため、簡易サーバ経由で開きます。

```bash
# リポジトリ直下で
python3 -m http.server 8000
# → ブラウザで http://localhost:8000/
```

## 🏗️ 構成（素の HTML/JS・依存なし）

```
index.html                   # 画面（設定 → プレイ → 答え合わせ の3画面）
web/style.css                # スタイル
web/app.js                   # 刺激生成ロジック + Web Audio 合成 + 成績保存
data/genji_ko.csv            # 源氏香 52 パターン (rgs / slug / heights)
fig_genjiko/                 # 各パターンの図 (52 枚 PNG)
.github/workflows/pages.yml  # GitHub Pages 自動デプロイ
.nojekyll                    # Jekyll 処理を無効化
```

### 仕組み

- **刺激生成**: 源氏香パターンの各桁を「組」として解釈（例 `12121` → 組 `01010`）。同じ組の位置には同一メロディ、異なる組には1音違いのメロディを割り当てる
- **音合成**: Web Audio API の `OscillatorNode`（正弦波）＋ゲイン包絡。WAV 出力は `OfflineAudioContext` でレンダリング
- **再現性**: seed 付き PRNG（mulberry32）で、同じ seed からは同じ問題・同じ音を再生成
- **成績**: `localStorage` に保存（サーバ不要）

## 🗺️ ロードマップ

- [x] **フェーズ1**: ブラウザ版の基盤（源氏香モード）＋答え合わせ音声の再生 ＋ GitHub Pages 配信
- [x] **フェーズ2**: `taketori` モード — reference と比較メロディの Same/Different を答え続け、間違えたら終了（連続正解記録）
- [ ] **フェーズ3**: 音声素材の拡張 — [klattsch](https://github.com/tgies/klattsch)（ブラウザで動く Klatt 音声合成、MIT）で話し方・スピード・抑揚を素材化

## 🖥️ デスクトップ版 (v0) について

初期版は PyQt6 の**デスクトップアプリ**（`script/kumikyo.py`、成績は SQLite）でした。
現在は上記の Web 版を主軸に開発しており、v0 は `v0_260724` ブランチに保存されています。

<details>
<summary>v0 (PyQt6 デスクトップ版) の起動方法</summary>

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python script/kumikyo.py
```

依存: PyQt6 / numpy / simpleaudio / pillow。成績は `~/Library/Application Support/Kumikyo/data.db`（macOS）等に保存されます。

</details>

## 📄 License / Contact

haruka.ij [at] gmail.com

## 🙏 Acknowledgments

- 源氏香（源氏物語 各帖の香パターン）
- [klattsch](https://github.com/tgies/klattsch) — Tony Gies (MIT)
- Web Audio API

---

<div align="center">

**🎵 耳を鍛えて、心を広げよう 🧠**

</div>
