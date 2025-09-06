# 🎵 Kumikyo - 組响 - Melodic Pattern Recognition Trainer

A sophisticated cognitive training application that helps users develop advanced auditory pattern recognition skills through classical Japanese literary symbols and melodic sequences. The overall game structure was inspired by Genjiko (源氏香) of Kumiko (組香)

## Demo
![Demo](data/demo.gif)

sorry for the gray color

## 🌟 Overview

Kumikyo trains users to recognize complex melodic grouping patterns by combining:
- **Auditory Learning**: Sequential 5-melody playback with varying complexity
- **Visual Recognition**: Classical Japanese Genji-ko symbols from *The Tale of Genji*
- **Progressive Difficulty**: From simple 3-note melodies to complex 5-note sequences
- **Cognitive Challenge**: Pattern matching between auditory and visual modalities

## 🎯 Features

### 🎵 Sequential Melody Training
- **5-Position Sequences**: Each session plays 5 melodies based on grouping patterns (e.g., Pattern A → B → C → B → A)
- **Dynamic Note Count**: Melody complexity increases with difficulty (3, 4, or 5 notes per melody)
- **Single-Note Substitutions**: Extremely challenging discrimination with only ±1 scale step differences
- **Precise Timing**: 0.5 seconds per note (1.5-2.5s melodies) with 1.5-second gaps

### 🖼️ Visual Pattern Recognition
- **52 Genji-ko Symbols**: Authentic patterns from classical Japanese literature
- **2×3 Selection Grid**: Clean, intuitive interface for pattern identification
- **High-Quality Images**: Scaled visual symbols from traditional Japanese aesthetics
- **Responsive Design**: Hover effects and selection highlighting for better UX

### 📊 Four Difficulty Levels

| Level | Notes/Melody | Timing | Visual Display | Challenge |
|-------|--------------|--------|----------------|----------|
| 🟢 **Easy** | 3 notes | 1.5s + 1.5s gap | Images | Single-note substitutions (±1 scale step) |
| 🟡 **Normal** | 4 notes | 2.0s + 1.5s gap | Images | Adjacent-note differences only |
| 🔴 **Hard** | 5 notes | 2.5s + 1.5s gap | Images | Extremely subtle single-step variations |
| 🟣 **Very Hard** | 5 notes | 2.5s + 1.5s gap | Text Only | Maximum discrimination with text labels |

### 💾 Performance Tracking
- **SQLite Database**: Comprehensive session and trial data storage
- **Reaction Time**: Millisecond-precision response tracking  
- **Success Rates**: 14-day performance statistics with detailed breakdowns
- **Progress Analytics**: Track improvement over time across difficulty levels

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Audio output device (speakers/headphones)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/kumikyo.git
cd kumikyo

# Create virtual environment
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- **PyQt6**: Modern GUI framework
- **numpy**: High-performance audio synthesis
- **simpleaudio**: Cross-platform audio playback
- **csv**: Genji pattern data processing

## 🎮 Usage

### Quick Start
```bash
# Activate environment
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Run the application
python script/kumikyo.py
```

### Training Workflow
1. **🎚️ Select Difficulty**: Choose from Easy, Normal, Hard, or Very Hard
2. **▶️ Start Session**: Click "Start New Training Session"
3. **🎵 Listen**: Click "Play Melodic Sequence" to hear 5 melodies
4. **👁️ Identify**: Click on the visual pattern that matches the sequence
5. **✅ Submit**: Get immediate feedback with correct answer highlighting
6. **📈 Track**: View progress with "Recent Stats" button

### Training Tips
- **🎧 Use headphones** for best audio clarity - essential for subtle differences
- **🟢 Start with Easy** but expect significant challenge even at basic level
- **👂 Focus on single-note differences** - melodies are extremely similar
- **🔄 Listen multiple times** - discrimination requires intense concentration
- **⚠️ Expect difficulty** - this is advanced auditory discrimination training
- **📊 Track progress** to see gradual improvement in subtle pattern recognition

## 🏗️ Technical Architecture

### Core Components
- **🎼 Stimulus Generation**: Dynamic melody creation with edit distance constraints
- **🎨 Audio Synthesis**: Pure sine wave generation using pentatonic scales
- **🖼️ Visual System**: PNG image loading and display with fallback text
- **💽 Data Persistence**: SQLite with WAL mode for reliable storage
- **🔄 Pattern Matching**: Algorithmic grouping based on 5-digit Genji sequences

### File Structure
```
kumikyo/
├── script/
│   └── kumikyo.py          # Main application
├── data/
│   └── genji_ko.csv        # 52 Genji pattern definitions
├── fig_genjiko/            # Visual pattern images (52 PNG files)
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── venv/                  # Virtual environment
```

### Audio Design Philosophy
- **🎵 Major Scale**: Constrained C-D-E-F-G scale for subtle discrimination training
- **⏱️ Precise Timing**: 0.5 seconds per note, 1.5-second gaps for optimal processing
- **🔊 Sine Waves**: Pure tones eliminate timbre distractions
- **📐 Single-Step Substitutions**: Adjacent-note-only changes for maximum challenge

## 💾 Data Storage

Training data is automatically saved to:
- **macOS**: `~/Library/Application Support/Kumikyo/data.db`
- **Windows**: `%APPDATA%\\Kumikyo\\data.db`
- **Linux**: `~/.local/share/Kumikyo/data.db`

### Database Schema
- **Sessions**: Metadata, difficulty, seed, stimulus JSON
- **Trials**: Individual responses, reaction times, correctness
- **Statistics**: Aggregated performance over time

## 🧪 Testing

Run the test suite to verify functionality:
```bash
# Activate environment
source venv/bin/activate

# Run tests
python script/dev/test_audio.py      # Audio synthesis tests
python script/dev/test_melody.py     # Melody generation tests  
python script/dev/test_difficulty.py # Difficulty scaling tests
python script/dev/test_complete.py   # End-to-end system tests
```

## 📚 Research Applications

Kumikyo is designed for:
- **🧠 Cognitive Research**: Fine-grained auditory discrimination studies
- **🎓 Music Education**: Advanced interval recognition and pitch discrimination
- **🧬 Neuroplasticity**: Challenging sequential memory and subtle pattern learning
- **📊 Behavioral Analysis**: Precision reaction time and accuracy in difficult tasks
- **🔬 Cross-modal Learning**: Audio-visual association under challenging conditions
- **🎯 Perceptual Training**: Development of expert-level auditory discrimination skills

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **🍴 Fork** the repository
2. **🌿 Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **💾 Commit changes** (`git commit -m 'Add amazing feature'`)
4. **📤 Push to branch** (`git push origin feature/amazing-feature`)
5. **🔄 Open Pull Request**

## 📄 License

Email me, haruka.ij [at] gmail.com

## 🙏 Acknowledgments

- **Classical Japanese Literature**: Genji-ko patterns from *The Tale of Genji*
- **Cognitive Science Research**: Pattern recognition and auditory learning principles
- **Open Source Libraries**: PyQt6, NumPy, and the Python audio ecosystem
- **Educational Philosophy**: Progressive difficulty and multimodal learning approaches

---

<div align="center">

**🎵 Train Your Ear, Expand Your Mind 🧠**

*Kumikyo combines the beauty of Japanese classical literature with cutting-edge cognitive training*

</div>