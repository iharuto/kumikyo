#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kumikyo Sound Cognition Trainer
-------------------------------
• Tech stack: Python 3.10+, PyQt6, numpy, simpleaudio, sqlite3
• Focus: sound cognition training with pattern recognition
• Training mode: 5 sounds with weak randomness, user identifies matching sounds

Install dependencies:
    pip install -r requirements.txt

Run:
    python kumikyo.py

DB path (mac):
    ~/Library/Application Support/Kumikyo/data.db
"""

from __future__ import annotations
import os
import sys
import math
import time
import json
import sqlite3
import random
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import simpleaudio as sa
from PyQt6 import QtCore, QtGui, QtWidgets

APP_NAME = "Kumikyo"
APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / APP_NAME
DB_PATH = APP_SUPPORT_DIR / "data.db"

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "genji_ko.csv"
FIG_PATH = PROJECT_ROOT / "fig_genjiko"

# ------------------------------
# Domain & stimuli
# ------------------------------
NOTE_FREQS = {
    "C4": 261.626, "C#4": 277.183,
    "D4": 293.665, "D#4": 311.127,
    "E4": 329.628,
    "F4": 349.228, "F#4": 369.994,
    "G4": 391.995, "G#4": 415.305,
    "A4": 440.000, "A#4": 466.164,
    "B4": 493.883,
}
NOTE_LIST = list(NOTE_FREQS.keys())

# Genji-ko patterns loaded from CSV
GENJI_PATTERNS: Dict[str, str] = {}  # rgs -> slug mapping
PATTERN_NAMES: List[str] = []  # list of rgs patterns

def load_genji_patterns():
    """Load Genji-ko patterns from CSV file"""
    global GENJI_PATTERNS, PATTERN_NAMES
    
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rgs = row['rgs']
                slug = row['slug']
                GENJI_PATTERNS[rgs] = slug
                PATTERN_NAMES.append(rgs)
        print(f"Loaded {len(GENJI_PATTERNS)} Genji-ko patterns")
    except Exception as e:
        print(f"Error loading Genji-ko patterns: {e}")
        # Fallback patterns if CSV fails to load
        GENJI_PATTERNS = {
            "12345": "hahakigi",
            "12344": "utsusemi",
            "12334": "yugao",
            "12233": "wakamurasaki",
            "11112": "suetsumuhana"
        }
        PATTERN_NAMES = list(GENJI_PATTERNS.keys())

# Difficulty levels
DIFFICULTY_LEVELS = {
    "easy": {"n_positions": 3, "max_edit_distance": 3},
    "normal": {"n_positions": 4, "max_edit_distance": 2},
    "hard": {"n_positions": 5, "max_edit_distance": 1},
    "very_hard": {"n_positions": 5, "max_edit_distance": 1},
}

def generate_difficulty_melody(group_id: int, difficulty: str, n_notes: int, seed: int = 42) -> List[int]:
    """Generate a melody with single-step substitutions for challenging discrimination"""
    rng = random.Random(seed + group_id)  # Deterministic but different per group
    
    # Constrained major scale: C, D, E, F, G (adjacent semitones for subtle differences)
    major_scale = [0, 2, 4, 5, 7]  # More constrained than pentatonic
    
    # Generate random base melody for group 0 (using seed for reproducibility)
    if group_id == 0:
        melody = []
        for i in range(n_notes):
            melody.append(rng.choice(major_scale))  # Random note from major scale
        return melody
    
    # For other groups, create subtle variations with single-step substitutions
    base_melody = generate_difficulty_melody(0, difficulty, n_notes, seed)
    max_distance = DIFFICULTY_LEVELS[difficulty]["max_edit_distance"]
    
    # Create variation by changing exactly 1 note (for maximum challenge)
    melody = base_melody.copy()
    
    # Only change 1 note, regardless of group_id (makes it very challenging)
    if n_notes > 0:
        # Pick one position to change
        pos_to_change = rng.randint(0, n_notes - 1)
        original_note = melody[pos_to_change]
        
        # Find the index of this note in our scale
        try:
            scale_index = major_scale.index(original_note)
            
            # Get adjacent notes (±1 step in scale)
            adjacent_options = []
            if scale_index > 0:  # Can go down one step
                adjacent_options.append(major_scale[scale_index - 1])
            if scale_index < len(major_scale) - 1:  # Can go up one step
                adjacent_options.append(major_scale[scale_index + 1])
            
            # If no adjacent options (shouldn't happen), use any other note
            if not adjacent_options:
                adjacent_options = [note for note in major_scale if note != original_note]
            
            # Choose a random adjacent note
            if adjacent_options:
                melody[pos_to_change] = rng.choice(adjacent_options)
                
        except ValueError:
            # Fallback: if original note not in scale, pick any different note
            available_notes = [note for note in major_scale if note != original_note]
            if available_notes:
                melody[pos_to_change] = rng.choice(available_notes)
    
    return melody

def calculate_melody_edit_distance(melody1: List[int], melody2: List[int]) -> int:
    """Calculate substitution distance between two melodies"""
    if len(melody1) != len(melody2):
        return max(len(melody1), len(melody2))
    
    # Count the number of note differences
    distance = sum(1 for a, b in zip(melody1, melody2) if a != b)
    return distance

def calculate_pattern_edit_distance(pattern1: str, pattern2: str) -> int:
    """Calculate substitution distance between two 5-digit patterns (legacy)"""
    if len(pattern1) != len(pattern2):
        return max(len(pattern1), len(pattern2))
    
    # Count the number of position differences
    distance = sum(1 for a, b in zip(pattern1, pattern2) if a != b)
    return distance

def get_compatible_patterns(reference_pattern: str, max_distance: int) -> List[str]:
    """Get all patterns within max_distance of the reference pattern"""
    compatible = []
    
    for pattern_rgs in PATTERN_NAMES:
        distance = calculate_pattern_edit_distance(reference_pattern, pattern_rgs)
        if distance <= max_distance:
            compatible.append(pattern_rgs)
    
    return compatible

@dataclass
class TrialResult:
    index: int                 # 0..N-1
    target_group: int          # 1..N (which group this position belongs to)
    user_choice: int           # 1..N
    correct: bool
    rt_ms: int                 # response time in ms

@dataclass
class SessionMeta:
    created_ts: int            # epoch seconds
    difficulty: str            # "easy", "normal", or "hard"
    n_positions: int
    instrument: str            # "sine" for MVP
    seed: int
    stimulus_json: str         # describes the stimuli pattern for reproducibility

# ------------------------------
# Persistence (sqlite)
# ------------------------------
class DB:
    def __init__(self, path: Path):
        self.path = path
        APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init()

    def _init(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                created_ts INTEGER NOT NULL,
                difficulty TEXT NOT NULL,
                n_positions INTEGER NOT NULL,
                instrument TEXT NOT NULL,
                seed INTEGER NOT NULL,
                stimulus_json TEXT NOT NULL
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL,
                idx INTEGER NOT NULL,
                target_group INTEGER NOT NULL,
                user_choice INTEGER NOT NULL,
                correct INTEGER NOT NULL,
                rt_ms INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            """
        )
        self.conn.commit()

    def create_session(self, meta: SessionMeta) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO sessions (created_ts, difficulty, n_positions, instrument, seed, stimulus_json) VALUES (?, ?, ?, ?, ?, ?)",
            (meta.created_ts, meta.difficulty, meta.n_positions, meta.instrument, meta.seed, meta.stimulus_json),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_trial(self, session_id: int, tr: TrialResult):
        self.conn.execute(
            "INSERT INTO trials (session_id, idx, target_group, user_choice, correct, rt_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, tr.index, tr.target_group, tr.user_choice, int(tr.correct), tr.rt_ms),
        )
        self.conn.commit()

    def recent_stats(self, limit_days: int = 14) -> List[Tuple[str, int, int]]:
        # returns list of (YYYY-MM-DD, correct_count, total_count)
        q = """
        SELECT date(datetime(created_ts, 'unixepoch')), 
               SUM(CASE WHEN t.correct=1 THEN 1 ELSE 0 END) as correct,
               COUNT(*) as total
        FROM sessions s
        JOIN trials t ON s.id = t.session_id
        WHERE s.created_ts >= strftime('%s','now','-? day')
        GROUP BY 1
        ORDER BY 1 DESC
        """
        cur = self.conn.cursor()
        cur.execute(q.replace("?", str(limit_days)))
        return cur.fetchall()

# ------------------------------
# Audio synthesis (simple sine)
# ------------------------------
class Synth:
    @staticmethod
    def sine_tone(freq: float, sec: float = 1.0, sr: int = 44100, fade_ms: int = 50):
        t = np.linspace(0, sec, int(sr * sec), False)
        wave = np.sin(2 * np.pi * freq * t)
        # simple envelope with longer fade for smoother sound
        fade = int(sr * fade_ms / 1000)
        if fade > 0:
            env = np.ones_like(wave)
            env[:fade] = np.linspace(0, 1, fade)
            env[-fade:] = np.linspace(1, 0, fade)
            wave *= env
        audio = (wave * 32767).astype(np.int16)
        return audio

    @staticmethod
    def create_melody(root_note: str, pattern: List[int], note_duration: float = 0.3, sr: int = 44100):
        """Create a melodic sequence from a root note and semitone pattern"""
        if root_note not in NOTE_FREQS:
            root_note = "C4"  # fallback
        
        root_freq = NOTE_FREQS[root_note]
        melody_buffers = []
        
        for semitones in pattern:
            # Calculate frequency using semitone formula: f = f0 * 2^(n/12)
            freq = root_freq * (2 ** (semitones / 12.0))
            note_buf = Synth.sine_tone(freq, sec=note_duration, fade_ms=20)
            melody_buffers.append(note_buf)
        
        # Concatenate all notes with tiny gaps
        gap_samples = int(sr * 0.05)  # 50ms gap between notes
        gap = np.zeros(gap_samples, dtype=np.int16)
        
        full_melody = melody_buffers[0]
        for buf in melody_buffers[1:]:
            full_melody = np.concatenate([full_melody, gap, buf])
        
        return full_melody
    
    @staticmethod
    def create_difficulty_melody(group_id: int, difficulty: str, root_note: str = "C4", note_duration: float = 0.3, seed: int = 42):
        """Create a melodic sequence based on group ID and difficulty constraints"""
        n_notes = DIFFICULTY_LEVELS[difficulty]["n_positions"]
        intervals = generate_difficulty_melody(group_id, difficulty, n_notes, seed)
        return Synth.create_melody(root_note, intervals, note_duration)
    
    @staticmethod
    def create_genji_pattern_melody(rgs: str, root_note: str = "C4", note_duration: float = 0.3):
        """Legacy function - kept for compatibility but not used in new system"""
        # This is kept for backwards compatibility but should not be used
        # Use create_difficulty_melody instead
        intervals = [0, 2, 4]  # Simple fallback
        return Synth.create_melody(root_note, intervals, note_duration)

    @staticmethod
    def play_buffer(buf: np.ndarray, sr: int = 44100):
        play_obj = sa.play_buffer(buf.tobytes(), 1, 2, sr)
        play_obj.wait_done()

# ------------------------------
# Stimulus generator with weak randomness for training
# ------------------------------
class Stimulus:
    def __init__(self, difficulty: str = "normal", seed: int | None = None):
        if difficulty not in DIFFICULTY_LEVELS:
            difficulty = "normal"
        
        self.difficulty = difficulty
        self.n = 5  # Always 5 positions for Genji-ko patterns
        self.max_edit_distance = DIFFICULTY_LEVELS[difficulty]["max_edit_distance"]
        self.seed = random.randint(0, 2**31-1) if seed is None else seed
        rng = random.Random(self.seed)

        # Select target pattern from available Genji-ko patterns
        if not PATTERN_NAMES:
            load_genji_patterns()
        
        self.target_pattern = rng.choice(PATTERN_NAMES)
        self.target_slug = GENJI_PATTERNS[self.target_pattern]
        
        # Select 5 distractor patterns based on difficulty constraints
        compatible_patterns = get_compatible_patterns(self.target_pattern, self.max_edit_distance)
        
        # Remove target from compatible patterns to avoid duplicates
        available_distractors = [p for p in compatible_patterns if p != self.target_pattern]
        
        # If we don't have enough compatible patterns, add more from all patterns
        if len(available_distractors) < 5:
            remaining_patterns = [p for p in PATTERN_NAMES if p != self.target_pattern and p not in available_distractors]
            available_distractors.extend(remaining_patterns[:5 - len(available_distractors)])
        
        # Select 5 distractors
        self.distractor_patterns = rng.sample(available_distractors, min(5, len(available_distractors)))
        
        # If still not enough, pad with random patterns
        while len(self.distractor_patterns) < 5:
            random_pattern = rng.choice([p for p in PATTERN_NAMES if p != self.target_pattern])
            if random_pattern not in self.distractor_patterns:
                self.distractor_patterns.append(random_pattern)
        
        # Create all 6 patterns (1 target + 5 distractors) for the grid
        self.all_patterns = [self.target_pattern] + self.distractor_patterns
        rng.shuffle(self.all_patterns)  # Randomize positions in 2x3 grid
        
        # Find where the target ended up after shuffling
        self.correct_position = self.all_patterns.index(self.target_pattern)
        
        # Create 5 position melody groups based on the target pattern's grouping structure
        self.root_note = rng.choice(NOTE_LIST)
        self.position_melody_groups = self._generate_position_melody_groups()
        self.position_patterns = self._generate_position_patterns(rng)  # Keep for visual display
        
        # Legacy compatibility - now represents the 5 position melodies
        self.melodies = [(self.root_note, f"group_{group}") for group in self.position_melody_groups]
        self.groups = [int(digit) for digit in self.target_pattern]  # Group assignments for each position
        self.sequence = [f"{self.root_note}-group_{group}" for group in self.position_melody_groups]
    
    def get_target_image_path(self) -> Path:
        """Get the image path for the target pattern"""
        return FIG_PATH / f"{self.target_pattern}_{self.target_slug}.png"
    
    def get_pattern_image_path(self, rgs: str) -> Path:
        """Get the image path for a given pattern"""
        slug = GENJI_PATTERNS.get(rgs, "unknown")
        return FIG_PATH / f"{rgs}_{slug}.png"
    
    def get_pattern_slug(self, rgs: str) -> str:
        """Get the slug name for a given pattern"""
        return GENJI_PATTERNS.get(rgs, "unknown")
    
    def _generate_position_melody_groups(self) -> List[int]:
        """Generate 5 melody group IDs based on the target pattern's digit grouping"""
        # Parse the target pattern to understand grouping (e.g., "12321" means pos 1&5 same, pos 2&4 same, pos 3 unique)
        target_digits = list(self.target_pattern)
        unique_digits = list(set(target_digits))
        
        # Map each unique digit to a melody group ID (0, 1, 2, etc.)
        digit_to_group = {}
        for i, digit in enumerate(unique_digits):
            digit_to_group[digit] = i
        
        # Generate 5 position melody group IDs based on the grouping
        position_groups = []
        for digit in target_digits:
            position_groups.append(digit_to_group[digit])
        
        return position_groups
    
    def _generate_position_patterns(self, rng: random.Random) -> List[str]:
        """Generate 5 patterns based on the target pattern's digit grouping (for visual display)"""
        # Parse the target pattern to understand grouping (e.g., "12321" means pos 1&5 same, pos 2&4 same, pos 3 unique)
        target_digits = list(self.target_pattern)
        unique_groups = list(set(target_digits))
        
        # Assign a random pattern from available patterns to each unique group
        group_to_pattern = {}
        available_patterns = [p for p in PATTERN_NAMES if p != self.target_pattern]
        
        for group_digit in unique_groups:
            # Select a random pattern for this group
            if available_patterns:
                group_pattern = rng.choice(available_patterns)
                available_patterns.remove(group_pattern)  # Don't reuse patterns
            else:
                # If we run out, use a random pattern (less ideal but functional)
                group_pattern = rng.choice(PATTERN_NAMES)
            group_to_pattern[group_digit] = group_pattern
        
        # Generate 5 position patterns based on the grouping
        position_patterns = []
        for digit in target_digits:
            position_patterns.append(group_to_pattern[digit])
        
        return position_patterns

    def to_json(self) -> str:
        return json.dumps({
            "n": self.n,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "target_pattern": self.target_pattern,
            "distractor_patterns": self.distractor_patterns,
            "all_patterns": self.all_patterns,
            "correct_position": self.correct_position,
            "root_note": self.root_note,
            "position_melody_groups": self.position_melody_groups,
            "melodies": self.melodies,
            "sequence": self.sequence,
            "groups": self.groups,
        })

# ------------------------------
# Qt UI
# ------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kumikyo - Melodic Pattern Recognition Trainer")
        self.resize(700, 600)
        self.db = DB(DB_PATH)
        self.current_session_id: int | None = None
        self.current_stim: Stimulus | None = None
        self.trial_started_ts: float | None = None
        self.current_difficulty: str = "normal"  # default difficulty
        self.choice_groups: List[QtWidgets.QButtonGroup] = []
        self.grid_widget: QtWidgets.QWidget | None = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.lbl_header = QtWidgets.QLabel("Melodic Pattern Recognition Training")
        self.lbl_header.setStyleSheet("font-size:24px; font-weight:bold; color:#2c3e50;")
        self.lbl_header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.lbl_instructions = QtWidgets.QLabel(
            "Choose difficulty, then listen to a melodic sequence (5 melodies) and click on the matching image below.\n"
            "Each melody has 3, 4, or 5 notes depending on difficulty. Visual patterns from classical Japanese literature."
        )
        self.lbl_instructions.setStyleSheet("font-size:14px; color:#34495e; margin:10px;")
        self.lbl_instructions.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Difficulty selection
        self.difficulty_widget = QtWidgets.QWidget()
        difficulty_layout = QtWidgets.QHBoxLayout(self.difficulty_widget)
        
        self.lbl_difficulty = QtWidgets.QLabel("Difficulty:")
        self.lbl_difficulty.setStyleSheet("font-size:16px; font-weight:bold;")
        
        self.btn_easy = QtWidgets.QPushButton("Easy (3 sounds)")
        self.btn_easy.setStyleSheet("font-size:14px; padding:8px; background-color:#2ecc71; color:white; border:none; border-radius:5px;")
        self.btn_easy.clicked.connect(lambda: self.set_difficulty("easy"))
        
        self.btn_normal = QtWidgets.QPushButton("Normal (4 sounds)")
        self.btn_normal.setStyleSheet("font-size:14px; padding:8px; background-color:#f39c12; color:white; border:none; border-radius:5px; font-weight:bold;")
        self.btn_normal.clicked.connect(lambda: self.set_difficulty("normal"))
        
        self.btn_hard = QtWidgets.QPushButton("Hard (5 sounds)")
        self.btn_hard.setStyleSheet("font-size:14px; padding:8px; background-color:#e74c3c; color:white; border:none; border-radius:5px;")
        self.btn_hard.clicked.connect(lambda: self.set_difficulty("hard"))
        
        self.btn_very_hard = QtWidgets.QPushButton("Very Hard (text)")
        self.btn_very_hard.setStyleSheet("font-size:14px; padding:8px; background-color:#8e44ad; color:white; border:none; border-radius:5px;")
        self.btn_very_hard.clicked.connect(lambda: self.set_difficulty("very_hard"))
        
        difficulty_layout.addWidget(self.lbl_difficulty)
        difficulty_layout.addWidget(self.btn_easy)
        difficulty_layout.addWidget(self.btn_normal)
        difficulty_layout.addWidget(self.btn_hard)
        difficulty_layout.addWidget(self.btn_very_hard)
        difficulty_layout.addStretch()

        self.btn_new = QtWidgets.QPushButton("Start New Training Session")
        self.btn_new.setStyleSheet("font-size:16px; padding:10px; background-color:#3498db; color:white; border:none; border-radius:5px;")
        self.btn_new.clicked.connect(self.start_new_session)

        self.play_btn = QtWidgets.QPushButton("Play Melodic Sequence")
        self.play_btn.setStyleSheet("font-size:16px; padding:10px; background-color:#2ecc71; color:white; border:none; border-radius:5px;")
        self.play_btn.clicked.connect(self.play_sequence)
        self.play_btn.setEnabled(False)

        # Create placeholder for pattern grid - will be created after layout
        self.grid_widget = None
        self.image_buttons = []
        self.selected_button = None

        self.btn_submit = QtWidgets.QPushButton("Submit Answer")
        self.btn_submit.setStyleSheet("font-size:16px; padding:10px; background-color:#e74c3c; color:white; border:none; border-radius:5px;")
        self.btn_submit.setEnabled(False)
        self.btn_submit.clicked.connect(self.submit_answers)

        self.lbl_feedback = QtWidgets.QLabel("")
        self.lbl_feedback.setStyleSheet("font-size:16px; padding:10px; border-radius:5px;")
        self.lbl_feedback.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.btn_stats = QtWidgets.QPushButton("View Recent Stats (14 days)")
        self.btn_stats.setStyleSheet("font-size:14px; padding:8px; background-color:#9b59b6; color:white; border:none; border-radius:5px;")
        self.btn_stats.clicked.connect(self.show_stats_dialog)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(15)
        layout.addWidget(self.lbl_header)
        layout.addWidget(self.lbl_instructions)
        layout.addWidget(self.difficulty_widget)
        layout.addWidget(self.btn_new)
        layout.addWidget(self.play_btn)
        if self.grid_widget:
            layout.addWidget(self.grid_widget)
        layout.addWidget(self.btn_submit)
        layout.addWidget(self.lbl_feedback)
        layout.addStretch()
        layout.addWidget(self.btn_stats)
        central.setLayout(layout)
        
        # Create pattern grid after layout is set
        self.create_pattern_grid()

    def set_difficulty(self, difficulty: str):
        """Set the current difficulty level and update UI"""
        self.current_difficulty = difficulty
        
        # Update button styles to show selection
        self.btn_easy.setStyleSheet("font-size:14px; padding:8px; background-color:#2ecc71; color:white; border:none; border-radius:5px;" + 
                                   (" font-weight:bold;" if difficulty == "easy" else ""))
        self.btn_normal.setStyleSheet("font-size:14px; padding:8px; background-color:#f39c12; color:white; border:none; border-radius:5px;" + 
                                     (" font-weight:bold;" if difficulty == "normal" else ""))
        self.btn_hard.setStyleSheet("font-size:14px; padding:8px; background-color:#e74c3c; color:white; border:none; border-radius:5px;" + 
                                   (" font-weight:bold;" if difficulty == "hard" else ""))
        self.btn_very_hard.setStyleSheet("font-size:14px; padding:8px; background-color:#8e44ad; color:white; border:none; border-radius:5px;" + 
                                        (" font-weight:bold;" if difficulty == "very_hard" else ""))
        
        # Update instructions based on difficulty
        max_distance = DIFFICULTY_LEVELS[difficulty]["max_edit_distance"]
        difficulty_desc = {
            "easy": "patterns are very different",
            "normal": "patterns are moderately similar", 
            "hard": "patterns are very similar",
            "very_hard": "patterns are extremely similar, shown as text"
        }
        n_notes = DIFFICULTY_LEVELS[difficulty]["n_positions"]
        if difficulty == "very_hard":
            self.lbl_instructions.setText(
                f"Listen to a melodic sequence (5 melodies, {n_notes} notes each) and click on the matching text below.\n"
                f"{difficulty.replace('_', ' ').title()} mode: {difficulty_desc[difficulty]}."
            )
        else:
            self.lbl_instructions.setText(
                f"Listen to a melodic sequence (5 melodies, {n_notes} notes each) and click on the matching image below.\n"
                f"{difficulty.title()} mode: {difficulty_desc[difficulty]}."
            )
    
    def create_pattern_grid(self):
        """Create 2x3 grid for pattern selection"""
        # Remove old grid if it exists
        if self.grid_widget:
            self.grid_widget.setParent(None)
        
        # Create new grid widget
        self.grid_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(self.grid_widget)
        grid.setSpacing(10)
        
        # Initially empty - will be populated when session starts
        self.image_buttons = []
        self.selected_button = None
        
        # Create 6 placeholder buttons in 2x3 grid
        for i in range(2):  # rows
            for j in range(3):  # columns
                btn = QtWidgets.QPushButton("")
                btn.setFixedSize(150, 150)
                btn.setStyleSheet(
                    "QPushButton { border: 2px solid #bdc3c7; background-color: #ecf0f1; border-radius: 10px; }"
                    "QPushButton:hover { border-color: #3498db; }"
                    "QPushButton:pressed { background-color: #d5dbdb; }"
                )
                btn.clicked.connect(lambda checked, button=btn: self.on_pattern_click(button))
                btn.setVisible(False)  # Hidden until session starts
                grid.addWidget(btn, i, j)
                self.image_buttons.append(btn)
        
        # Add the grid to the main layout
        layout = self.centralWidget().layout()
        # Insert the grid widget before the submit button
        for i in range(layout.count()):
            if layout.itemAt(i).widget() == self.btn_submit:
                layout.insertWidget(i, self.grid_widget)
                break
    
    def on_pattern_click(self, button: QtWidgets.QPushButton):
        """Handle pattern button click"""
        # Reset previous selection with mode-appropriate styling
        if self.selected_button:
            if self.current_difficulty == "very_hard":
                # Very Hard mode: text-only styling
                self.selected_button.setStyleSheet(
                    "QPushButton { border: 2px solid #bdc3c7; background-color: #ecf0f1; border-radius: 10px; "
                    "font-size: 12px; font-weight: bold; color: #2c3e50; }"
                    "QPushButton:hover { border-color: #3498db; }"
                    "QPushButton:pressed { background-color: #d5dbdb; }"
                )
            else:
                # Other modes: image-friendly styling
                self.selected_button.setStyleSheet(
                    "QPushButton { border: 2px solid #bdc3c7; background-color: #ecf0f1; border-radius: 10px; }"
                    "QPushButton:hover { border-color: #3498db; }"
                    "QPushButton:pressed { background-color: #d5dbdb; }"
                )
        
        # Highlight new selection with mode-appropriate styling
        if self.current_difficulty == "very_hard":
            # Very Hard mode: text-only selection styling
            button.setStyleSheet(
                "QPushButton { border: 3px solid #e74c3c; background-color: #fadbd8; border-radius: 10px; "
                "font-size: 12px; font-weight: bold; color: #2c3e50; }"
                "QPushButton:hover { border-color: #c0392b; }"
                "QPushButton:pressed { background-color: #f1c2c0; }"
            )
        else:
            # Other modes: standard selection styling
            button.setStyleSheet(
                "QPushButton { border: 3px solid #e74c3c; background-color: #fadbd8; border-radius: 10px; }"
                "QPushButton:hover { border-color: #c0392b; }"
                "QPushButton:pressed { background-color: #f1c2c0; }"
            )
        self.selected_button = button
    
    def populate_pattern_grid(self):
        """Populate the 2x3 grid with current stimulus patterns"""
        if not self.current_stim:
            return
        
        # Show all buttons
        for btn in self.image_buttons:
            btn.setVisible(True)
        
        # Populate buttons with patterns
        for i, (btn, pattern_rgs) in enumerate(zip(self.image_buttons, self.current_stim.all_patterns)):
            if self.current_difficulty == "very_hard":
                # Use slug text for very hard mode
                slug = self.current_stim.get_pattern_slug(pattern_rgs)
                btn.setText(slug.replace('_', ' ').title())
                # Clear any existing icon from previous sessions
                btn.setIcon(QtGui.QIcon())
                btn.setIconSize(QtCore.QSize(0, 0))
                btn.setStyleSheet(
                    "QPushButton { border: 2px solid #bdc3c7; background-color: #ecf0f1; border-radius: 10px; "
                    "font-size: 12px; font-weight: bold; color: #2c3e50; }"
                    "QPushButton:hover { border-color: #3498db; }"
                    "QPushButton:pressed { background-color: #d5dbdb; }"
                )
            else:
                # Use images for other modes
                image_path = self.current_stim.get_pattern_image_path(pattern_rgs)
                if image_path.exists():
                    pixmap = QtGui.QPixmap(str(image_path))
                    # Scale image to fit button while maintaining aspect ratio
                    scaled_pixmap = pixmap.scaled(140, 140, QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
                                                 QtCore.Qt.TransformationMode.SmoothTransformation)
                    icon = QtGui.QIcon(scaled_pixmap)
                    btn.setIcon(icon)
                    btn.setIconSize(QtCore.QSize(140, 140))
                    btn.setText("")
                else:
                    # Fallback to text if image not found
                    slug = self.current_stim.get_pattern_slug(pattern_rgs)
                    btn.setText(f"{pattern_rgs}\n{slug}")
        
        # Reset selection
        self.selected_button = None

    # -------- session lifecycle --------
    def start_new_session(self):
        stim = Stimulus(difficulty=self.current_difficulty)
        meta = SessionMeta(
            created_ts=int(time.time()),
            difficulty=stim.difficulty,
            n_positions=stim.n,
            instrument="sine",
            seed=stim.seed,
            stimulus_json=stim.to_json(),
        )
        sid = self.db.create_session(meta)
        self.current_session_id = sid
        self.current_stim = stim
        
        # Populate the pattern grid with new stimulus
        self.populate_pattern_grid()
        
        difficulty_name = self.current_difficulty.replace('_', ' ').title()
        mode_type = "text" if self.current_difficulty == "very_hard" else "images"
        self.lbl_feedback.setText(f"New {difficulty_name} session started with {mode_type}. Click 'Play Melodic Sequence' to begin.")
        self.lbl_feedback.setStyleSheet("font-size:16px; padding:10px; background-color:#d5f4e6; color:#27ae60; border-radius:5px;")
        self.play_btn.setEnabled(True)
        self.btn_submit.setEnabled(True)
        self.trial_started_ts = None

    def play_sequence(self):
        if not self.current_stim:
            return
        n_notes = DIFFICULTY_LEVELS[self.current_difficulty]["n_positions"]
        self.lbl_feedback.setText(f"Playing melodic sequence (5 melodies, {n_notes} notes each)... Please listen carefully.")
        self.lbl_feedback.setStyleSheet("font-size:16px; padding:10px; background-color:#fef9e7; color:#f39c12; border-radius:5px;")
        QtWidgets.QApplication.processEvents()
        
        # Play all 5 position melodies in sequence
        for i, group_id in enumerate(self.current_stim.position_melody_groups):
            # Create and play melody for this position with difficulty-appropriate note count
            # Each note is 0.5 seconds, so total melody duration = n_notes * 0.5 seconds
            n_notes = DIFFICULTY_LEVELS[self.current_difficulty]["n_positions"]
            note_duration = 0.5  # Fixed 0.5 seconds per note
            
            buf = Synth.create_difficulty_melody(
                group_id, 
                self.current_stim.difficulty, 
                self.current_stim.root_note, 
                note_duration=note_duration,
                seed=self.current_stim.seed
            )
            Synth.play_buffer(buf)
            
            # Add 1.5 second gap between patterns (but not after the last one)
            if i < len(self.current_stim.position_melody_groups) - 1:
                time.sleep(1.5)
        
        self.trial_started_ts = time.time()
        mode_instruction = "Click on the matching visual pattern below" if self.current_difficulty != "very_hard" else "Click on the matching text below"
        self.lbl_feedback.setText(f"Sequence played! {mode_instruction} and click 'Submit Answer'.")
        self.lbl_feedback.setStyleSheet("font-size:16px; padding:10px; background-color:#e8f5e8; color:#2c3e50; border-radius:5px;")

    def submit_answers(self):
        if not (self.current_stim and self.current_session_id is not None):
            return
        
        # Check if user made a selection
        if not self.selected_button:
            self.lbl_feedback.setText("Please select a pattern before submitting.")
            self.lbl_feedback.setStyleSheet("font-size:16px; padding:10px; background-color:#fff3cd; color:#856404; border-radius:5px;")
            return
        
        rt_ms = 0
        if self.trial_started_ts:
            rt_ms = int((time.time() - self.trial_started_ts) * 1000)
        
        # Find which button was selected (0-based index)
        selected_index = self.image_buttons.index(self.selected_button)
        
        # Check if correct (compare with target position after shuffling)
        correct = (selected_index == self.current_stim.correct_position)
        
        # Store as single trial result
        tr = TrialResult(
            index=0,  # Single trial per session now
            target_group=self.current_stim.correct_position + 1,  # 1-indexed for DB
            user_choice=selected_index + 1,  # 1-indexed for DB
            correct=correct,
            rt_ms=rt_ms,
        )
        self.db.add_trial(self.current_session_id, tr)
        
        if correct:
            target_slug = self.current_stim.target_slug.replace('_', ' ').title()
            self.lbl_feedback.setText(f"✅ Correct! You identified '{target_slug}' successfully! Start a new session to continue training.")
            self.lbl_feedback.setStyleSheet("font-size:16px; padding:10px; background-color:#d5f4e6; color:#27ae60; border-radius:5px;")
        else:
            target_slug = self.current_stim.target_slug.replace('_', ' ').title()
            selected_pattern = self.current_stim.all_patterns[selected_index]
            selected_slug = self.current_stim.get_pattern_slug(selected_pattern).replace('_', ' ').title()
            self.lbl_feedback.setText(f"❌ Incorrect. You selected '{selected_slug}' but the correct answer was '{target_slug}'. Try a new session!")
            self.lbl_feedback.setStyleSheet("font-size:16px; padding:10px; background-color:#fadbd8; color:#e74c3c; border-radius:5px;")
        
        # Highlight the correct answer
        correct_button = self.image_buttons[self.current_stim.correct_position]
        correct_button.setStyleSheet(
            "QPushButton { border: 3px solid #27ae60; background-color: #d5f4e6; border-radius: 10px; }"
        )

    # -------- simple stats --------
    def show_stats_dialog(self):
        rows = self.db.recent_stats(limit_days=14)
        msg = "Recent Training Performance (Last 14 Days)\n\n"
        if not rows:
            msg += "No training records yet. Start your first session!"
        else:
            for d, ok, tot in rows:
                pct = (ok / tot * 100.0) if tot else 0.0
                msg += f"{d}: {ok}/{tot} correct ({pct:.1f}%)\n"
        QtWidgets.QMessageBox.information(self, "Training Stats", msg)

# ------------------------------
# main
# ------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Load Genji patterns at startup
    load_genji_patterns()
    
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()