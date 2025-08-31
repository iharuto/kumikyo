# genji_kou_links_levels.py
# pip install pillow

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import csv

# ----------------------------
# スタイル
# ----------------------------
# Simple configuration - only what you need to change
LINE_WIDTH = 32  # Easy to modify - controls both bars and connections
IMAGE_SIZE = (480, 380)
COLORS = {
    'bg': (255, 255, 255),
    'bar': (92, 45, 145),  # Purple
    'text': (20, 20, 20)
}

# ----------------------------
# ユーティリティ
# ----------------------------
def parse_rgs(rgs: str) -> List[int]:
    assert len(rgs) == 5 and all(c in "12345" for c in rgs), f"invalid rgs: {rgs}"
    return [int(c) for c in rgs]

def parse_heights(code: str) -> List[int]:
    """各桁は 0..4（4が最長）。"""
    assert len(code) == 5 and all(c in "01234" for c in code), f"invalid heights: {code}"
    return [int(c) for c in code]

def groups_from_rgs(rgs: List[int]) -> Dict[int, List[int]]:
    """グループ番号 -> そのグループに属する x-index（0..4）のリスト"""
    g: Dict[int, List[int]] = {}
    for i, label in enumerate(rgs):
        g.setdefault(label, []).append(i)
    return g

def group_level(label: int) -> int:
    """
    グループ番号から接続線レベルを決定。
    例: 1->4, 2->3, 3->2, 4->1, 5->1（5は実質単独なので描かれない想定）
    """
    mapping = {1: 4, 2: 3, 3: 2, 4: 1, 5: 1}
    return mapping.get(label, 1)

# ----------------------------
# 描画本体
# ----------------------------
def draw_genji_panel(rgs_str: str, heights_str: str, slug: str = "") -> Image.Image:
    W, H = IMAGE_SIZE
    img = Image.new("RGB", IMAGE_SIZE, COLORS['bg'])
    drw = ImageDraw.Draw(img)

    # 5% margins, 10% top margin
    margin = min(W, H) * 0.05
    top_margin = min(W, H) * 0.10
    
    # Available space for the plot
    plot_width = W - 2 * margin
    plot_height = H - top_margin - margin
    
    # Center the 5-bar plot
    bar_spacing = plot_width / 4  # Space between 5 bars
    xs = [margin + i * bar_spacing for i in range(5)]

    # y coordinates (bottom to top)
    base_y = H - margin  # Bottom baseline
    unit = plot_height / 4.0  # Height per level (0-4)

    # Draw vertical bars
    rgs = parse_rgs(rgs_str)
    levels = parse_heights(heights_str)
    half_width = LINE_WIDTH // 2
    
    for x, lvl in zip(xs, levels):
        top_y = base_y - unit * lvl
        drw.rectangle([
            (x - half_width, top_y),
            (x + half_width, base_y)
        ], fill=COLORS['bar'])

    # Draw connection lines for groups
    gmap = groups_from_rgs(rgs)
    
    for label, indices in gmap.items():
        if len(indices) < 2:
            continue
        
        # Find highest bar in group for connection level
        group_levels = [levels[i] for i in indices]
        max_level = max(group_levels)
        y_conn = base_y - unit * max_level - 16  # 16px above highest bar
        
        x1 = xs[min(indices)]
        x2 = xs[max(indices)]
        drw.rectangle([
            (x1 - half_width, y_conn - half_width),
            (x2 + half_width, y_conn + half_width)
        ], fill=COLORS['bar'])

    return img

# ----------------------------
# まとめて出力（任意）
# ----------------------------
def export_records(records: List[Tuple[str, str, str]],
                   out_dir: Path = Path("../fig_genjiko")):
    out_dir.mkdir(parents=True, exist_ok=True)
    for rgs, slug, h in records:
        img = draw_genji_panel(rgs, h, slug=slug)
        img.save(out_dir / f"{rgs}_{slug}.png")
    print(f"Generated {len(records)} figures in {out_dir}/")

def load_csv_data(csv_path: str = "../data/genji_ko.csv") -> List[Tuple[str, str, str]]:
    """Load all pattern data from CSV file"""
    records = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rgs = row['rgs'].strip()
                slug = row['slug'].strip()
                heights = row['heights'].strip()
                if rgs and slug and heights:  # Skip empty rows
                    records.append((rgs, slug, heights))
        print(f"Loaded {len(records)} patterns from {csv_path}")
        return records
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found, using samples")
        return [("11111", "tenarai", "44444")]
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return [("11111", "tenarai", "44444")]

if __name__ == "__main__":
    # Load all data from CSV
    all_records = load_csv_data()
    export_records(all_records)
