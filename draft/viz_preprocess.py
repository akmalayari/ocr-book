"""
viz_preprocess.py — Enregistre l'image originale vs binarisée pour chaque page testée.
Usage : python draft/viz_preprocess.py
Sortie : draft/preprocess_viz/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from preprocess import preprocess_image
from config import Config

PHOTOS  = Path(__file__).parent.parent / "photos"
OUT_DIR = Path(__file__).parent / "preprocess_viz"
PAGES   = ["page_1.jpg", "page_2.jpg", "page_3.jpg", "page_4.jpg", "page_5.jpg"]

OUT_DIR.mkdir(exist_ok=True)
cfg = Config()

for name in PAGES:
    src = PHOTOS / name
    if not src.exists():
        print(f"[skip] {name} introuvable")
        continue

    binarized_path = preprocess_image(src, cfg.binarize_block_size, cfg.binarize_c)
    binarized      = cv2.imread(str(binarized_path))

    stem = Path(name).stem
    cv2.imwrite(str(OUT_DIR / f"{stem}_binarized.jpg"), binarized)
    print(f"[ok] {stem}")
