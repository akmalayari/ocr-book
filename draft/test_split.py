"""
Détection de la pliure et découpe en deux demi-pages.
Méthode : érosion morphologique verticale dans la bande centrale → colonne la plus noire.
Sortie : draft/split/IMG_left.jpg et IMG_right.jpg

Lancer depuis la racine du projet : python draft/test_split.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "src")
from preprocess import preprocess_image

PHOTOS = Path("photos")
OUT = Path("draft/split")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

OUT.mkdir(exist_ok=True)


def find_spine(bw: np.ndarray) -> int:
    """
    Détecte la colonne de la pliure dans une image binarisée.
    Cherche le centre de la bande noire verticale la plus longue,
    dans la bande centrale (tiers central en x).
    """
    h, w = bw.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h * 3 // 4))
    eroded = cv2.erode(bw, kernel)

    x_start = w // 3
    x_end = 2 * w // 3
    band = eroded[:, x_start:x_end]

    col_sums = band.sum(axis=0)
    threshold = col_sums.min() * 1.2  # colonnes proches du minimum = bande noire
    dark_cols = np.where(col_sums <= threshold)[0]
    spine_in_band = int(dark_cols.mean())
    return x_start + spine_in_band


images = sorted(p for p in PHOTOS.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)

for src in images:
    print(f"\n--- {src.name} ---")

    tmp = preprocess_image(src, block_size=31, c=10, blur_ksize=5)
    bw = cv2.imread(str(tmp), cv2.IMREAD_GRAYSCALE)
    tmp.unlink()

    spine_x = find_spine(bw)
    print(f"  pliure détectée à x={spine_x} (largeur={bw.shape[1]})")

    orig = cv2.imread(str(src))
    left = orig[:, :spine_x]
    right = orig[:, spine_x:]

    cv2.imwrite(str(OUT / (src.stem + "_left.jpg")), left)
    cv2.imwrite(str(OUT / (src.stem + "_right.jpg")), right)

    bw_left = bw[:, :spine_x]
    bw_right = bw[:, spine_x:]
    cv2.imwrite(str(OUT / (src.stem + "_left_bw.jpg")), bw_left)
    cv2.imwrite(str(OUT / (src.stem + "_right_bw.jpg")), bw_right)
    print(f"  → {src.stem}_left.jpg / _right.jpg / _left_bw.jpg / _right_bw.jpg")
