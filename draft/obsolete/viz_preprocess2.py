"""
viz_preprocess2.py — Compare différentes variantes de prétraitement sur les pages testées.
Sortie : draft/preprocess_viz2/<page>/<variant>.jpg

Variantes :
  01_equalizehist
  02_gaussianblur
  03_equalizehist_binarize
  04_gaussianblur_binarize
  05_bg_divide
  06_bg_divide_binarize
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from config import Config

PHOTOS  = Path(__file__).parent.parent / "photos"
OUT_DIR = Path(__file__).parent / "preprocess_viz2"
PAGES   = ["page_1.jpg", "page_2.jpg", "page_3.jpg", "page_4.jpg", "page_5.jpg"]

cfg = Config()
BLOCK = cfg.binarize_block_size
C     = cfg.binarize_c

def binarize(gray):
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        BLOCK, C,
    )

for name in PAGES:
    src = PHOTOS / name
    if not src.exists():
        print(f"[skip] {name} introuvable")
        continue

    stem    = Path(name).stem
    page_dir = OUT_DIR / stem
    page_dir.mkdir(parents=True, exist_ok=True)

    img  = cv2.imread(str(src))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. equalizeHist seul
    eq = cv2.equalizeHist(gray)
    cv2.imwrite(str(page_dir / "01_equalizehist.jpg"), eq)

    # 2. GaussianBlur seul
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(str(page_dir / "02_gaussianblur.jpg"), blur)

    # 3. equalizeHist + binarize_adaptive
    cv2.imwrite(str(page_dir / "03_equalizehist_binarize.jpg"), binarize(eq))

    # 4. GaussianBlur + binarize_adaptive
    cv2.imwrite(str(page_dir / "04_gaussianblur_binarize.jpg"), binarize(blur))

    # 5. bg_divide (normalisation illumination) : fond estimé par grand blur
    background = cv2.GaussianBlur(gray, (101, 101), 0)
    divided    = cv2.divide(gray, background, scale=255)
    cv2.imwrite(str(page_dir / "05_bg_divide.jpg"), divided)

    # 6. bg_divide + binarize_adaptive
    cv2.imwrite(str(page_dir / "06_bg_divide_binarize.jpg"), binarize(divided))

    print(f"[ok] {stem}")
