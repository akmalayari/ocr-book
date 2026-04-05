"""
Coupe chaque image en deux moitiés (gauche/droite), applique page-dewarp sur chacune,
puis recombine. Compare avec le binarize seul.

Sortie : draft/dewarped/
  _binarize.jpg       — méthode actuelle (image entière)
  _dewarp.jpg         — dewarp sur chaque moitié + recombinaison

Lancer depuis la racine du projet : python draft/test_dewarp.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "src")
from config import Config
from preprocess import preprocess_image

_cfg = Config(binarize_block_size=31, binarize_c=10)
from page_dewarp.cli import Config as DewarpConfig
from page_dewarp.image import WarpedImage

PHOTOS = Path("photos")
OUT = Path("draft/dewarped")
TMP = Path("draft/dewarped/tmp")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

OUT.mkdir(exist_ok=True)
TMP.mkdir(exist_ok=True)


def dewarp_half(img: np.ndarray, name: str) -> np.ndarray | None:
    """Sauvegarde une moitié dans un fichier temporaire, dewarp, retourne l'image."""
    tmp_in = TMP / f"{name}.jpg"
    cv2.imwrite(str(tmp_in), img)

    w = WarpedImage(str(tmp_in), config=DewarpConfig(NO_BINARY=1, PAGE_MARGIN_X=0, PAGE_MARGIN_Y=0))
    thresh_path = Path(f"{tmp_in.stem}_thresh.png")

    if not w.written:
        thresh_path.unlink(missing_ok=True)
        tmp_in.unlink(missing_ok=True)
        return None

    result = cv2.imread(str(thresh_path))
    thresh_path.unlink(missing_ok=True)
    tmp_in.unlink(missing_ok=True)
    return result


images = sorted(p for p in PHOTOS.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)

for src in images:
    print(f"\n--- {src.name} ---")

    # 1. Binarize seul (référence)
    tmp = preprocess_image(src, _cfg)
    shutil.copy(str(tmp), str(OUT / (src.stem + "_binarize.jpg")))
    tmp.unlink()
    print(f"  binarize → {src.stem}_binarize.jpg")

    # 2. Coupe à mi-largeur, dewarp chaque moitié, recombinaison
    orig = cv2.imread(str(src))
    mid = orig.shape[1] // 2
    left_orig = orig[:, :mid]
    right_orig = orig[:, mid:]

    left_dw = dewarp_half(left_orig, src.stem + "_left")
    right_dw = dewarp_half(right_orig, src.stem + "_right")

    if left_dw is not None and right_dw is not None:
        # Redimensionner à la même hauteur pour pouvoir concaténer
        h = min(left_dw.shape[0], right_dw.shape[0])
        left_dw = cv2.resize(left_dw, (left_dw.shape[1], h))
        right_dw = cv2.resize(right_dw, (right_dw.shape[1], h))
        combined = np.hstack([left_dw, right_dw])
        cv2.imwrite(str(OUT / (src.stem + "_dewarp.jpg")), combined)
        print(f"  dewarp   → {src.stem}_dewarp.jpg")
    else:
        print(f"  dewarp   [échec partiel] left={left_dw is not None} right={right_dw is not None}")

TMP.rmdir()
