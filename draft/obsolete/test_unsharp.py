"""
test_unsharp.py — Unsharp Mask (ordre inversé) vs preprocess actuel.

Unsharp Mask inversé : blurred est la base, on y soustrait le bruit haute fréquence.
    sharpened = addWeighted(blurred, 1+alpha, img, -alpha)
    => (1+alpha)*blurred - alpha*img = blurred - alpha*(img - blurred)

Affichage par image :
    original | binarize (preprocess actuel) | unsharp_inv+binarize (configs)

Usage :
    python draft/test_unsharp.py
    python draft/test_unsharp.py --images photos/page_2.jpg photos/page_5.jpg
    python draft/test_unsharp.py --save
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# ── Configs à comparer ───────────────────────────────────────────────────────

UNSHARP_CONFIGS = [
    {"ksize": (3, 3), "sigma": 1.0, "alpha": 0.5},
    {"ksize": (5, 5), "sigma": 1.0, "alpha": 0.5},
    {"ksize": (5, 5), "sigma": 1.0, "alpha": 1.0},
    {"ksize": (5, 5), "sigma": 2.0, "alpha": 1.0},
]

# ── Binarisation (identique à preprocess.py / config.py) ────────────────────

BINARIZE_BLOCK_SIZE = 31
BINARIZE_C = 10


def binarize(img: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        BINARIZE_BLOCK_SIZE, BINARIZE_C,
    )


# ── Unsharp Mask inversé ─────────────────────────────────────────────────────

def unsharp_inv(img: np.ndarray, ksize=(5, 5), sigma=1.0, alpha=1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, ksize, sigma)
    return cv2.addWeighted(blurred, 1 + alpha, img, -alpha, 0)


# ── Assemblage ───────────────────────────────────────────────────────────────

def make_strip(label: str, img: np.ndarray, h: int, w: int) -> np.ndarray:
    strip = np.zeros((h + 30, w, 3), dtype=np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    strip[30:, :] = bgr
    cv2.putText(strip, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 0), 1, cv2.LINE_AA)
    return strip


GAUSSIAN_BLUR_KSIZE = (5, 5)


def preprocess_current(img: np.ndarray) -> np.ndarray:
    """GaussianBlur(5,5) + binarize — preprocess de référence validé."""
    blurred = cv2.GaussianBlur(img, GAUSSIAN_BLUR_KSIZE, 0)
    return binarize(blurred)


def make_comparison(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    strips = [make_strip("original", gray, h, w),
              make_strip("gaussianblur+binarize (actuel)", preprocess_current(gray), h, w)]
    for cfg in UNSHARP_CONFIGS:
        result = unsharp_inv(gray, **cfg)
        label = f"inv k={cfg['ksize'][0]} s={cfg['sigma']} a={cfg['alpha']} +binarize"
        strips.append(make_strip(label, binarize(result), h, w))
    return np.hstack(strips)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", type=Path,
                        default=sorted(Path("photos").glob("*.jpg")))
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    out_dir = Path("draft/unsharp_out")
    if args.save:
        out_dir.mkdir(exist_ok=True)

    for img_path in args.images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] {img_path} illisible")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        comparison = make_comparison(gray)
        print(f"{img_path.name}")
        if args.save:
            cv2.imwrite(str(out_dir / f"{img_path.stem}.jpg"), comparison)
        else:
            cv2.imshow(img_path.name, comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
