"""
test_binarize_grid.py — Grid test des paramètres block_size / C de la binarisation adaptive.

Phase 1 (défaut) : sauvegarde les images binarisées pour inspection visuelle.
Phase 2 (--ocr)  : lance l'OCR sur les configs spécifiées.

Sorties : output/binarize_grid/{page}_{block_size}_{c}.jpg
          output/binarize_grid/{page}_{block_size}_{c}.md  (si --ocr)

Usage :
    python draft/test_binarize_grid.py
    python draft/test_binarize_grid.py --ocr 31_10 21_5
    python draft/test_binarize_grid.py --pages page_5 page_6
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import patch  # noqa: F401
from config import Config
from preprocess import preprocess_image
from ocr_client import ocr_image

PHOTOS_DIR = Path(__file__).parent.parent / "photos"
OUT_DIR    = Path(__file__).parent.parent / "output" / "binarize_grid"

BLOCK_SIZES = [11, 21, 31, 41, 51]
C_VALUES    = [5, 10, 15, 20]
DEFAULT_PAGES = ["page_5", "page_6"]


def laplacian_variance(image_path: Path) -> float:
    img  = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", nargs="+", default=DEFAULT_PAGES,
                        help="Noms de pages sans extension (ex: page_5 page_6)")
    parser.add_argument("--ocr", nargs="*", metavar="BS_C",
                        help="Lancer l'OCR sur ces configs (ex: 31_10 21_5). Sans valeur = toutes.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Résoudre les chemins d'images
    images = []
    for name in args.pages:
        candidates = list(PHOTOS_DIR.glob(f"{name}.*"))
        if not candidates:
            print(f"[WARN] Aucune image trouvée pour '{name}' dans {PHOTOS_DIR}")
            continue
        images.append(candidates[0])

    if not images:
        print("Aucune image à traiter.")
        sys.exit(1)

    # Afficher la variance Laplacien (calibration seuil de flou)
    print("── Variance Laplacien (netteté) ──────────────────────────")
    for img_path in images:
        var = laplacian_variance(img_path)
        print(f"  {img_path.name:20s}  {var:.1f}")
    print()

    # Phase 1 : générer toutes les images binarisées
    print("── Génération des images binarisées ──────────────────────")
    for img_path in images:
        for bs in BLOCK_SIZES:
            for c in C_VALUES:
                save_path = OUT_DIR / f"{img_path.stem}_{bs}_{c}.jpg"
                preprocess_image(
                    img_path,
                    block_size=bs,
                    c=c,
                    blur_ksize=5,
                    blur_sigma=0.0,
                    save_path=save_path,
                )
        print(f"  {img_path.name} → {len(BLOCK_SIZES) * len(C_VALUES)} fichiers dans {OUT_DIR}")

    # Phase 2 : OCR optionnel
    if args.ocr is None:
        print("\nPhase 1 terminée. Inspecter output/binarize_grid/ visuellement.")
        print("Pour lancer l'OCR : --ocr 31_10 21_5  (ou --ocr sans valeur pour tout tester)")
        return

    # Déterminer les configs à tester
    if len(args.ocr) == 0:
        ocr_configs = [(bs, c) for bs in BLOCK_SIZES for c in C_VALUES]
    else:
        ocr_configs = []
        for token in args.ocr:
            try:
                bs, c = token.split("_")
                ocr_configs.append((int(bs), int(c)))
            except ValueError:
                print(f"[WARN] Format invalide '{token}', attendu BS_C (ex: 31_10)")

    if not ocr_configs:
        print("Aucune config OCR valide.")
        return

    print(f"\n── OCR sur {len(ocr_configs)} config(s) × {len(images)} image(s) ──────────")
    import patch as _patch  # already imported, VLM needs the model loaded once
    from nexaai import VLM
    cfg_base = Config(prompt_mode="layout", preprocess_mode="none")  # preprocess déjà fait
    vlm = VLM.from_(model=cfg_base.model, quant=cfg_base.quant, config=cfg_base.to_model_config())

    for img_path in images:
        for bs, c in ocr_configs:
            binarized = OUT_DIR / f"{img_path.stem}_{bs}_{c}.jpg"
            if not binarized.exists():
                print(f"  [SKIP] {binarized.name} non trouvé")
                continue
            out_md = OUT_DIR / f"{img_path.stem}_{bs}_{c}.md"
            print(f"  [{img_path.stem} bs={bs} c={c}] ...", end=" ", flush=True)
            try:
                text, metrics = ocr_image(binarized, vlm, cfg_base)
                out_md.write_text(text, encoding="utf-8")
                words = len(text.split())
                print(f"{words} mots ({metrics['total_latency']:.1f}s)")
            except Exception as e:
                print(f"ERREUR: {e}")

    print("\nPhase 2 terminée.")


if __name__ == "__main__":
    main()
