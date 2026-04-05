"""
test_crop_vs_preprocess.py — Compare deux ordres preprocess/crop pour la passe 2 :
  A : binarize(page complète) → crop → parse
  B : crop(page originale) → binarize(crop) → parse

Bbox du graphique Document 1 connue depuis le layout sur page_6 : (106, 98, 412, 390)

Usage (depuis la racine du projet) :
    python draft/test_crop_vs_preprocess.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import patch  # noqa: F401 — doit précéder tout import nexaai
from nexaai import VLM

from config import Config
from preprocess import preprocess_image
from figure import crop_image
from ocr_client import ocr_image

IMAGE_PATH = Path("photos/page_6.jpg")
OUT_DIR    = Path("output/draft/test_crop_vs_preprocess")

# Bbox connue du graphique Document 1 (issue du layout sur page_6)
BBOX = (106, 98, 412, 390)


def main():
    cfg = Config(prompt_mode="parse")

    print("Chargement VLM...")
    vlm = VLM.from_(model=cfg.model, quant=cfg.quant, config=cfg.to_model_config())

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Variante A : binarize(page) → crop → parse ───────────────────────────
    print("\n--- Variante A : binarize(page) → crop → parse ---")
    binarized_page = preprocess_image(
        IMAGE_PATH,
        cfg.binarize_block_size, cfg.binarize_c,
        cfg.blur_ksize, cfg.blur_sigma,
        save_path=OUT_DIR / "page_binarized.jpg",
    )
    crop_a = OUT_DIR / "crop_A.jpg"
    crop_image(binarized_page, BBOX, crop_a)
    text_a, metrics_a = ocr_image(crop_a, vlm, cfg)
    print(f"  latency: {metrics_a['total_latency']:.1f}s")
    (OUT_DIR / "result_A__binarize_page_then_crop.md").write_text(text_a, encoding="utf-8")
    print(f"  → result_A__binarize_page_then_crop.md")

    # ── Variante B : crop → binarize(crop) → parse ───────────────────────────
    print("\n--- Variante B : crop → binarize(crop) → parse ---")
    crop_b_raw = OUT_DIR / "crop_B_raw.jpg"
    crop_image(IMAGE_PATH, BBOX, crop_b_raw)
    binarized_crop = preprocess_image(
        crop_b_raw,
        cfg.binarize_block_size, cfg.binarize_c,
        cfg.blur_ksize, cfg.blur_sigma,
        save_path=OUT_DIR / "crop_B_binarized.jpg",
    )
    text_b, metrics_b = ocr_image(binarized_crop, vlm, cfg)
    print(f"  latency: {metrics_b['total_latency']:.1f}s")
    (OUT_DIR / "result_B__crop_then_binarize.md").write_text(text_b, encoding="utf-8")
    print(f"  → result_B__crop_then_binarize.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
