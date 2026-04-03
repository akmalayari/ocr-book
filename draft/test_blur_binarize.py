"""
test_blur_binarize.py — OCR avec GaussianBlur + binarize_adaptive sur toutes les photos.
Sortie : draft/blur_binarize_out/<prompt_mode>/<page>.md

Usage : python draft/test_blur_binarize.py [--prompt plain|layout|describe|parse]
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import patch  # noqa: F401  — monkey-patch nexaai Windows
from nexaai import VLM
from nexaai.nexa_sdk.types import GenerationConfig, VlmChatMessage, VlmContent
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="plain", choices=["plain", "layout", "describe", "parse"])
args = parser.parse_args()

PHOTOS  = Path(__file__).parent.parent / "photos"
OUT_DIR = Path(__file__).parent / "blur_binarize_out" / args.prompt
OUT_DIR.mkdir(parents=True, exist_ok=True)

cfg = Config(prompt_mode=args.prompt)


def preprocess(image_path: Path) -> Path:
    img  = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw   = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        cfg.binarize_block_size, cfg.binarize_c,
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, bw)
    return Path(tmp.name)


pages = sorted(PHOTOS.glob("*.jpg"))
if not pages:
    print("Aucune image trouvée dans photos/")
    sys.exit(1)

print(f"Chargement du modèle {cfg.model}...")
vlm = VLM.from_(model=cfg.model, config=cfg.to_model_config())

for src in pages:
    print(f"\n[{src.name}] prétraitement...")
    input_path = preprocess(src)

    msg = VlmChatMessage(
        role="user",
        contents=[
            VlmContent(type="image", text=str(input_path.resolve())),
            VlmContent(type="text",  text=cfg.prompt),
        ],
    )
    formatted  = vlm.apply_chat_template([msg])
    gen_config = GenerationConfig(
        image_paths=[str(input_path.resolve())],
        max_tokens=cfg.max_tokens,
#        sampler_config=cfg.to_sampler_config(),
    )

    print(f"[{src.name}] génération...")
    result = vlm.generate(formatted, config=gen_config)
    text   = result.full_text.strip() if result.full_text else ""

    out = OUT_DIR / f"{src.stem}.md"
    out.write_text(text, encoding="utf-8")
    print(f"[{src.name}] → {out} ({len(text)} caractères)")
