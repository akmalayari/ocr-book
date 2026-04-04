"""
test_prompts.py — Test de prompts custom sur une image ou un dossier.

Usage:
    python draft/test_prompts.py <image_or_dir> [--out DIR] [--preprocess none|binarize] [--quant q8_0|bf16]

Configurer les prompts à tester directement dans PROMPTS ci-dessous.
"""

import sys
from pathlib import Path

# patch doit être importé en premier
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import patch  # noqa: F401

from nexaai import VLM

from config import Config
from ocr_client import ocr_image

sys.path.insert(0, str(Path(__file__).parent))
from compare import compare as compare_files

# ── Prompts à tester ─────────────────────────────────────────────────────────
PROMPTS = {
    #"plain":  "Convert the document to markdown.",
    "layout": "<|grounding|>Convert the document to markdown.",
    #"rec" :  "Locate <|ref|>A figure or graph<|/ref|> in the image.",
    #"rec2":      "Locate <|ref|>everything<|/ref|> in the image."
}

# ── Paramètres ────────────────────────────────────────────────────────────────
INPUT       = "photos/page_10.jpg"           # image ou dossier
OUT_DIR     = "output/prompt_results"
PREPROCESS  = "binarize"              # "none" ou "binarize"
QUANT       = "bf16"              # "bf16" ou "q8_0"
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)



def write_result(out_dir: Path, image_path: Path, prompt_name: str, prompt_text: str, text: str, latency: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    out_file = out_dir / f"{stem}__{prompt_name}.md"
    content = f"# {stem} — prompt: {prompt_name}\n\n"
    content += f"**Prompt:** `{prompt_text}`  \n"
    content += f"**Latency:** {latency:.1f}s\n\n"
    content += "---\n\n"
    content += text + "\n"
    out_file.write_text(content, encoding="utf-8")
    print(f"  → {out_file.relative_to(Path.cwd())}  ({latency:.1f}s, {len(text)} chars)")


def main():
    input_path = Path(INPUT)
    if not input_path.exists():
        sys.exit(f"Erreur : '{input_path}' introuvable")

    images = collect_images(input_path)
    if not images:
        sys.exit(f"Aucune image trouvée dans '{input_path}'")

    out_dir = Path(OUT_DIR)
    cfg = Config(preprocess_mode=PREPROCESS, quant=QUANT)

    print(f"Chargement du modèle ({cfg.model}, {cfg.quant})…")
    vlm = VLM.from_(model=cfg.model, quant=cfg.quant, config=cfg.to_model_config())

    print(f"\n{len(images)} image(s) × {len(PROMPTS)} prompt(s)\n")

    for image_path in images:
        print(f"[{image_path.name}]")
        for prompt_label, prompt_text in PROMPTS.items():
            try:
                cfg_prompt = Config(preprocess_mode=PREPROCESS, quant=QUANT, prompt_mode=prompt_label)
                pp_save = out_dir / f"{image_path.stem}__preprocess.jpg" if PREPROCESS == "binarize" else None
                text, metrics = ocr_image(image_path, vlm, cfg_prompt, preprocess_save_path=pp_save)
                write_result(out_dir, image_path, prompt_label, prompt_text, text, metrics["total_latency"])
            except Exception as e:
                print(f"  ✗ {prompt_label}: {e}")

    if "plain" in PROMPTS and "layout" in PROMPTS:
        print("\n── Comparaisons plain vs layout ──")
        compare_dir = out_dir / "compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        for image_path in images:
            stem = image_path.stem
            path_plain  = out_dir / f"{stem}__plain.md"
            path_layout = out_dir / f"{stem}__layout.md"
            if path_plain.exists() and path_layout.exists():
                out_cmp = compare_dir / f"{stem}__plain_vs_layout.md"
                compare_files(path_plain, path_layout, mode="sentence", out_path=out_cmp)
            else:
                missing = [p for p in (path_plain, path_layout) if not p.exists()]
                print(f"  ✗ {stem}: fichier(s) manquant(s): {[p.name for p in missing]}")


if __name__ == "__main__":
    main()
