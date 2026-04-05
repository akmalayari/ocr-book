"""
prompt_test.py — Test de chaque prompt sur des pages représentatives.

Génère un fichier par image dans draft/results/ avec le texte produit par
chaque prompt. Utiliser compare.py pour comparer deux fichiers de résultats.

Usage (depuis le repo root):
    python draft/prompt_test.py
    python draft/prompt_test.py --no-preprocess
    python draft/prompt_test.py --images photos/page_1.jpg photos/page_3.jpg
"""
import sys
import argparse
import time
from pathlib import Path

# --- path setup so we can import from src/ ---
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import patch  # noqa: F401  (monkey-patch avant tout import nexaai)
from nexaai import VLM
from config import Config
from ocr_client import ocr_image

RESULTS_DIR = Path(__file__).parent / "prompt_results" / "new_tests"

TEST_IMAGES = [
    "photos/page_1.jpg",  # texte simple, mauvais éclairage
    "photos/page_2.jpg",  # même texte, éclairage normal
    "photos/page_3.jpg",  # texte + tableau textuel
    "photos/page_4.jpg",  # texte + tableau numérique
    "photos/page_5.jpg",  # texte + tableau + graphe
]

PROMPTS = Config().PROMPTS.keys()


def metrics(text: str) -> str:
    words     = len(text.split())
    md_tokens = sum(text.count(c) for c in ("#", "|", "*", "_", "`"))
    return f"{len(text)} chars, {words} words, {md_tokens} md-tokens"


def run(image_path: Path, vlm, preprocess_mode: str) -> dict[str, str]:
    results = {}
    for mode in PROMPTS:
        cfg = Config(prompt_mode=mode, preprocess_mode=preprocess_mode)
        print(f"  [{mode}]", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            text, _ = ocr_image(image_path, vlm, cfg)
        except Exception as e:
            text = f"ERROR: {e}"
        elapsed = time.perf_counter() - t0
        print(f"{elapsed:.1f}s — {metrics(text)}")
        results[mode] = text
    return results


def write_results(image_path: Path, results: dict[str, str]):
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / (image_path.stem + ".md")
    lines = [f"# {image_path.name}\n"]
    for mode, text in results.items():
        lines += [f"## {mode}\n", text.strip(), "\n"]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Désactive la binarisation (preprocess_mode=none)")
    parser.add_argument("--images", nargs="+", default=TEST_IMAGES,
                        help="Images à tester (chemins depuis le repo root)")
    args = parser.parse_args()

    preprocess_mode = "none" if args.no_preprocess else "binarize"
    repo_root = Path(__file__).parent.parent

    cfg0 = Config()
    print("Chargement du modèle…")
    vlm = VLM.from_(model=cfg0.model, config=cfg0.to_model_config())

    for img_str in args.images:
        image_path = repo_root / img_str
        if not image_path.exists():
            print(f"[SKIP] {image_path} introuvable")
            continue
        print(f"\n{image_path.name} (preprocess={preprocess_mode})")
        results = run(image_path, vlm, preprocess_mode)
        write_results(image_path, results)


if __name__ == "__main__":
    main()
