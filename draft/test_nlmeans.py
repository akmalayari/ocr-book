"""
test_nlmeans.py — Comparaison des modes de prétraitement du pipeline sur pages cibles.

Phase 1 (défaut)    : génère les images prétraitées.
Phase 2 (--ocr)     : lance l'OCR (preprocess → OCR → 2e passe → postprocess).

Sorties : output/nlmeans/

Usage :
    python draft/test_nlmeans.py
    python draft/test_nlmeans.py --ocr
    python draft/test_nlmeans.py --ocr median_and nlmeans_and
    python draft/test_nlmeans.py --list
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import patch  # noqa: F401
from config import Config
from ocr_client import ocr_image
from preprocess import preprocess_image, sauvola_binarize, median_and, nlmeans_and
from figure import process_figures

PHOTOS_DIR = Path(__file__).parent.parent / "photos"
OUT_DIR    = Path(__file__).parent.parent / "output" / "nlmeans"

DEFAULT_PAGES = [
    "page_4", 
    "page_9", 
    #"page_5",
    #"page_6"
    ]

OCR_CONFIGS: dict[str, str] = {
    "none":          "image originale",
    "blur_adaptive": "GaussianBlur(5,5) + adaptive C=15",
    "sauvola_and":   "AND(Sauvola w=51 k=0.3, blur_adaptive)",
    "median_and":    "medianBlur(3) + AND(Sauvola w=51, adaptive)",
    "nlmeans_and":   "fastNlMeans + AND(Sauvola w=51, adaptive)",
}

_PREPROCESS_FNS = {
    "blur_adaptive": preprocess_image,
    "sauvola_and":   sauvola_binarize,
    "median_and":    median_and,
    "nlmeans_and":   nlmeans_and,
}


def _preprocess(img_path: Path, config_name: str, cfg: Config, save_path: Path) -> Path:
    if config_name == "none":
        return img_path
    return _PREPROCESS_FNS[config_name](img_path, cfg, save_path)


# ── Phase 1 : visualisation ───────────────────────────────────────────────────

def phase1_visualize(images: list[Path]) -> None:
    print("── Génération des images prétraitées ───────────────────────")
    cfg = Config()
    for img_path in images:
        n = 0
        for config_name in OCR_CONFIGS:
            if config_name == "none":
                continue
            save_path = OUT_DIR / f"{img_path.stem}_{config_name}.jpg"
            _preprocess(img_path, config_name, cfg, save_path)
            n += 1
        print(f"  {img_path.name} → {n} fichiers")
    print(f"\nImages dans {OUT_DIR}")
    print("Pour lancer l'OCR : --ocr  (ou --ocr median_and nlmeans_and ...)")


# ── Phase 2 : OCR ─────────────────────────────────────────────────────────────

def phase2_ocr(images: list[Path], configs_to_run: list[str]) -> None:
    from nexaai import VLM
    cfg_base = Config(prompt_mode="layout")
    vlm = VLM.from_(model=cfg_base.model, quant=cfg_base.quant, config=cfg_base.to_model_config())

    results = []
    print(f"\n── OCR sur {len(configs_to_run)} config(s) × {len(images)} image(s) ──")

    for img_path in images:
        for config_name in configs_to_run:
            if config_name not in OCR_CONFIGS:
                print(f"  [SKIP] '{config_name}' inconnu. Configs : {', '.join(OCR_CONFIGS)}")
                continue

            cfg = Config(prompt_mode="layout", preprocess_mode=config_name)
            img_file = OUT_DIR / f"{img_path.stem}_{config_name}.jpg"
            preprocessed_path = _preprocess(img_path, config_name, cfg, img_file)

            out_md = OUT_DIR / f"{img_path.stem}_{config_name}.md"
            print(f"  [{img_path.stem} {config_name}] ...", end=" ", flush=True)
            row = {"page": img_path.stem, "config": config_name,
                   "looped": False, "words": 0, "latency": 0.0, "error": ""}
            try:
                text, metrics = ocr_image(preprocessed_path, vlm, cfg)

                if cfg.prompt_mode == "layout" and cfg.two_pass:
                    text, fig_metrics = process_figures(text, img_path, vlm, cfg, img_path.stem)
                    metrics["total_latency"] += fig_metrics["total_latency"]

                out_md.write_text(text, encoding="utf-8")
                row["words"]   = len(text.split())
                row["latency"] = metrics["total_latency"]
                row["looped"]  = metrics.get("looped", False)
                flag = " [BOUCLE]" if row["looped"] else ""
                print(f"{row['words']} mots ({row['latency']:.1f}s){flag}")
            except Exception as e:
                row["error"] = str(e)
                print(f"ERREUR: {e}")
            results.append(row)

    _write_ocr_report(results)


def _write_ocr_report(results: list[dict]) -> None:
    lines = [
        "# Rapport OCR\n",
        "| Page | Config | Boucle | Mots | Durée (s) | Note |",
        "|------|--------|--------|------|-----------|------|",
    ]
    for r in results:
        boucle = "**oui**" if r["looped"] else "non"
        lines.append(
            f"| {r['page']} | {r['config']} | {boucle} "
            f"| {r['words']} | {r['latency']:.1f} | {r.get('error', '')} |"
        )
    out = OUT_DIR / "ocr_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", nargs="+", default=DEFAULT_PAGES)
    parser.add_argument("--ocr", nargs="*", metavar="CONFIG",
                        help="Sans valeur = toutes les configs.")
    parser.add_argument("--list", action="store_true",
                        help="Afficher les configs disponibles et quitter.")
    args = parser.parse_args()

    if args.list:
        print("Configs disponibles :")
        for name, desc in OCR_CONFIGS.items():
            print(f"  {name:20s}  {desc}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images: list[Path] = []
    for name in args.pages:
        candidates = list(PHOTOS_DIR.glob(f"{name}.*"))
        if not candidates:
            print(f"[WARN] Aucune image trouvée pour '{name}'")
            continue
        images.append(candidates[0])
    if not images:
        print("Aucune image.")
        sys.exit(1)

    phase1_visualize(images)

    if args.ocr is None:
        return

    configs_to_run = list(OCR_CONFIGS.keys()) if len(args.ocr) == 0 else args.ocr
    phase2_ocr(images, configs_to_run)


if __name__ == "__main__":
    main()
