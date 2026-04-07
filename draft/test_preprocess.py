"""
test_preprocess.py — Comparaison des prétraitements légers (sans binarisation) sur pages cibles.

Phase 1 (défaut)    : génère les images prétraitées.
Phase 2 (--ocr)     : lance l'OCR (preprocess → OCR → 2e passe → postprocess).

Sorties : output/preprocess/

Usage :
    python draft/test_preprocess.py
    python draft/test_preprocess.py --ocr
    python draft/test_preprocess.py --ocr nlmeans sesr
    python draft/test_preprocess.py --device cpu
    python draft/test_preprocess.py --list
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import patch  # noqa: F401
from config import Config
from ocr_client import ocr_image
from preprocess import nlmeans, bilateral
from figure import process_figures
from realesrgan_sesr import generate_sr

PHOTOS_DIR     = Path(__file__).parent.parent / "photos"
OUT_DIR        = Path(__file__).parent.parent / "output" / "preprocess"
DEFAULT_PAGES  = ["page_4", "page_5", "page_6", "page_9"]
DEFAULT_DEVICE = "npu"

OCR_CONFIGS: dict[str, str] = {
    "none":      "image originale",
    "nlmeans":   "fastNlMeans(h=noise_level)",
    "bilateral": "bilateralFilter(d=9, σ=75)",
    "sesr":      "SESR-M7 x2 → resize",
    "esrgan":    "RealESRGAN x4 → resize",
}

SR_CONFIGS = {"sesr", "esrgan"}

_PREPROCESS_FNS = {
    "nlmeans":   nlmeans,
    "bilateral": bilateral,
}


def _preprocess(img_path: Path, config_name: str, cfg: Config, save_path: Path, device: str,
                reuse: bool = False) -> Path:
    if config_name == "none":
        return img_path
    if reuse and save_path.exists():
        return save_path
    if config_name in SR_CONFIGS:
        return generate_sr(img_path, config_name, device, save_path)
    return _PREPROCESS_FNS[config_name](img_path, cfg, save_path)


# ── Phase 1 : visualisation ───────────────────────────────────────────────────

def phase1_visualize(images: list[Path], device: str) -> None:
    print("── Génération des images prétraitées ───────────────────────")
    cfg = Config()
    for img_path in images:
        n = 0
        for config_name in OCR_CONFIGS:
            if config_name == "none":
                continue
            save_path = OUT_DIR / f"{img_path.stem}_{config_name}.jpg"
            try:
                _preprocess(img_path, config_name, cfg, save_path, device)
                n += 1
            except RuntimeError as e:
                print(f"  [ERREUR] {img_path.stem} {config_name}: {e}")
        print(f"  {img_path.name} → {n} fichiers")
    print(f"\nImages dans {OUT_DIR}")
    print("Pour lancer l'OCR : --ocr  (ou --ocr nlmeans sesr ...)")


# ── Phase 2 : OCR ─────────────────────────────────────────────────────────────

def phase2_ocr(images: list[Path], configs_to_run: list[str], device: str) -> None:
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

            cfg      = Config(prompt_mode="layout")
            img_file = OUT_DIR / f"{img_path.stem}_{config_name}.jpg"
            out_md   = OUT_DIR / f"{img_path.stem}_{config_name}.md"
            print(f"  [{img_path.stem} {config_name}] ...", end=" ", flush=True)

            row = {"page": img_path.stem, "config": config_name,
                   "looped": False, "words": 0, "latency": 0.0, "error": ""}
            try:
                preprocessed_path = _preprocess(img_path, config_name, cfg, img_file, device, reuse=True)
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
    parser.add_argument("--pages",  nargs="+", default=DEFAULT_PAGES)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["npu", "cpu"],
                        help="Device pour les modèles SR (défaut: npu).")
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

    if args.ocr is None:
        phase1_visualize(images, args.device)
        return

    configs_to_run = list(OCR_CONFIGS.keys()) if len(args.ocr) == 0 else args.ocr
    phase2_ocr(images, configs_to_run, args.device)


if __name__ == "__main__":
    main()
