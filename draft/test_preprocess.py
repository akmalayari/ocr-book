"""
test_preprocess.py — Baseline texte pur sur images clean (none / nlmeans / sesr).

Phase 1 (défaut)    : génère les images prétraitées.
Phase 2 (--ocr)     : lance l'OCR (preprocess → OCR → 2e passe → postprocess).

Sorties : output/preprocess_clean/

Usage :
    python draft/test_preprocess.py
    python draft/test_preprocess.py --ocr
    python draft/test_preprocess.py --ocr nlmeans sesr
    python draft/test_preprocess.py --device cpu
    python draft/test_preprocess.py --list
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import patch  # noqa: F401
from config import Config
from ocr_client import ocr_image
from preprocess import nlmeans
from figure import process_figures
from realesrgan_sesr import generate_sr

PHOTOS_DIR     = Path(__file__).parent.parent / "photos"
OUT_DIR        = Path(__file__).parent.parent / "output" / "preprocess_clean"
DEFAULT_PAGES  = ["page_4_clean", "page_5-6_clean", "page_9_clean", "page_10_clean"]
DEFAULT_DEVICE = "npu"

OCR_CONFIGS: dict[str, str] = {
    "none":    "image originale",
    "nlmeans": "fastNlMeans(h=noise_level)",
    "sesr":    "SESR-M7 x2 → resize",
}

SR_CONFIGS = {"sesr"}

_PREPROCESS_FNS = {
    "nlmeans": nlmeans,
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

PREPROCESS_TIMES_FILE = OUT_DIR / "preprocess_times.json"


def _load_preprocess_times() -> dict:
    if PREPROCESS_TIMES_FILE.exists():
        return json.loads(PREPROCESS_TIMES_FILE.read_text(encoding="utf-8"))
    return {}


def _save_preprocess_times(times: dict) -> None:
    PREPROCESS_TIMES_FILE.write_text(json.dumps(times, indent=2), encoding="utf-8")


def phase1_visualize(images: list[Path], device: str) -> None:
    print("── Génération des images prétraitées ───────────────────────")
    cfg   = Config()
    times = _load_preprocess_times()
    for img_path in images:
        n = 0
        for config_name in OCR_CONFIGS:
            if config_name == "none":
                continue
            save_path = OUT_DIR / f"{img_path.stem}_{config_name}.jpg"
            key = f"{img_path.stem}_{config_name}"
            try:
                t0 = time.perf_counter()
                _preprocess(img_path, config_name, cfg, save_path, device)
                elapsed = time.perf_counter() - t0
                times[key] = elapsed
                print(f"  {img_path.stem} [{config_name}] → {elapsed:.1f}s")
                n += 1
            except RuntimeError as e:
                print(f"  [ERREUR] {img_path.stem} {config_name}: {e}")
        print(f"  {img_path.name} → {n} fichiers")
    _save_preprocess_times(times)
    print(f"\nImages dans {OUT_DIR}")
    print("Pour lancer l'OCR : --ocr  (ou --ocr nlmeans sesr ...)")


# ── Phase 2 : OCR ─────────────────────────────────────────────────────────────

def phase2_ocr(images: list[Path], configs_to_run: list[str], device: str) -> None:
    from nexaai import VLM
    cfg_base = Config(prompt_mode="layout")
    vlm = VLM.from_(model=cfg_base.model, quant=cfg_base.quant, config=cfg_base.to_model_config())

    preprocess_times = _load_preprocess_times()
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

            key = f"{img_path.stem}_{config_name}"
            row = {"page": img_path.stem, "config": config_name,
                   "looped": False, "words": 0, "preprocess_s": 0.0, "ocr_s": 0.0, "error": ""}
            try:
                t0 = time.perf_counter()
                preprocessed_path = _preprocess(img_path, config_name, cfg, img_file, device, reuse=True)
                row["preprocess_s"] = preprocess_times.get(key, time.perf_counter() - t0)

                t0 = time.perf_counter()
                text, metrics = ocr_image(preprocessed_path, vlm, cfg)
                if cfg.prompt_mode == "layout" and cfg.two_pass:
                    text, fig_metrics = process_figures(text, img_path, vlm, cfg, img_path.stem)
                    metrics["total_latency"] += fig_metrics["total_latency"]
                row["ocr_s"] = time.perf_counter() - t0

                out_md.write_text(text, encoding="utf-8")
                row["words"]  = len(text.split())
                row["looped"] = metrics.get("looped", False)
                flag = " [BOUCLE]" if row["looped"] else ""
                print(f"{row['words']} mots  pre={row['preprocess_s']:.1f}s  ocr={row['ocr_s']:.1f}s{flag}")
            except Exception as e:
                row["error"] = str(e)
                print(f"ERREUR: {e}")
            results.append(row)

    _write_ocr_report(results)


def _write_ocr_report(results: list[dict]) -> None:
    lines = [
        "# Rapport OCR\n",
        "| Page | Config | Boucle | Mots | Preprocess (s) | OCR (s) | Total (s) | Note |",
        "|------|--------|--------|------|----------------|---------|-----------|------|",
    ]
    for r in results:
        boucle = "**oui**" if r["looped"] else "non"
        total = r["preprocess_s"] + r["ocr_s"]
        lines.append(
            f"| {r['page']} | {r['config']} | {boucle} "
            f"| {r['words']} | {r['preprocess_s']:.1f} | {r['ocr_s']:.1f} | {total:.1f} | {r.get('error', '')} |"
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
