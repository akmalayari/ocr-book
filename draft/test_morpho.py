"""
test_morpho.py — Test des opérations morphologiques après binarisation adaptative.

Phase 1 (défaut)   : sauvegarde les images à chaque étape du prétraitement.
Phase 2 (--ocr)    : lance l'OCR sur les configs spécifiées.
Phase 3 (--compare): compare les sorties OCR contre le baseline (même page),
                     et pour page_5 : contre le baseline de page_6.

Sorties : output/morpho_test/

Usage :
    python draft/test_morpho.py
    python draft/test_morpho.py --ocr
    python draft/test_morpho.py --ocr baseline opening_2 close2_open2
    python draft/test_morpho.py --compare
    python draft/test_morpho.py --pages page_5 page_6
"""

import argparse
import re
import sys
import difflib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import patch  # noqa: F401
from config import Config
from ocr_client import ocr_image

PHOTOS_DIR = Path(__file__).parent.parent / "photos"
OUT_DIR    = Path(__file__).parent.parent / "output" / "morpho_test"

DEFAULT_PAGES = ["page_1", "page_2", "page_5", "page_6"]

# op: "open" | "close" | "open_close" | "close_open"  —  k: kernel size (px)
MORPHO_CONFIGS: dict[str, tuple[str, int] | None] = {
    "baseline":     None,
    "opening_2":    ("open",       2),
    "opening_3":    ("open",       3),
    "closing_2":    ("close",      2),
    "closing_3":    ("close",      3),
    "open2_close2": ("open_close", 2),
    "close2_open2": ("close_open", 2),
}


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _binarize(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )


def _apply_morpho(bw: np.ndarray, op: str, k: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    if op == "open":
        return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    if op == "close":
        return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    if op == "open_close":
        return cv2.morphologyEx(
            cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel),
            cv2.MORPH_CLOSE, kernel,
        )
    if op == "close_open":
        return cv2.morphologyEx(
            cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel),
            cv2.MORPH_OPEN, kernel,
        )
    return bw


def _get_image(img_path: Path, config_name: str) -> np.ndarray:
    img  = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw   = _binarize(gray)
    cfg  = MORPHO_CONFIGS[config_name]
    if cfg is None:
        return bw
    return _apply_morpho(bw, *cfg)


# ── Phase 1 : visualisation ───────────────────────────────────────────────────

def phase1_visualize(images: list[Path]) -> None:
    print("── Génération des images step-by-step ─────────────────────")
    for img_path in images:
        img  = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0.0)
        bw = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 15,
        )

        steps: list[tuple[str, np.ndarray]] = [
            ("01_gray",       gray),
            ("02_blurred",    blurred),
            ("03_binarized",  bw),  # = baseline
        ]
        for config_name, cfg in MORPHO_CONFIGS.items():
            if cfg is None:
                continue
            steps.append((f"morpho_{config_name}", _apply_morpho(bw, *cfg)))

        for name, arr in steps:
            cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_{name}.jpg"), arr)

        print(f"  {img_path.name} → {len(steps)} fichiers")
    print(f"\nImages dans {OUT_DIR}")
    print("Pour lancer l'OCR : --ocr  (ou --ocr baseline opening_2)")


# ── Phase 2 : OCR ─────────────────────────────────────────────────────────────

def phase2_ocr(images: list[Path], configs_to_run: list[str]) -> None:
    from nexaai import VLM
    cfg_base = Config(prompt_mode="layout", preprocess_mode="none")
    vlm = VLM.from_(model=cfg_base.model, quant=cfg_base.quant, config=cfg_base.to_model_config())

    results = []
    print(f"\n── OCR sur {len(configs_to_run)} config(s) × {len(images)} image(s) ──")

    for img_path in images:
        for config_name in configs_to_run:
            if config_name not in MORPHO_CONFIGS:
                print(f"  [SKIP] config inconnue: {config_name}")
                continue

            # Sauvegarder l'image prétraitée
            preprocessed = _get_image(img_path, config_name)
            if config_name == "baseline":
                img_file = OUT_DIR / f"{img_path.stem}_03_binarized.jpg"
            else:
                img_file = OUT_DIR / f"{img_path.stem}_morpho_{config_name}.jpg"
            cv2.imwrite(str(img_file), preprocessed)

            out_md = OUT_DIR / f"{img_path.stem}_{config_name}.md"
            print(f"  [{img_path.stem} {config_name}] ...", end=" ", flush=True)
            row = {"page": img_path.stem, "config": config_name,
                   "looped": False, "words": 0, "latency": 0.0, "error": ""}
            try:
                text, metrics = ocr_image(img_file, vlm, cfg_base)
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
    print("\nPour comparer : --compare")


def _write_ocr_report(results: list[dict]) -> None:
    lines = [
        "# Rapport OCR — Opérations morphologiques\n",
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


# ── Phase 3 : comparaison ─────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    from postprocess import _clean_layout
    _HTML_TAG = re.compile(r'<[^>]+>')
    text = _clean_layout(text)
    text = _HTML_TAG.sub(' ', text)
    text = re.sub(r'(\w)- (\w)', r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _sim(path_a: Path, path_b: Path) -> float:
    from compare import tokenize_words
    ta = tokenize_words(path_a.read_text(encoding="utf-8"))
    tb = tokenize_words(path_b.read_text(encoding="utf-8"))
    return difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()


def phase3_compare(images: list[Path]) -> None:
    from compare import compare

    cmp_dir = OUT_DIR / "comparisons"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cmp_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    print("\n── Comparaisons vs baseline (même page) ──────────────────")

    for img_path in images:
        ref_md = OUT_DIR / f"{img_path.stem}_baseline.md"
        if not ref_md.exists():
            print(f"  [SKIP] {img_path.stem}_baseline.md manquant")
            continue

        ref_norm = tmp_dir / f"{img_path.stem}_baseline.md"
        ref_norm.write_text(_normalize(ref_md.read_text(encoding="utf-8")), encoding="utf-8")

        for config_name in MORPHO_CONFIGS:
            if config_name == "baseline":
                continue
            cand_md = OUT_DIR / f"{img_path.stem}_{config_name}.md"
            if not cand_md.exists():
                continue
            cand_norm = tmp_dir / f"{img_path.stem}_{config_name}.md"
            cand_norm.write_text(_normalize(cand_md.read_text(encoding="utf-8")), encoding="utf-8")
            out_path = cmp_dir / f"diff_{img_path.stem}_baseline_vs_{config_name}.md"
            print(f"  {img_path.stem} baseline vs {config_name} ...", end=" ", flush=True)
            compare(ref_norm, cand_norm, mode="sentence", out_path=out_path)
            sim = _sim(ref_norm, cand_norm)
            summary.append({"page": img_path.stem, "config": config_name,
                             "sim_vs_baseline": sim, "diff": out_path})

    summary_sorted = sorted(summary, key=lambda r: (r["page"], -r["sim_vs_baseline"]))
    lines = [
        "# Rapport global — Opérations morphologiques\n",
        "Similarité vs baseline (GaussianBlur + adaptive C=15) sur la même page.",
        "Une similarité < 100 % signale une différence OCR (pas nécessairement une amélioration).\n",
        "| Page | Config | Sim. vs baseline | Diff |",
        "|------|--------|-----------------|------|",
    ]
    for r in summary_sorted:
        diff_link = f"[diff]({r['diff'].name})"
        lines.append(f"| {r['page']} | {r['config']} | {r['sim_vs_baseline']:.1%} | {diff_link} |")

    # page_5 variantes vs page_6 baseline (ground truth)
    p5_base = OUT_DIR / "page_5_baseline.md"
    p6_base = OUT_DIR / "page_6_baseline.md"
    if p5_base.exists() and p6_base.exists():
        lines += [
            "",
            "## page_5 — variantes vs page_6 baseline (référence nette)\n",
            "| Config | Sim. vs page_6 |",
            "|--------|---------------|",
        ]
        ref6 = tmp_dir / "page_6_baseline.md"
        ref6.write_text(_normalize(p6_base.read_text(encoding="utf-8")), encoding="utf-8")
        for config_name in MORPHO_CONFIGS:
            cand_md = OUT_DIR / f"page_5_{config_name}.md"
            if not cand_md.exists():
                continue
            cand_n = tmp_dir / f"page_5_{config_name}_for_p6.md"
            cand_n.write_text(_normalize(cand_md.read_text(encoding="utf-8")), encoding="utf-8")
            sim = _sim(ref6, cand_n)
            lines.append(f"| {config_name} | {sim:.1%} |")

    out = cmp_dir / "global_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  → rapport global : {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", nargs="+", default=DEFAULT_PAGES,
                        help="Pages à tester (ex: page_5 page_6)")
    parser.add_argument("--ocr", nargs="*", metavar="CONFIG",
                        help="Configs OCR. Sans valeur = toutes. Ex: baseline opening_2")
    parser.add_argument("--compare", action="store_true",
                        help="Comparer les sorties OCR contre le baseline")
    args = parser.parse_args()

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

    if args.compare:
        phase3_compare(images)
        return

    phase1_visualize(images)

    if args.ocr is None:
        return

    configs_to_run = list(MORPHO_CONFIGS.keys()) if len(args.ocr) == 0 else args.ocr
    phase2_ocr(images, configs_to_run)


if __name__ == "__main__":
    main()
