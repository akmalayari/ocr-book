"""
test_sauvola_patch.py — Patches pour corriger la perte de texte dans la pliure avec Sauvola.

Problème : Sauvola rend bien le texte général mais efface le texte dans la zone de pliure
(ombre uniforme → faible variance locale → seuil trop bas → texte disparaît).

Deux patches testés :
  - and   : bitwise_and(sauvola, baseline) → texte retenu si l'un OU l'autre le détecte.
  - bgdiv : bg_divide avant Sauvola → normalise le gradient d'éclairage (incl. ombre de pliure).

Configs de base Sauvola : toutes les combinaisons (window_size × k) sans pré-blur.
Plus baseline seul pour référence.

Phase 1 (défaut)    : génère les images prétraitées pour inspection visuelle.
Phase 2 (--ocr)     : lance l'OCR sur les configs choisies.
Phase 3 (--compare) : compare contre baseline et contre page_6 (ground truth pour page_5).

Sorties : output/sauvola_patch/

Usage :
    python draft/test_sauvola_patch.py
    python draft/test_sauvola_patch.py --ocr
    python draft/test_sauvola_patch.py --ocr baseline sauvola_25_02 and_25_02 bgdiv_25_02
    python draft/test_sauvola_patch.py --compare
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

try:
    from skimage.filters import threshold_sauvola
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

PHOTOS_DIR = Path(__file__).parent.parent / "photos"
OUT_DIR    = Path(__file__).parent.parent / "output" / "sauvola_patch"

DEFAULT_PAGES = ["page_1", "page_2", "page_5", "page_6"]

# Paramètres Sauvola à tester (window_size, k)
SAUVOLA_PARAMS: list[tuple[int, float]] = [
    (25, 0.2),
    (51, 0.2),
    (25, 0.3),
    (51, 0.3),
]

# Configs = baseline + sauvola brut + patch AND + patch bg_divide pour chaque jeu de params
# Le dict associe config_name → description lisible
def _build_configs() -> dict[str, str]:
    configs: dict[str, str] = {"baseline": "GaussianBlur+adaptive C=15"}
    for w, k in SAUVOLA_PARAMS:
        tag = f"{w}_{str(k).replace('.', '')}"
        configs[f"sauvola_{tag}"]  = f"Sauvola w={w} k={k}"
        configs[f"and_{tag}"]      = f"AND(Sauvola w={w} k={k}, baseline)"
        configs[f"bgdiv_{tag}"]    = f"bg_divide + Sauvola w={w} k={k}"
    return configs

CONFIGS = _build_configs()


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _baseline(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )


def _sauvola(gray: np.ndarray, w: int, k: float) -> np.ndarray:
    thresh = threshold_sauvola(gray, window_size=w, k=k)
    return ((gray > thresh).astype(np.uint8)) * 255


def _bg_divide(gray: np.ndarray, blur_ksize: int = 101) -> np.ndarray:
    """Normalise l'illumination en divisant par le fond estimé (grande gaussienne)."""
    bg = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0).astype(np.float32)
    bg = np.where(bg < 1, 1, bg)
    norm = (gray.astype(np.float32) / bg * 128).clip(0, 255).astype(np.uint8)
    return norm


def _get_image(img_path: Path, config_name: str) -> np.ndarray:
    img  = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if config_name == "baseline":
        return _baseline(gray)

    # Extraire le tag de params (ex: "25_02" de "sauvola_25_02")
    for w, k in SAUVOLA_PARAMS:
        tag = f"{w}_{str(k).replace('.', '')}"
        if config_name == f"sauvola_{tag}":
            return _sauvola(gray, w, k)
        if config_name == f"and_{tag}":
            s   = _sauvola(gray, w, k)
            b   = _baseline(gray)
            # Texte = 0 (noir). AND conserve les 0 des deux → union des pixels texte.
            return cv2.bitwise_and(s, b)
        if config_name == f"bgdiv_{tag}":
            return _sauvola(_bg_divide(gray), w, k)

    raise ValueError(f"Config inconnue : {config_name}")


# ── Phase 1 : visualisation ───────────────────────────────────────────────────

def phase1_visualize(images: list[Path]) -> None:
    print("── Génération des images step-by-step ─────────────────────")
    for img_path in images:
        img     = cv2.imread(str(img_path))
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0.0)
        bg_norm = _bg_divide(gray)

        # Étapes intermédiaires
        for name, arr in [
            ("step_01_gray",      gray),
            ("step_02_blurred",   blurred),
            ("step_03_bg_divide", bg_norm),
            ("step_04_baseline",  _baseline(gray)),
        ]:
            cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_{name}.jpg"), arr)

        # Résultats pour chaque config
        n = 4
        for config_name in CONFIGS:
            result = _get_image(img_path, config_name)
            cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_{config_name}.jpg"), result)
            n += 1

        print(f"  {img_path.name} → {n} fichiers")
    print(f"\nImages dans {OUT_DIR}")
    print("Pour lancer l'OCR : --ocr  (ou --ocr baseline sauvola_25_02 and_25_02 bgdiv_25_02)")


# ── Phase 2 : OCR ─────────────────────────────────────────────────────────────

def phase2_ocr(images: list[Path], configs_to_run: list[str]) -> None:
    from nexaai import VLM
    cfg_base = Config(prompt_mode="layout", preprocess_mode="none")
    vlm = VLM.from_(model=cfg_base.model, quant=cfg_base.quant, config=cfg_base.to_model_config())

    results = []
    print(f"\n── OCR sur {len(configs_to_run)} config(s) × {len(images)} image(s) ──")

    for img_path in images:
        for config_name in configs_to_run:
            if config_name not in CONFIGS:
                print(f"  [SKIP] config inconnue: {config_name}")
                print(f"  Configs disponibles : {', '.join(CONFIGS)}")
                continue

            preprocessed = _get_image(img_path, config_name)
            img_file = OUT_DIR / f"{img_path.stem}_{config_name}.jpg"
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
        "# Rapport OCR — Sauvola patches\n",
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

    # Normaliser tous les md disponibles
    normed: dict[str, Path] = {}
    for md in OUT_DIR.glob("*.md"):
        if md.stem in ("ocr_report",):
            continue
        n = tmp_dir / md.name
        n.write_text(_normalize(md.read_text(encoding="utf-8")), encoding="utf-8")
        normed[md.stem] = n

    summary: list[dict] = []
    print("\n── Comparaisons vs baseline (même page) ──────────────────")

    for img_path in images:
        ref_key = f"{img_path.stem}_baseline"
        if ref_key not in normed:
            print(f"  [SKIP] {ref_key}.md manquant")
            continue

        for config_name in CONFIGS:
            if config_name == "baseline":
                continue
            cand_key = f"{img_path.stem}_{config_name}"
            if cand_key not in normed:
                continue
            out_path = cmp_dir / f"diff_{img_path.stem}_baseline_vs_{config_name}.md"
            print(f"  {img_path.stem} baseline vs {config_name} ...", end=" ", flush=True)
            compare(normed[ref_key], normed[cand_key], mode="sentence", out_path=out_path)
            sim = _sim(normed[ref_key], normed[cand_key])
            summary.append({"page": img_path.stem, "config": config_name,
                             "sim_vs_baseline": sim, "diff": out_path})

    summary_sorted = sorted(summary, key=lambda r: (r["page"], -r["sim_vs_baseline"]))
    lines = [
        "# Rapport global — Sauvola patches\n",
        "Similarité vs baseline (GaussianBlur + adaptive C=15) sur la même page.",
        "Une similarité < 100 % signale une différence ; à croiser avec le nombre de mots.\n",
        "| Page | Config | Sim. vs baseline | Diff |",
        "|------|--------|-----------------|------|",
    ]
    for r in summary_sorted:
        diff_link = f"[diff]({r['diff'].name})"
        lines.append(f"| {r['page']} | {r['config']} | {r['sim_vs_baseline']:.1%} | {diff_link} |")

    # page_5 variantes vs page_6 baseline (ground truth)
    ref6_key = "page_6_baseline"
    p5_base_key = "page_5_baseline"
    if ref6_key in normed and p5_base_key in normed:
        lines += [
            "",
            "## page_5 — variantes vs page_6 baseline (référence nette)\n",
            "| Config | Sim. vs page_6 |",
            "|--------|---------------|",
        ]
        for config_name in CONFIGS:
            cand_key = f"page_5_{config_name}"
            if cand_key not in normed:
                continue
            sim = _sim(normed[ref6_key], normed[cand_key])
            lines.append(f"| {config_name} | {sim:.1%} |")

    out = cmp_dir / "global_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  → rapport global : {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not SKIMAGE_OK:
        print("[ERREUR] scikit-image non installé. Lancer : pip install scikit-image")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", nargs="+", default=DEFAULT_PAGES)
    parser.add_argument("--ocr", nargs="*", metavar="CONFIG",
                        help="Sans valeur = toutes les configs.")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--list", action="store_true",
                        help="Afficher les configs disponibles et quitter")
    args = parser.parse_args()

    if args.list:
        for name, desc in CONFIGS.items():
            print(f"  {name:30s}  {desc}")
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

    if args.compare:
        phase3_compare(images)
        return

    phase1_visualize(images)

    if args.ocr is None:
        return

    configs_to_run = list(CONFIGS.keys()) if len(args.ocr) == 0 else args.ocr
    phase2_ocr(images, configs_to_run)


if __name__ == "__main__":
    main()
