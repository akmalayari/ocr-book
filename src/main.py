"""
main.py — Point d'entrée CLI du pipeline OCR livre

Usage :
    python src/main.py                          # config par défaut
    python src/main.py --images ./photos --out output/livre.md
    python src/main.py --mode plain             # OCR texte brut
    python src/main.py --mode rec:titre         # localiser un élément dans l'image
    python src/main.py --quant q8_0             # quantization (q8_0, bf16; défaut: bf16)
    python src/main.py --max-tokens 2048        # tokens max par page
    python src/main.py --no-resume              # recommencer depuis le début
    python src/main.py --preprocess none        # sans pré-traitement (défaut: binarize)
    python src/main.py --rename                 # renommer les images puis OCR
    python src/main.py --rename --dry-run       # affiche les renommages sans les faire ni OCR
    python src/main.py --verbose                # logs détaillés
"""

import patch  # noqa: F401 — doit être importé avant nexaai

import argparse
import logging
import sys

from config import Config
from images import rename_images
from pipeline import run_pipeline
from progress import setup_logging


def build_parser() -> argparse.ArgumentParser:
    _cfg = Config()
    p = argparse.ArgumentParser(
        description="Pipeline OCR livre → Markdown via DeepSeek-OCR (Nexa)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Chemins
    p.add_argument("--images", default=_cfg.images_dir,
                   help="Dossier contenant les photos de pages")
    p.add_argument("--out", default=_cfg.output_file,
                   help="Fichier Markdown de sortie")

    # Modèle
    p.add_argument("--model", default=_cfg.model,
                   help="Modèle Nexa à utiliser")
    p.add_argument("--quant", choices=list(_cfg.QUANTS), default=_cfg.quant,
                   help="Quantization du modèle")

    # OCR
    _modes = [m if m != "rec" else "rec:<cible>" for m in _cfg.PROMPTS.keys()]
    p.add_argument("--mode", default=_cfg.prompt_mode, metavar="MODE",
                   help=f"Mode OCR : {', '.join(_modes)}")
    p.add_argument("--max-tokens", type=int, default=_cfg.max_tokens,
                   help="Tokens max par page")
    p.add_argument("--preprocess", choices=["none", "binarize", "sauvola"],
                   default=_cfg.preprocess_mode,
                   help="Pré-traitement image")

    # Comportement
    p.add_argument("--no-resume", action="store_true",
                   help="Recommencer depuis le début (ignore le fichier existant)")
    p.add_argument("--one-pass", action="store_true",
                   help="Désactiver la passe 2 figures (défaut: two-pass activé en mode layout)")
    p.add_argument("--verbose", action="store_true",
                   help="Logs détaillés (DEBUG)")

    # Utilitaires
    p.add_argument("--rename", action="store_true",
                   help="Renommer les images en page_001.jpg, page_002.jpg… puis lancer l'OCR")
    p.add_argument("--rename-prefix", default=_cfg.rename_prefix,
                   help="Préfixe pour --rename (défaut: page)")
    p.add_argument("--dry-run", action="store_true",
                   help="Avec --rename : affiche les renommages sans les faire ni lancer l'OCR")

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()

    # ── Parsing --mode (supporte rec:<cible>) ────────────────────────────────
    mode_str = args.mode
    if mode_str.startswith("rec"):
        if ":" not in mode_str or not mode_str[4:].strip():
            p.error("--mode rec requiert une cible : --mode rec:<élément à localiser>")
        prompt_mode, locate_target = "rec", mode_str[4:]
    elif mode_str not in Config.PROMPTS:
        valid = [m if m != "rec" else "rec:<cible>" for m in Config.PROMPTS]
        p.error(f"--mode invalide : {mode_str!r}. Valeurs possibles : {', '.join(valid)}")
    else:
        prompt_mode, locate_target = mode_str, ""

    cfg = Config(
        images_dir=args.images,
        output_file=args.out,
        model=args.model,
        quant=args.quant,
        prompt_mode=prompt_mode,
        locate_target=locate_target,
        max_tokens=args.max_tokens,
        preprocess_mode=args.preprocess,
        resume=not args.no_resume,
        two_pass=not args.one_pass,
        verbose=args.verbose,
    )

    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    # ── Renommage (optionnel, avant OCR) ─────────────────────────────────────
    if args.rename:
        logger.info("Renommage des images dans : %s", cfg.images_dir)
        rename_images(cfg.images_dir, cfg.extensions, prefix=args.rename_prefix, dry_run=args.dry_run)
        if args.dry_run:
            return 0

    # ── Pipeline OCR ─────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("Pipeline OCR — %s", cfg.model)
    logger.info("  Images      : %s", cfg.images_path.resolve())
    logger.info("  Sortie      : %s", cfg.output_path.resolve())
    logger.info("  Mode        : %s", cfg.prompt_mode)
    logger.info("  Prétraitement : %s", cfg.preprocess_mode)
    logger.info("  Reprise     : %s", cfg.resume)
    logger.info("═" * 60)

    try:
        stats = run_pipeline(cfg)
    except Exception as e:
        logger.error("Erreur fatale : %s", e)
        return 1

    return 0 if stats.errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
