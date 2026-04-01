"""
main.py — Point d'entrée CLI du pipeline OCR livre

Usage :
    python main.py                          # config par défaut
    python main.py --images ./photos --out output/livre.md
    python main.py --mode plain             # OCR texte brut
    python main.py --no-resume              # recommencer depuis le début
    python main.py --preprocess none        # sans pré-traitement
    python main.py --rename-only            # renommer les images sans OCR
    python main.py --verbose                # logs détaillés
"""

import patch  # noqa: F401 — doit être importé avant nexaai

import argparse
import logging
import sys

from config import Config
from images import rename_images
from pipeline import run_pipeline
from progress import setup_logging


def parse_args() -> argparse.Namespace:
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
    p.add_argument("--mode", choices=list(_cfg.PROMPTS.keys()),
                   default=_cfg.prompt_mode,
                   help="Mode OCR")
    p.add_argument("--max-tokens", type=int, default=_cfg.max_tokens,
                   help="Tokens max par page")
    p.add_argument("--preprocess", choices=["none", "binarize"],
                   default=_cfg.preprocess_mode,
                   help="Pré-traitement image")

    # Comportement
    p.add_argument("--no-resume", action="store_true",
                   help="Recommencer depuis le début (ignore le fichier existant)")
    p.add_argument("--verbose", action="store_true",
                   help="Logs détaillés (DEBUG)")

    # Utilitaires
    p.add_argument("--rename-only", action="store_true",
                   help="Renommer les images en page_001.jpg, page_002.jpg… sans OCR")
    p.add_argument("--rename-prefix", default=_cfg.rename_prefix,
                   help="Préfixe pour --rename-only (défaut: page)")
    p.add_argument("--dry-run", action="store_true",
                   help="Avec --rename-only : affiche les renommages sans les faire")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    cfg = Config(
        images_dir=args.images,
        output_file=args.out,
        model=args.model,
        quant=args.quant,
        prompt_mode=args.mode,
        max_tokens=args.max_tokens,
        preprocess_mode=args.preprocess,
        resume=not args.no_resume,
        verbose=args.verbose,
    )

    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    # ── Mode renommage uniquement ─────────────────────────────────────────────
    if args.rename_only:
        logger.info("Renommage des images dans : %s", cfg.images_dir)
        rename_images(cfg.images_dir, cfg.extensions, prefix=args.rename_prefix, dry_run=args.dry_run)
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
