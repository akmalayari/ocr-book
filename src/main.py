"""
main.py — Point d'entrée CLI du pipeline OCR livre

Usage :
    python src/main.py                          # config par défaut
    python src/main.py --images ./photos --out output/livre.md
    python src/main.py --no-resume              # recommencer depuis le début
    python src/main.py --no-layout              # fallback sans layout detection
    python src/main.py --no-postprocess         # sortie brute
    python src/main.py --rename                 # renommer les images puis OCR
    python src/main.py --rename --dry-run       # affiche les renommages sans les faire ni OCR
    python src/main.py --verbose                # logs détaillés
"""

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
        description="Pipeline OCR livre → Markdown via PaddleOCR-VL-1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Chemins
    p.add_argument("--images", default=_cfg.images_dir,
                   help="Dossier contenant les photos de pages")
    p.add_argument("--out", default=_cfg.output_file,
                   help="Fichier Markdown de sortie")

    # PaddleOCR
    p.add_argument("--no-layout", action="store_true",
                   help="Désactiver la détection de layout (fallback simple)")

    # Comportement
    p.add_argument("--no-resume", action="store_true",
                   help="Recommencer depuis le début (ignore le fichier existant)")
    p.add_argument("--no-postprocess", action="store_true",
                   help="Désactiver le post-traitement")
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

    cfg = Config(
        images_dir=args.images,
        output_file=args.out,
        use_layout_detection=not args.no_layout,
        resume=not args.no_resume,
        postprocess=not args.no_postprocess,
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
    logger.info("Pipeline OCR — PaddleOCR-VL-1.5")
    logger.info("  Images  : %s", cfg.images_path.resolve())
    logger.info("  Sortie  : %s", cfg.output_path.resolve())
    logger.info("  Layout  : %s", cfg.use_layout_detection)
    logger.info("  Reprise : %s", cfg.resume)
    logger.info("═" * 60)

    try:
        stats = run_pipeline(cfg)
    except Exception as e:
        logger.error("Erreur fatale : %s", e)
        return 1

    return 0 if stats.errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
