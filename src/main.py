"""
main.py — Point d'entrée CLI du pipeline OCR livre

Usage :
    python main.py                          # config par défaut
    python main.py --images ./photos --out output/livre.md
    python main.py --mode plain             # OCR texte brut
    python main.py --no-resume              # recommencer depuis le début
    python main.py --rename-only            # renommer les images sans OCR
    python main.py --verbose                # logs détaillés
"""

import argparse
import sys
import logging

from config import Config
from pipeline import run_pipeline
from images import rename_images
from progress import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline OCR livre → Markdown via DeepSeek-OCR (Nexa)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Chemins
    p.add_argument("--images", default="./photos",
                   help="Dossier contenant les photos de pages (défaut: ./photos)")
    p.add_argument("--out", default="output/livre.md",
                   help="Fichier Markdown de sortie (défaut: output/livre.md)")

    # Modèle / serveur
    p.add_argument("--model", default="NexaAI/DeepSeek-OCR-GGUF",
                   help="Modèle Nexa à utiliser")

    # OCR
    p.add_argument("--mode", choices=["markdown", "plain", "figure"],
                   default="markdown",
                   help="Mode OCR (défaut: markdown)")
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="Tokens max par page (défaut: 4096)")
    p.add_argument("--timeout", type=int, default=180,
                   help="Timeout par image en secondes (défaut: 180)")

    # Comportement
    p.add_argument("--no-resume", action="store_true",
                   help="Recommencer depuis le début (ignore le fichier existant)")
    p.add_argument("--verbose", action="store_true",
                   help="Logs détaillés (DEBUG)")

    # Utilitaires
    p.add_argument("--rename-only", action="store_true",
                   help="Renommer les images en page_001.jpg, page_002.jpg… sans OCR")
    p.add_argument("--rename-prefix", default="page",
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
        prompt_mode=args.mode,
        max_tokens=args.max_tokens,
        request_timeout_s=args.timeout,
        resume=not args.no_resume,
        verbose=args.verbose,
    )

    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    # ── Mode renommage uniquement ─────────────────────────────────────────────
    if args.rename_only:
        logger.info("Renommage des images dans : %s", cfg.images_dir)
        rename_images(cfg.images_dir, prefix=args.rename_prefix, dry_run=args.dry_run)
        return 0

    # ── Pipeline OCR ─────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("Pipeline OCR — %s", cfg.model)
    logger.info("  Images  : %s", cfg.images_path.resolve())
    logger.info("  Sortie  : %s", cfg.output_path.resolve())
    logger.info("  Mode    : %s", cfg.prompt_mode)
    logger.info("  Reprise : %s", cfg.resume)
    logger.info("═" * 60)

    try:
        stats = run_pipeline(cfg)
    except Exception as e:
        logger.error("Erreur fatale : %s", e)
        return 1

    return 0 if stats.errors == 0 else 2  # code 2 = terminé avec des erreurs


if __name__ == "__main__":
    sys.exit(main())
