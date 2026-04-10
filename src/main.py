"""
main.py — Point d'entrée CLI du pipeline OCR livre

Usage :
    python src/main.py                                # config par défaut (mode base)
    python src/main.py --images ./photos --out output/livre.md
    python src/main.py --no-resume                    # recommencer depuis le début
    python src/main.py --no-layout                    # désactiver la détection de layout
    python src/main.py --no-postprocess               # sortie brute
    python src/main.py --mode obsidian                # OCR + postprocess Obsidian (prompt vault si non configuré)
    python src/main.py --mode obsidian --postprocess-only  # postprocess Obsidian sans relancer l'OCR
    python src/main.py --rename                       # renommer les images puis OCR
    python src/main.py --rename-only                  # renommer les images sans lancer l'OCR
    python src/main.py --rename-only 15               # renommer en partant de page_015
    python src/main.py --rename --dry-run             # affiche les renommages sans les faire ni OCR
    python src/main.py --rename-only --chapters "Leçon 1" "Leçon 3"  # sous-dossiers choisis
    python src/main.py --rename-only --dir-level      # ordre : dossiers alpha > sous-dossiers alpha > images par date
    python src/main.py --verbose                      # logs détaillés
"""

import argparse
import logging
import sys

from config import Config
from images import rename_images, copy_from_subdirs, has_image_subdirs
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
                   help="Désactiver la détection de layout")

    # Comportement
    p.add_argument("--no-resume", action="store_true",
                   help="Recommencer depuis le début (ignore le fichier existant)")
    p.add_argument("--no-postprocess", action="store_true",
                   help="Désactiver le post-traitement")
    p.add_argument("--mode", choices=["base", "obsidian"], default=_cfg.mode,
                   help="Mode de sortie : base (HTML img) ou obsidian (wikilinks ![[]])")
    p.add_argument("--postprocess-only", action="store_true",
                   help="Avec --mode obsidian : applique le postprocess sur le .md existant sans relancer l'OCR")
    p.add_argument("--verbose", action="store_true",
                   help="Logs détaillés (DEBUG)")

    # Utilitaires
    p.add_argument("--rename", action="store_true",
                   help="Renommer les images en page_001.jpg, page_002.jpg… puis lancer l'OCR")
    p.add_argument("--rename-prefix", default=_cfg.rename_prefix,
                   help="Préfixe pour --rename (défaut: page)")
    p.add_argument("--rename-only", nargs="?", type=int, const=1, default=None, metavar="START",
                   help="Renommer les images sans lancer l'OCR (optionnel: numéro de départ, défaut: 1)")
    p.add_argument("--chapters", nargs="+", metavar="NOM",
                   help="Sous-dossiers à traiter (dans l'ordre fourni). Implique la copie vers le dossier parent.")
    p.add_argument("--dir-level", action="store_true",
                   help="Avec --rename/--rename-only : ordre par dossier (dossiers alpha, sous-dossiers alpha, images par date)")
    p.add_argument("--dry-run", action="store_true",
                   help="Avec --rename/--rename-only : affiche les renommages sans les faire ni lancer l'OCR")

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
        mode=args.mode,
        verbose=args.verbose,
    )

    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    # ── Mode obsidian --postprocess-only ─────────────────────────────────────
    if args.postprocess_only:
        if cfg.mode != "obsidian":
            logger.error("--postprocess-only requiert --mode obsidian")
            return 1
        from obsidian import prompt_if_needed, postprocess_file
        prompt_if_needed(cfg)
        postprocess_file(cfg)
        logger.info("Postprocess obsidian appliqué : %s", cfg.output_path.resolve())
        return 0

    # ── Renommage / copie depuis sous-dossiers (optionnel, avant OCR) ────────
    if args.rename or args.rename_only is not None:
        start = args.rename_only if args.rename_only is not None else 1
        images_path = cfg.images_path
        if args.chapters or (images_path.is_dir() and has_image_subdirs(images_path, cfg.extensions)):
            logger.info("Mode sous-dossiers : copie vers %s", cfg.images_dir)
            copy_from_subdirs(
                images_path, cfg.extensions,
                chapters=args.chapters,
                prefix=args.rename_prefix,
                start=start,
                dry_run=args.dry_run,
                dir_level=args.dir_level,
            )
        else:
            logger.info("Renommage des images dans : %s", cfg.images_dir)
            rename_images(cfg.images_dir, cfg.extensions, prefix=args.rename_prefix, dry_run=args.dry_run, start=start)
        if args.dry_run or args.rename_only is not None:
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
        from pipeline import run_pipeline
        stats = run_pipeline(cfg)
    except Exception as e:
        logger.error("Erreur fatale : %s", e)
        return 1

    return 0 if stats.errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
