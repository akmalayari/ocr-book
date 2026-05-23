"""
main.py — CLI entry point for the book OCR pipeline

Supports images, PDFs and EPUBs (auto-detected from --images).

Usage :
    python main.py                                # default config (base mode)
    python main.py --images ./photos --out output/book.md
    python main.py --images book.epub --out output/book.md     # EPUB auto-detected
    python main.py --images book.pdf --out output/book.md      # PDF auto-detected
    python main.py --no-resume                    # restart from the beginning
    python main.py --no-layout                    # disable layout detection
    python main.py --no-postprocess               # raw output
    python main.py --mode obsidian                # OCR + Obsidian postprocess (prompts vault if not configured)
    python main.py --mode obsidian --postprocess-only  # Obsidian postprocess without re-running OCR
    python main.py --rename                       # rename images then OCR
    python main.py --rename-only                  # rename images without running OCR
    python main.py --rename-only 15               # rename starting at page_015
    python main.py --rename --dry-run             # print renames without doing them or OCR
    python main.py --rename-only --chapters "Lesson 1" "Lesson 3"  # selected subfolders
    python main.py --rename-only --dir-level      # order: folders alpha > subfolders alpha > images by date
    python main.py --verbose                      # detailed logs
    python main.py --header-pattern "^[IVX]+\\." 2 --header-pattern "^[A-Z]\\." 3  # enable header detection
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow absolute imports from src/ when running main.py from the project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config
from src.images import rename_images, copy_from_subdirs, has_image_subdirs
from src.progress import setup_logging


def build_parser() -> argparse.ArgumentParser:
    _cfg = Config()
    p = argparse.ArgumentParser(
        description="Book OCR / EPUB extraction pipeline -> Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Paths
    p.add_argument("--images", default=_cfg.images_dir,
                   help="Folder containing page photos, or path to a single file (PDF, EPUB, image)")
    p.add_argument("--out", default=_cfg.output_file,
                   help="Output Markdown file")
    p.add_argument("--llama-server", default=_cfg.llama_server_path,
                   help="Path to llama-server executable (env: LLAMA_SERVER_PATH)")
    p.add_argument("--model", default=_cfg.model_path,
                   help="Path to model .gguf file (env: MODEL_PATH)")
    p.add_argument("--mmproj", default=_cfg.mmproj_path,
                   help="Path to mmproj .gguf file (env: MMPROJ_PATH)")
    p.add_argument("--max-tokens", type=int, default=_cfg.max_tokens,
                   help="Max tokens generated per page (default: 4096)")
    p.add_argument("--n-ctx", type=int, default=None,
                   help="KV cache size (default: n_parallel * 2048; increase for large tables)")
    p.add_argument("--n-parallel", type=int, default=_cfg.n_parallel,
                   help="Intra-page parallel slots (default: 1; set >1 only with the parallel patch)")
    p.add_argument("--n-threads", type=int, default=_cfg.n_threads,
                   help="CPU threads for llama-server (default: 4)")
    p.add_argument("--n-servers", type=int, default=_cfg.n_servers,
                   help="Number of parallel llama-server instances (default: 1)")
    p.add_argument("--no-kv-offload", action="store_true",
                   help="Disable KV cache GPU offload")

    # PaddleOCR
    p.add_argument("--no-layout", action="store_true",
                   help="Disable layout detection")
    p.add_argument("--method", choices=["text", "docling", "paddleocrvl"], default=_cfg.extraction_method,
                   help="Extraction method for PDFs: text (fast/basic), docling (structured), paddleocrvl (default, best quality)")

    # Behavior
    p.add_argument("--no-resume", action="store_true",
                   help="Restart from the beginning (ignore existing file)")
    p.add_argument("--no-postprocess", action="store_true",
                   help="Disable post-processing")
    p.add_argument("--keep-html", action="store_true",
                   help="Keep HTML tables, figures and formatting (default: convert to pure Markdown)")
    p.add_argument("--mode", choices=["base", "obsidian"], default=_cfg.mode,
                   help="Output mode: base (HTML img) or obsidian (wikilinks ![[]])")
    p.add_argument("--postprocess-only", action="store_true",
                   help="With --mode obsidian: apply postprocess on existing .md without re-running OCR")
    p.add_argument("--verbose", action="store_true",
                   help="Detailed logs (DEBUG)")

    # Utilities
    p.add_argument("--rename", action="store_true",
                   help="Rename images to page_001.jpg, page_002.jpg… then run OCR")
    p.add_argument("--rename-prefix", default=_cfg.rename_prefix,
                   help="Prefix for --rename (default: page)")
    p.add_argument("--rename-only", nargs="?", type=int, const=1, default=None, metavar="START",
                   help="Rename images without running OCR (optional: starting number, default: 1)")
    p.add_argument("--chapters", nargs="+", metavar="NAME",
                   help="Subfolders to process (in the given order). Implies copying to the parent folder.")
    p.add_argument("--dir-level", action="store_true",
                   help="With --rename/--rename-only: order by folder (folders alpha, subfolders alpha, images by date)")
    p.add_argument("--dry-run", action="store_true",
                   help="With --rename/--rename-only: print renames without doing them or running OCR")
    p.add_argument("--migrate", action="store_true",
                   help="Copy figures to the Obsidian vault (vault_path/vault_figures_dir) without running OCR")

    # Header detection
    p.add_argument("--header-pattern", nargs=2, action="append", metavar=("REGEX", "LEVEL"),
                   help="Promote matching lines to Markdown headers. Regex and heading level (e.g. --header-pattern '^[IVX]+\\.' 2). Repeatable.")

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()

    cfg = Config(
        images_dir=args.images,
        output_file=args.out,
        llama_server_path=args.llama_server,
        model_path=args.model,
        mmproj_path=args.mmproj,
        max_tokens=args.max_tokens,
        n_ctx=args.n_ctx,
        n_parallel=args.n_parallel,
        n_threads=args.n_threads,
        n_servers=args.n_servers,
        kv_offload=not args.no_kv_offload,
        rename_prefix=args.rename_prefix,
        use_layout_detection=not args.no_layout,
        resume=not args.no_resume,
        postprocess=not args.no_postprocess,
        keep_html=args.keep_html,
        mode=args.mode,
        verbose=args.verbose,
        extraction_method=args.method,
    )

    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    if args.header_pattern:
        cfg.header_patterns = [(regex, int(level)) for regex, level in args.header_pattern]

    # ── Obsidian setup (common to all obsidian modes) ────────────────────────
    if cfg.mode == "obsidian" or args.migrate:
        from pathlib import Path as _Path
        from src.obsidian import prompt_if_needed
        prompt_if_needed(cfg)
        if not cfg.vault_root or not cfg.vault_figures_dir or not cfg.vault_path:
            logger.error("vault_root, vault_path and vault_figures_dir must be configured.")
            return 1
        if not args.migrate:
            cfg.output_file = str(_Path(cfg.vault_root) / cfg.vault_path / _Path(cfg.output_file).name)

    # ── Figure migration to vault ────────────────────────────────────────────
    if args.migrate:
        from src.obsidian import migrate_figures
        migrate_figures(cfg, dry_run=args.dry_run)
        return 0

    # ── Obsidian --postprocess-only mode ─────────────────────────────────────
    if args.postprocess_only:
        if cfg.mode != "obsidian":
            logger.error("--postprocess-only requires --mode obsidian")
            return 1
        from src.obsidian import postprocess_file, migrate_figures
        postprocess_file(cfg)
        migrate_figures(cfg)
        logger.info("Obsidian postprocess applied: %s", cfg.output_path.resolve())
        return 0

    # ── Rename / copy from subfolders (optional, before OCR) ─────────────────
    if args.rename or args.rename_only is not None:
        start = args.rename_only if args.rename_only is not None else 1
        images_path = cfg.images_path
        if args.chapters or (images_path.is_dir() and has_image_subdirs(images_path, cfg.extensions)):
            logger.info("Subfolder mode: copying to %s", cfg.images_dir)
            copy_from_subdirs(
                images_path, cfg.extensions,
                chapters=args.chapters,
                prefix=args.rename_prefix,
                start=start,
                dry_run=args.dry_run,
                dir_level=args.dir_level,
            )
        else:
            logger.info("Renaming images in: %s", cfg.images_dir)
            rename_images(cfg.images_dir, cfg.extensions, prefix=args.rename_prefix, dry_run=args.dry_run, start=start)
        if args.dry_run or args.rename_only is not None:
            return 0

    # ── Detect input format and reject mixed folders ─────────────────────────
    from src.images import _collect_sources, ImageCollectionError
    try:
        sources = _collect_sources(cfg)
    except ImageCollectionError as e:
        logger.error(str(e))
        return 1

    # Categorize sources
    images = [s for s in sources if s.suffix.lower() in cfg.extensions]
    pdfs   = [s for s in sources if s.suffix.lower() == ".pdf"]
    epubs  = [s for s in sources if s.suffix.lower() == ".epub"]
    active_types = {t for t, lst in [("image", images), ("pdf", pdfs), ("epub", epubs)] if lst}

    # Single file shortcuts (bypass folder logic)
    if len(sources) == 1:
        src = sources[0]
        if src.suffix.lower() == ".epub":
            from src.epub import extract_epub
            logger.info("═" * 60)
            logger.info("EPUB Extraction — %s", src)
            logger.info("  Output  : %s", cfg.output_path.resolve())
            logger.info("  Figures : %s", cfg.figures_path.resolve())
            logger.info("═" * 60)
            try:
                extract_epub(epub_path=src, output_md=cfg.output_path, figures_dir=cfg.figures_path)
            except Exception as e:
                logger.error("Fatal error during EPUB extraction: %s", e)
                import traceback
                traceback.print_exc()
                return 1
            if cfg.mode == "obsidian":
                from src.obsidian import fix_markdown_image_paths_obsidian, migrate_figures
                text = cfg.output_path.read_text(encoding="utf-8")
                text = fix_markdown_image_paths_obsidian(text, cfg.vault_figures_dir)
                cfg.output_path.write_text(text, encoding="utf-8", newline="\n")
                logger.info("Obsidian image links applied.")
                migrate_figures(cfg, flat=True)
            logger.info("Output file: %s", cfg.output_path.resolve())
            return 0
        # PDFs and images fall through to the pipeline below

    # Reject multiple PDFs
    if len(pdfs) > 1:
        logger.error("Multiple PDFs detected in %s:", cfg.images_dir)
        logger.error("  %s", ", ".join(f.name for f in pdfs[:5]))
        logger.error("Please place only one PDF per folder.")
        return 1

    # Reject mixed-type folders
    if len(active_types) > 1:
        logger.error("Mixed file types detected in %s:", cfg.images_dir)
        for t, lst in [("image", images), ("pdf", pdfs), ("epub", epubs)]:
            if lst:
                logger.error("  %s: %s", t.upper(), ", ".join(f.name for f in lst[:5]))
        logger.error("Please use a folder containing only one type of file (images, a single PDF, or a single EPUB).")
        return 1

    # Reject multiple EPUBs
    if len(epubs) > 1:
        logger.error("Multiple EPUBs detected in %s:", cfg.images_dir)
        logger.error("  %s", ", ".join(f.name for f in epubs[:5]))
        logger.error("Please place only one EPUB per folder.")
        return 1

    # Single EPUB in folder (not passed as direct file)
    if epubs:
        epub_path = epubs[0]
        from src.epub import extract_epub
        logger.info("═" * 60)
        logger.info("EPUB Extraction — %s", epub_path)
        logger.info("  Output  : %s", cfg.output_path.resolve())
        logger.info("  Figures : %s", cfg.figures_path.resolve())
        logger.info("═" * 60)
        try:
            extract_epub(epub_path=epub_path, output_md=cfg.output_path, figures_dir=cfg.figures_path)
        except Exception as e:
            logger.error("Fatal error during EPUB extraction: %s", e)
            import traceback
            traceback.print_exc()
            return 1
        if cfg.mode == "obsidian":
            from src.obsidian import fix_markdown_image_paths_obsidian, migrate_figures
            text = cfg.output_path.read_text(encoding="utf-8")
            text = fix_markdown_image_paths_obsidian(text, cfg.vault_figures_dir)
            cfg.output_path.write_text(text, encoding="utf-8", newline="\n")
            logger.info("Obsidian image links applied.")
            migrate_figures(cfg, flat=True)
        logger.info("Output file: %s", cfg.output_path.resolve())
        return 0

    try:
        # ── OCR Pipeline ─────────────────────────────────────────────────────────
        logger.info("═" * 60)
        logger.info("OCR Pipeline — method=%s", cfg.extraction_method)
        logger.info("  Images  : %s", cfg.images_path.resolve())
        logger.info("  Output  : %s", cfg.output_path.resolve())
        logger.info("  Layout  : %s", cfg.use_layout_detection)
        logger.info("  Resume  : %s", cfg.resume)
        logger.info("═" * 60)

        try:
            from src.pipeline import run_pipeline
            stats = run_pipeline(cfg)
        except Exception as e:
            logger.error("Fatal error: %s", e)
            import traceback
            traceback.print_exc()
            return 1

        return 0 if stats.errors == 0 else 2
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
