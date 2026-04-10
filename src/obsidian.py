"""
obsidian.py — Utilitaires pour l'export Obsidian
"""

import logging
import re
import shutil
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)


def prompt_if_needed(cfg: Config) -> None:
    """Prompt interactif pour vault_root, vault_path et vault_figures_dir si non configurés."""
    if cfg.vault_root is None:
        cfg.vault_root = input("Chemin absolu de la racine du vault Obsidian : ").strip().strip('"').strip("'")
    if cfg.vault_path is None:
        cfg.vault_path = input("Sous-dossier de sortie relatif à la racine du vault (ex: Documents/OCR) : ").strip().strip('"').strip("'")
    if cfg.vault_figures_dir is None:
        cfg.vault_figures_dir = input("Chemin des figures relatif à la RACINE du vault (ex: Documents/OCR/Files) : ").strip().strip('"').strip("'")


def fix_image_paths_obsidian(text: str, vault_figures_dir: str) -> str:
    """
    Convertit les balises <img src="imgs/..."> en wikilinks Obsidian ![[...]].
    Utilisé pendant l'OCR (chemins raw PaddleOCR).
    Le chemin est construit depuis la racine du vault.
    """
    if not vault_figures_dir:
        raise ValueError("vault_figures_dir non configuré")
    prefix = vault_figures_dir.replace("\\", "/").rstrip("/")

    def _replace(m: re.Match) -> str:
        return f'![[{prefix}/{m.group(1)}]]'

    text = re.sub(r'<img\b[^>]*\bsrc="imgs/([^"]+)"[^>]*/?\s*>', _replace, text)
    # Strip <div> wrappers that solely contain the image wikilink
    text = re.sub(r'<div\b[^>]*>\s*(!\[\[[^\]]+\]\])\s*</div>', r'\1', text)
    return text


def migrate_figures(cfg: Config, page_ids: list[str] | None = None, dry_run: bool = False) -> int:
    """
    Copie les figures depuis output/figures/*/imgs/* vers vault_root/vault_figures_dir/.
    Structure aplatie : page_001/imgs/fig.jpg → vault_figures_dir/fig.jpg.
    Skippe les fichiers déjà présents. Retourne le nombre de fichiers copiés.

    page_ids : si fourni, limite la copie aux sous-dossiers correspondants (pages du run courant).
               Si None, copie tout le contenu de figures_path.
    """
    if not cfg.vault_root or not cfg.vault_figures_dir:
        raise ValueError("vault_root et vault_figures_dir doivent être configurés")

    dest = Path(cfg.vault_root) / cfg.vault_figures_dir
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    if page_ids is not None:
        sources = sorted(
            src
            for page_id in page_ids
            for src in (cfg.figures_path / page_id / "imgs").glob("*")
            if (cfg.figures_path / page_id / "imgs").exists()
        )
    else:
        sources = sorted(cfg.figures_path.glob("*/imgs/*"))

    copied = 0
    for src in sources:
        if not src.is_file():
            continue
        target = dest / src.name
        if target.exists():
            continue
        if dry_run:
            logger.info("[dry-run] %s → %s", src.name, target)
        else:
            shutil.copy2(src, target)
            logger.info("Copié : %s", src.name)
        copied += 1

    logger.info("%d fichier(s) copié(s) vers %s", copied, dest)
    return copied


def postprocess_file(cfg: Config) -> None:
    """
    Applique le postprocess complet sur un fichier .md déjà généré (sans relancer l'OCR) :
      - nettoyage texte (clean_page, strip_table_styles)
      - conversion des balises <img> en wikilinks ![[]]
      - détection de headers (si cfg.header_patterns)
    """
    from postprocess import clean_page, strip_table_styles, apply_header_detection

    if not cfg.vault_figures_dir:
        raise ValueError("vault_figures_dir non configuré")
    prefix = cfg.vault_figures_dir.replace("\\", "/").rstrip("/")

    def _replace(m: re.Match) -> str:
        return f'![[{prefix}/{m.group(1)}]]'

    md_path = cfg.output_path
    text = md_path.read_text(encoding="utf-8")

    if cfg.postprocess:
        text = clean_page(text, cfg)
        text = strip_table_styles(text)

    text = re.sub(r'<img\b[^>]*\bsrc="[^"]*/imgs/([^"]+)"[^>]*/?\s*>', _replace, text)
    text = re.sub(r'<div\b[^>]*>\s*(!\[\[[^\]]+\]\])\s*</div>', r'\1', text)

    if cfg.header_patterns:
        text = apply_header_detection(text, cfg.header_patterns)

    md_path.write_text(text, encoding="utf-8", newline="\n")
