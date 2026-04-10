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
    """Prompt interactif pour vault_path et vault_figures_dir si non configurés."""
    if cfg.vault_path is None:
        cfg.vault_path = input("Chemin absolu du vault Obsidian : ").strip().strip('"').strip("'")
    if cfg.vault_figures_dir is None:
        cfg.vault_figures_dir = input("Chemin des figures dans le vault (relatif à la racine, ex: Livres/figures) : ").strip().strip('"').strip("'")


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


def migrate_figures(cfg: Config, dry_run: bool = False) -> int:
    """
    Copie les figures depuis output/figures/*/imgs/* vers vault_path/vault_figures_dir/.
    Structure aplatie : page_001/imgs/fig.jpg → vault_figures_dir/fig.jpg.
    Skippe les fichiers déjà présents. Retourne le nombre de fichiers copiés.
    """
    if not cfg.vault_path or not cfg.vault_figures_dir:
        raise ValueError("vault_path et vault_figures_dir doivent être configurés")

    dest = Path(cfg.vault_path) / cfg.vault_figures_dir
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)

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
    Applique le postprocess obsidian sur un fichier .md déjà généré (sans relancer l'OCR).
    Convertit les balises <img src=".../.../imgs/..."> en wikilinks ![[...]].
    """
    if not cfg.vault_figures_dir:
        raise ValueError("vault_figures_dir non configuré")
    prefix = cfg.vault_figures_dir.replace("\\", "/").rstrip("/")

    def _replace(m: re.Match) -> str:
        return f'![[{prefix}/{m.group(1)}]]'

    md_path = cfg.output_path
    text = md_path.read_text(encoding="utf-8")
    result = re.sub(r'<img\b[^>]*\bsrc="[^"]*/imgs/([^"]+)"[^>]*/?\s*>', _replace, text)
    result = re.sub(r'<div\b[^>]*>\s*(!\[\[[^\]]+\]\])\s*</div>', r'\1', result)
    md_path.write_text(result, encoding="utf-8", newline="\n")
