"""
obsidian.py — Utilitaires pour l'export Obsidian
"""

import re
from pathlib import Path

from config import Config


def prompt_if_needed(cfg: Config) -> None:
    """Prompt interactif pour vault_path et vault_figures_dir si non configurés."""
    if cfg.vault_path is None:
        cfg.vault_path = input("Chemin absolu du vault Obsidian : ").strip()
    if cfg.vault_figures_dir is None:
        cfg.vault_figures_dir = input("Chemin des figures dans le vault (relatif à la racine, ex: Livres/figures) : ").strip()


def fix_image_paths_obsidian(text: str, vault_figures_dir: str) -> str:
    """
    Convertit les balises <img src="imgs/..."> en wikilinks Obsidian ![[...]].
    Utilisé pendant l'OCR (chemins raw PaddleOCR).
    Le chemin est construit depuis la racine du vault.
    """
    prefix = vault_figures_dir.replace("\\", "/").rstrip("/")

    def _replace(m: re.Match) -> str:
        return f'![[{prefix}/{m.group(1)}]]'

    return re.sub(r'<img\b[^>]*\bsrc="imgs/([^"]+)"[^>]*/?\s*>', _replace, text)


def postprocess_file(cfg: Config) -> None:
    """
    Applique le postprocess obsidian sur un fichier .md déjà généré (sans relancer l'OCR).
    Convertit les balises <img src=".../.../imgs/..."> en wikilinks ![[...]].
    """
    prefix = cfg.vault_figures_dir.replace("\\", "/").rstrip("/")

    def _replace(m: re.Match) -> str:
        return f'![[{prefix}/{m.group(1)}]]'

    md_path = cfg.output_path
    text = md_path.read_text(encoding="utf-8")
    result = re.sub(
        r'<img\b[^>]*\bsrc="[^"]*/imgs/([^"]+)"[^>]*/?\s*>',
        _replace,
        text,
    )
    md_path.write_text(result, encoding="utf-8", newline="\n")
