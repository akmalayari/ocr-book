"""
postprocess.py — Nettoyage du texte OCR avant écriture dans le Markdown final
"""

import re
import logging

from config import Config

logger = logging.getLogger(__name__)


def clean_page(text: str, cfg: Config) -> str:
    """
    Applique les nettoyages activés dans la config :
      - suppression des numéros de page isolés
      - réassemblage des mots coupés en fin de ligne (ex: condi-\ntion)
      - réduction des lignes vides excessives
    """
    if cfg.remove_isolated_page_numbers:
        text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)

    if cfg.rejoin_hyphenated_words:
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- (\w)', r'\1\2', text)
        # Paragraphe coupé mid-phrase : ligne se terminant par minuscule/virgule
        # suivie d'un paragraphe vide puis d'une minuscule ou '('
        text = re.sub(r'([a-zéèêëàâîïôùûüçœ,])\n\n(\(|[a-zéèêëàâîïôùûüçœ])', r'\1 \2', text)

    if cfg.collapse_blank_lines:
        text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def strip_table_styles(text: str) -> str:
    """
    Supprime les styles inline des balises table générés par PaddleOCR (pretty=True).
    Les <div style="..."> (captions, titres) sont conservés.
    Toujours appliqué, indépendamment de cfg.postprocess.
    """
    text = re.sub(r"<table\b[^>]*\bstyle='[^']*'", "<table border=1", text)
    text = re.sub(r"<(t[dh])\b[^>]*\bstyle='[^']*'>", r"<\1>", text)
    return text


def format_page_block(page_id: str, text: str) -> str:
    return f"\n\n<!-- Page {page_id} -->\n\n{text}\n"


def format_error_block(page_id: str, error: str) -> str:
    return f"\n\n<!-- Page {page_id} — ERREUR: {error} -->\n"


def extract_done_pages(output_text: str) -> set[str]:
    return set(re.findall(r'<!-- Page (\S+) -->', output_text))


def fix_image_paths(text: str, page_id: str, figures_rel: str) -> str:
    """
    Corrige les chemins relatifs aux images générées par PaddleOCR.
    PaddleOCR utilise 'imgs/...' relatif au dossier de la page.
    Quand on combine tous les markdowns en un seul fichier, on recalcule
    le chemin relatif depuis le dossier du fichier de sortie.
    """
    prefix = figures_rel.replace("\\", "/")
    return re.sub(r'src="imgs/', f'src="{prefix}/{page_id}/imgs/', text)
