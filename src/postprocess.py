"""
postprocess.py — Nettoyage du texte OCR avant écriture dans le Markdown final
"""

import re
import logging

from config import Config

logger = logging.getLogger(__name__)


_GROUNDING_RE  = re.compile(r'<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>\n?')
_FOOTNOTE_ROW_RE = re.compile(r'<tr><td>((?:(?!</td>).)+)</td>(?:<td></td>)+</tr>')


def extract_table_footnotes(text: str) -> str:
    """Déplace hors du tableau les <tr> dont seule la première cellule est non vide."""
    footnotes = []

    def _pull_row(m):
        footnotes.append(m.group(1))
        return ''

    cleaned = _FOOTNOTE_ROW_RE.sub(_pull_row, text)
    if footnotes:
        cleaned = cleaned.rstrip() + '\n\n' + '\n'.join(footnotes) + '\n'
    return cleaned


def _clean_layout(text: str) -> str:
    """Supprime les balises grounding et fusionne les blocs de continuation."""
    text = _GROUNDING_RE.sub('', text)

    blocks = re.split(r'\n{2,}', text.strip())
    merged = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if merged and block and block[0].islower():
            merged[-1] = merged[-1].rstrip() + ' ' + block
        else:
            merged.append(block)

    return '\n\n'.join(merged)


def clean_page(text: str, cfg: Config) -> str:
    """
    Applique les nettoyages activés dans la config :
      - suppression des numéros de page isolés
      - réassemblage des mots coupés en fin de ligne (ex: condi-\ntion)
      - réduction des lignes vides excessives
    """
    if cfg.prompt_mode == "layout":
        text = _clean_layout(text)

    if cfg.remove_isolated_page_numbers:
        # Ligne ne contenant qu'un nombre (ex: " 42 ")
        text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)

    if cfg.rejoin_hyphenated_words:
        # "condi-\ntion" → "condition"
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        # "distribu- tion" → "distribution" (tiret + espace en fin/milieu de ligne)
        text = re.sub(r'(\w)- (\w)', r'\1\2', text)

    if cfg.collapse_blank_lines:
        # 3+ sauts de ligne → 2 max
        text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def format_page_block(page_id: str, text: str) -> str:
    """
    Formate un bloc de page avec un commentaire HTML pour reprise facile.
    Permet de repérer les pages dans le fichier de sortie.
    """
    return f"\n\n<!-- Page {page_id} -->\n\n{text}\n"


def format_error_block(page_id: str, error: str) -> str:
    """Insère un marqueur d'erreur pour la page concernée."""
    return f"\n\n<!-- Page {page_id} — ERREUR: {error} -->\n"


def extract_done_pages(output_text: str) -> set[str]:
    """
    Lit un fichier de sortie existant et retourne les page_id déjà traités.
    Permet la reprise après interruption.
    """
    return set(re.findall(r'<!-- Page (\S+) -->', output_text))
