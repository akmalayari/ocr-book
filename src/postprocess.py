"""
postprocess.py — Nettoyage du texte OCR avant écriture dans le Markdown final
"""

import re
import logging

from config import Config

logger = logging.getLogger(__name__)


def clean_page(text: str, cfg: Config, no_layout: bool = False) -> str:
    """
    Applique les nettoyages activés dans la config :
      - suppression des numéros de page isolés
      - réassemblage des mots coupés en fin de ligne (ex: condi-\ntion)
      - réduction des lignes vides excessives
    no_layout : True si la page a été traitée sans layout detection (fallback).
    """
    if cfg.remove_isolated_page_numbers:
        text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)

    if cfg.rejoin_hyphenated_words:
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        # Continuation minuscule après coupure avec double saut (layout ou no-layout)
        # Restreint aux minuscules pour ne pas coller avec un titre/paragraphe suivant
        text = re.sub(r'(\w)-\n\n([a-zéèêëàâîïôùûüçœ])', r'\1\2', text)
        if no_layout:
            # Supprime les boucles de génération : sous-chaîne ≥15 chars répétée 3+ fois
            text = re.sub(r'(.{15,}?)\1{2,}', r'\1', text)
        text = re.sub(r'(\w)- (\w)', r'\1\2', text)
        # Paragraphe coupé mid-phrase : ligne se terminant par minuscule/virgule
        # suivie d'un paragraphe vide puis d'une minuscule ou '('
        text = re.sub(r'([a-zéèêëàâîïôùûüçœ,])\n\n(\(|[a-zéèêëàâîïôùûüçœ\d])', r'\1 \2', text)

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


def extract_page_number(text: str) -> tuple[str | None, str]:
    """
    Extrait le(s) numéro(s) de page imprimés depuis les N premières/dernières lignes.
    Retourne (page_number, texte_nettoyé) où page_number vaut None, "42", ou "42-43".

    Règle : un seul numéro, ou deux consécutifs → plage ; sinon premier rencontré.
    Les lignes candidates sont toujours supprimées du texte retourné.
    """
    N = 5
    lines = text.splitlines()
    n = len(lines)
    search = set(range(min(N, n))) | set(range(max(0, n - N), n))

    found = []
    for i in sorted(search):
        m = re.match(r'^\s*(\d{1,3})\s*$', lines[i])
        if m:
            found.append((i, int(m.group(1))))

    if not found:
        return None, text

    nums = [num for _, num in found]
    if len(nums) >= 2 and nums[1] == nums[0] + 1:
        label = f"{nums[0]}-{nums[1]}"
    else:
        label = str(nums[0])

    remove = {i for i, _ in found}
    cleaned = '\n'.join(line for i, line in enumerate(lines) if i not in remove)
    return label, cleaned


def format_page_block(page_id: str, text: str, page_number: str | None = None) -> str:
    label = f"{page_id} (p. {page_number})" if page_number else page_id
    return f"\n\n<!-- Page {label} -->\n\n{text}\n"


def format_error_block(page_id: str, error: str) -> str:
    return f"\n\n<!-- Page {page_id} — ERREUR: {error} -->\n"


def extract_done_pages(output_text: str) -> set[str]:
    return set(re.findall(r'<!-- Page (\S+)', output_text))


def fix_image_paths(text: str, page_id: str, figures_rel: str) -> str:
    """
    Corrige les chemins relatifs aux images générées par PaddleOCR.
    PaddleOCR utilise 'imgs/...' relatif au dossier de la page.
    Quand on combine tous les markdowns en un seul fichier, on recalcule
    le chemin relatif depuis le dossier du fichier de sortie.
    """
    prefix = figures_rel.replace("\\", "/")
    return re.sub(r'src="imgs/', f'src="{prefix}/{page_id}/imgs/', text)


