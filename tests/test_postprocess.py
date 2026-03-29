"""
test_postprocess.py — Tests unitaires pour postprocess.py

Couvre :
  - clean_page : chaque nettoyage individuellement + combinaisons
  - clean_page : cas limites (texte vide, texte déjà propre, grands textes)
  - format_page_block : structure et contenu
  - format_error_block : structure et contenu
  - extract_done_pages : détection, multi-pages, cas vides, pages d'erreur
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from postprocess import (
    clean_page,
    extract_done_pages,
    format_error_block,
    format_page_block,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def cfg_with(remove_page_nums=True, rejoin_hyphens=True, collapse_blanks=True):
    return Config(
        remove_isolated_page_numbers=remove_page_nums,
        rejoin_hyphenated_words=rejoin_hyphens,
        collapse_blank_lines=collapse_blanks,
        log_file="",
    )


# ── clean_page : suppression des numéros de page ─────────────────────────────

class TestCleanPageIsolatedNumbers:

    def test_removes_single_digit_page_number(self):
        cfg = cfg_with()
        result = clean_page("Texte\n5\nSuite", cfg)
        assert "5" not in result.split("\n") or result.count("\n5\n") == 0

    def test_removes_three_digit_page_number(self):
        cfg = cfg_with()
        text = "Paragraphe A\n\n123\n\nParagraphe B"
        result = clean_page(text, cfg)
        assert "\n123\n" not in result

    def test_removes_page_number_with_surrounding_spaces(self):
        cfg = cfg_with()
        text = "Debut\n  42  \nFin"
        result = clean_page(text, cfg)
        assert "  42  " not in result

    def test_does_not_remove_inline_number(self):
        """Un nombre dans une phrase ne doit pas être supprimé."""
        cfg = cfg_with()
        text = "Il y a 42 pommes dans le panier."
        result = clean_page(text, cfg)
        assert "42" in result

    def test_does_not_remove_four_digit_number(self):
        """Les nombres à 4 chiffres (années, etc.) ne doivent pas être supprimés."""
        cfg = cfg_with()
        text = "Début\n1984\nFin"
        result = clean_page(text, cfg)
        assert "1984" in result

    def test_disabled_leaves_page_numbers(self):
        cfg = cfg_with(remove_page_nums=False)
        text = "Texte\n42\nSuite"
        result = clean_page(text, cfg)
        assert "42" in result


# ── clean_page : réassemblage des mots coupés ─────────────────────────────────

class TestCleanPageHyphenatedWords:

    def test_rejoins_hyphenated_word(self):
        cfg = cfg_with()
        text = "Le mot condi-\ntion est réassemblé."
        result = clean_page(text, cfg)
        assert "condition" in result
        assert "condi-\ntion" not in result

    def test_rejoins_multiple_hyphenated_words(self):
        cfg = cfg_with()
        text = "C'est impor-\ntant et néces-\nsaire."
        result = clean_page(text, cfg)
        assert "important" in result
        assert "nécessaire" in result

    def test_does_not_rejoin_standalone_hyphen(self):
        """Un tiret entre deux espaces (liste) ne doit pas être affecté."""
        cfg = cfg_with()
        text = "- Premier élément\n- Deuxième élément"
        result = clean_page(text, cfg)
        assert "- Premier" in result
        assert "- Deuxième" in result

    def test_disabled_leaves_hyphens(self):
        cfg = cfg_with(rejoin_hyphens=False)
        text = "Le mot condi-\ntion reste tel quel."
        result = clean_page(text, cfg)
        assert "condi-\ntion" in result


# ── clean_page : réduction des lignes vides ───────────────────────────────────

class TestCleanPageBlankLines:

    def test_collapses_three_blank_lines(self):
        cfg = cfg_with()
        text = "Paragraphe A\n\n\n\nParagraphe B"
        result = clean_page(text, cfg)
        assert "\n\n\n" not in result

    def test_collapses_many_blank_lines(self):
        cfg = cfg_with()
        text = "A\n\n\n\n\n\n\nB"
        result = clean_page(text, cfg)
        assert "\n\n\n" not in result
        assert "A" in result and "B" in result

    def test_preserves_double_blank_line(self):
        """Un double saut de ligne (séparation de paragraphe) doit être conservé."""
        cfg = cfg_with()
        text = "Para A\n\nPara B"
        result = clean_page(text, cfg)
        assert "\n\n" in result

    def test_disabled_leaves_blank_lines(self):
        cfg = cfg_with(collapse_blanks=False)
        text = "A\n\n\n\nB"
        result = clean_page(text, cfg)
        assert "\n\n\n" in result


# ── clean_page : cas limites ──────────────────────────────────────────────────

class TestCleanPageEdgeCases:

    def test_empty_string(self):
        cfg = cfg_with()
        assert clean_page("", cfg) == ""

    def test_whitespace_only(self):
        cfg = cfg_with()
        assert clean_page("   \n\n   ", cfg) == ""

    def test_strip_leading_trailing_whitespace(self):
        cfg = cfg_with()
        result = clean_page("\n\nTexte\n\n", cfg)
        assert not result.startswith("\n")
        assert not result.endswith("\n")

    def test_already_clean_text_unchanged(self):
        cfg = cfg_with()
        text = "## Titre\n\nParagraphe propre sans artefacts."
        result = clean_page(text, cfg)
        assert "## Titre" in result
        assert "Paragraphe propre" in result

    def test_all_cleanups_disabled_returns_stripped_text(self):
        cfg = cfg_with(remove_page_nums=False, rejoin_hyphens=False, collapse_blanks=False)
        text = "  Texte  "
        assert clean_page(text, cfg) == "Texte"

    def test_combined_cleanups(self):
        """Test avec plusieurs artefacts simultanés."""
        cfg = cfg_with()
        text = "\n42\n\nMot condi-\ntion\n\n\n\nFin\n"
        result = clean_page(text, cfg)
        assert "condition" in result
        assert "\n42\n" not in result
        assert "\n\n\n" not in result


# ── format_page_block ─────────────────────────────────────────────────────────

class TestFormatPageBlock:

    def test_contains_page_id_in_comment(self):
        block = format_page_block("page_001", "Contenu")
        assert "<!-- Page page_001 -->" in block

    def test_contains_text_content(self):
        block = format_page_block("page_001", "## Chapitre\n\nTexte.")
        assert "## Chapitre" in block
        assert "Texte." in block

    def test_starts_with_newlines(self):
        block = format_page_block("page_001", "Texte")
        assert block.startswith("\n\n")

    def test_comment_before_content(self):
        block = format_page_block("page_001", "Contenu")
        comment_pos = block.index("<!-- Page")
        content_pos = block.index("Contenu")
        assert comment_pos < content_pos

    def test_ends_with_newline(self):
        block = format_page_block("page_001", "Texte")
        assert block.endswith("\n")

    def test_different_page_ids(self):
        b1 = format_page_block("page_001", "A")
        b2 = format_page_block("page_042", "B")
        assert "page_001" in b1
        assert "page_042" in b2
        assert "page_001" not in b2

    def test_empty_content(self):
        block = format_page_block("page_001", "")
        assert "<!-- Page page_001 -->" in block


# ── format_error_block ────────────────────────────────────────────────────────

class TestFormatErrorBlock:

    def test_contains_page_id(self):
        block = format_error_block("page_007", "Timeout")
        assert "page_007" in block

    def test_contains_erreur_keyword(self):
        block = format_error_block("page_007", "Timeout 180s")
        assert "ERREUR" in block

    def test_contains_error_message(self):
        block = format_error_block("page_007", "Timeout 180s")
        assert "Timeout 180s" in block

    def test_is_html_comment(self):
        block = format_error_block("page_007", "msg")
        assert "<!--" in block and "-->" in block

    def test_starts_with_newlines(self):
        block = format_error_block("page_007", "msg")
        assert block.startswith("\n\n")

    def test_ends_with_newline(self):
        block = format_error_block("page_007", "msg")
        assert block.endswith("\n")


# ── extract_done_pages ────────────────────────────────────────────────────────

class TestExtractDonePages:

    def test_empty_string_returns_empty_set(self):
        assert extract_done_pages("") == set()

    def test_no_page_markers_returns_empty_set(self):
        assert extract_done_pages("Texte sans marqueurs") == set()

    def test_extracts_single_page(self):
        text = "<!-- Page page_001 -->\n\nContenu"
        result = extract_done_pages(text)
        assert "page_001" in result

    def test_extracts_multiple_pages(self):
        text = (
            "<!-- Page page_001 -->\n\nContenu 1\n\n"
            "<!-- Page page_002 -->\n\nContenu 2\n\n"
            "<!-- Page page_042 -->\n\nContenu 42"
        )
        result = extract_done_pages(text)
        assert result == {"page_001", "page_002", "page_042"}

    def test_returns_set_not_list(self):
        text = "<!-- Page page_001 -->\n"
        assert isinstance(extract_done_pages(text), set)

    def test_no_duplicates_for_repeated_page(self):
        """Si un page_id apparaît deux fois (anomalie), set() déduplique."""
        text = "<!-- Page page_001 -->\n<!-- Page page_001 -->\n"
        result = extract_done_pages(text)
        assert len(result) == 1

    def test_does_not_extract_error_page_id(self):
        """Les blocs ERREUR ne sont pas détectés comme 'faits' (retentés à la reprise)."""
        text = "<!-- Page page_005 — ERREUR: Timeout -->\n"
        result = extract_done_pages(text)
        assert "page_005" not in result

    def test_roundtrip_format_then_extract(self):
        """format_page_block → extract_done_pages doit retrouver le même page_id."""
        page_id = "page_123"
        block = format_page_block(page_id, "Contenu de test")
        result = extract_done_pages(block)
        assert page_id in result

    def test_roundtrip_error_format_does_not_extract(self):
        """format_error_block → extract_done_pages ne retourne pas le page_id (reprise)."""
        page_id = "page_007"
        block = format_error_block(page_id, "Erreur quelconque")
        result = extract_done_pages(block)
        assert page_id not in result

    def test_large_document_with_many_pages(self):
        """Test de performance/exactitude sur 100 pages."""
        pages = [format_page_block(f"page_{i:03d}", f"Contenu {i}") for i in range(1, 101)]
        text = "".join(pages)
        result = extract_done_pages(text)
        assert len(result) == 100
        assert "page_001" in result
        assert "page_100" in result
