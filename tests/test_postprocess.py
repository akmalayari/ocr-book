"""
Unit tests for src/postprocess.py — pure-logic functions only,
no llama-server or model stack required.
"""

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config import Config
from postprocess import (
    apply_header_detection,
    clean_page,
    extract_done_pages,
    extract_page_number,
    fix_double_scripts,
    fix_image_paths,
    format_error_block,
    format_page_block,
    strip_math_spacing,
    strip_table_styles,
)


def _cfg(**kwargs) -> Config:
    """Return a Config with postprocess flags set explicitly."""
    cfg = Config()
    cfg.remove_isolated_page_numbers = kwargs.get("remove_isolated_page_numbers", True)
    cfg.rejoin_hyphenated_words = kwargs.get("rejoin_hyphenated_words", True)
    cfg.collapse_blank_lines = kwargs.get("collapse_blank_lines", True)
    return cfg


# ---------------------------------------------------------------------------
# clean_page
# ---------------------------------------------------------------------------

class TestCleanPagePageNumbers(unittest.TestCase):
    def test_removes_isolated_page_number(self):
        text = "Some text\n\n42\n\nMore text"
        result = clean_page(text, _cfg())
        self.assertNotIn("42", result)

    def test_removes_page_number_with_surrounding_spaces(self):
        text = "Intro\n  7  \nBody"
        result = clean_page(text, _cfg())
        self.assertNotIn("7", result)

    def test_keeps_inline_number(self):
        text = "There are 42 items listed here."
        result = clean_page(text, _cfg())
        self.assertIn("42", result)

    def test_flag_disabled_keeps_page_number(self):
        text = "Text\n\n42\n\nMore"
        result = clean_page(text, _cfg(remove_isolated_page_numbers=False))
        self.assertIn("42", result)


class TestCleanPageHyphenation(unittest.TestCase):
    def test_rejoins_single_newline_hyphen(self):
        text = "condi-\ntion"
        result = clean_page(text, _cfg())
        self.assertIn("condition", result)
        self.assertNotIn("-\n", result)

    def test_rejoins_double_newline_hyphen_lowercase(self):
        text = "condi-\n\ntion"
        result = clean_page(text, _cfg())
        self.assertIn("condition", result)

    def test_does_not_rejoin_double_newline_hyphen_uppercase(self):
        # Uppercase after double break = new paragraph/title, must not merge
        text = "condi-\n\nTitle"
        result = clean_page(text, _cfg())
        self.assertIn("Title", result)
        self.assertNotIn("condiTitle", result)

    def test_removes_inline_hyphen_space(self):
        text = "condi- tion"
        result = clean_page(text, _cfg())
        self.assertIn("condition", result)

    def test_rejoins_mid_sentence_paragraph_break_lowercase(self):
        # Line ending with lowercase, then blank line, then lowercase → merge
        text = "the quick brown\n\nfox jumps"
        result = clean_page(text, _cfg())
        self.assertIn("the quick brown fox jumps", result)

    def test_rejoins_mid_sentence_paragraph_break_after_comma(self):
        text = "first part,\n\nsecond part"
        result = clean_page(text, _cfg())
        self.assertIn("first part, second part", result)

    def test_flag_disabled_keeps_hyphens(self):
        text = "condi-\ntion"
        result = clean_page(text, _cfg(rejoin_hyphenated_words=False))
        self.assertIn("-\n", result)

    def test_repetition_loop_removed_no_layout(self):
        repeated = "abcdefghijklmnop" * 4  # 16 chars × 4 repetitions
        result = clean_page(repeated, _cfg(), no_layout=True)
        self.assertLess(len(result), len(repeated))

    def test_repetition_loop_kept_with_layout(self):
        # Without no_layout=True, the dedup regex is not applied
        repeated = "abcdefghijklmnop" * 4
        result = clean_page(repeated, _cfg(), no_layout=False)
        self.assertEqual(result, repeated)


class TestCleanPageBlankLines(unittest.TestCase):
    def test_collapses_triple_blank_lines(self):
        text = "A\n\n\n\nB"
        result = clean_page(text, _cfg())
        self.assertNotIn("\n\n\n", result)
        self.assertIn("A", result)
        self.assertIn("B", result)

    def test_double_blank_line_untouched(self):
        text = "A\n\nB"
        result = clean_page(text, _cfg())
        self.assertIn("\n\n", result)

    def test_flag_disabled_keeps_blank_lines(self):
        text = "A\n\n\n\nB"
        result = clean_page(text, _cfg(collapse_blank_lines=False))
        self.assertIn("\n\n\n", result)


# ---------------------------------------------------------------------------
# strip_table_styles
# ---------------------------------------------------------------------------

class TestStripTableStyles(unittest.TestCase):
    def test_removes_style_from_table(self):
        html = "<table style='color:red; width:100%'>content</table>"
        result = strip_table_styles(html)
        self.assertNotIn("style=", result)
        self.assertIn('<table align="center" border=1', result)

    def test_removes_style_from_td(self):
        html = "<td style='padding:4px'>cell</td>"
        result = strip_table_styles(html)
        self.assertEqual(result, "<td>cell</td>")

    def test_removes_style_from_th(self):
        html = "<th style='font-weight:bold'>header</th>"
        result = strip_table_styles(html)
        self.assertEqual(result, "<th>header</th>")

    def test_keeps_div_style(self):
        html = "<div style='text-align:center'>caption</div>"
        result = strip_table_styles(html)
        self.assertEqual(result, html)

    def test_table_without_style_unchanged(self):
        html = '<table align="center" border=1>data</table>'
        result = strip_table_styles(html)
        self.assertEqual(result, html)


# ---------------------------------------------------------------------------
# strip_math_spacing
# ---------------------------------------------------------------------------

class TestStripMathSpacing(unittest.TestCase):
    def test_strips_leading_space(self):
        self.assertEqual(strip_math_spacing("$ x $"), "$x$")

    def test_strips_leading_and_trailing_spaces(self):
        self.assertEqual(strip_math_spacing("$  a + b  $"), "$a + b$")

    def test_already_clean_unchanged(self):
        self.assertEqual(strip_math_spacing("$x + y$"), "$x + y$")

    def test_display_math_untouched(self):
        text = "$$ x + y $$"
        self.assertEqual(strip_math_spacing(text), text)

    def test_inline_inside_sentence(self):
        result = strip_math_spacing("The value is $ n^2 $ here.")
        self.assertIn("$n^2$", result)

    def test_multiple_inline_formulas(self):
        result = strip_math_spacing("$ a $ and $ b $")
        self.assertIn("$a$", result)
        self.assertIn("$b$", result)


# ---------------------------------------------------------------------------
# fix_double_scripts
# ---------------------------------------------------------------------------

class TestFixDoubleScripts(unittest.TestCase):
    def test_double_superscript_inline(self):
        result = fix_double_scripts("$x^a^b$")
        self.assertIn("^{a^b}", result)
        self.assertNotIn("^a^b", result)

    def test_double_subscript_inline(self):
        result = fix_double_scripts("$x_a_b$")
        self.assertIn("_{a_b}", result)
        self.assertNotIn("_a_b", result)

    def test_braced_left_superscript(self):
        result = fix_double_scripts("$x^{a}^b$")
        self.assertIn("^{a^b}", result)

    def test_braced_right_superscript(self):
        result = fix_double_scripts("$x^a^{b}$")
        self.assertIn("^{a^b}", result)

    def test_double_superscript_display(self):
        result = fix_double_scripts("$$x^a^b$$")
        self.assertIn("^{a^b}", result)

    def test_no_double_script_unchanged(self):
        text = "$x^{ab}$"
        result = fix_double_scripts(text)
        self.assertEqual(result, text)

    def test_triple_chained_superscript(self):
        result = fix_double_scripts("$x^a^b^c$")
        # After iterative fix, no bare double-script should remain
        self.assertNotIn("^a^b", result)


# ---------------------------------------------------------------------------
# extract_page_number
# ---------------------------------------------------------------------------

class TestExtractPageNumber(unittest.TestCase):
    def test_number_on_first_line(self):
        text = "42\nSome content here."
        label, cleaned = extract_page_number(text)
        self.assertEqual(label, "42")
        self.assertNotIn("42", cleaned.splitlines()[0] if cleaned else "")

    def test_number_on_last_line(self):
        text = "Some content here.\n99"
        label, cleaned = extract_page_number(text)
        self.assertEqual(label, "99")

    def test_two_consecutive_numbers_become_range(self):
        text = "12\n13\nContent"
        label, _ = extract_page_number(text)
        self.assertEqual(label, "12-13")

    def test_no_number_returns_none(self):
        text = "No page number in this text at all."
        label, cleaned = extract_page_number(text)
        self.assertIsNone(label)
        self.assertEqual(cleaned, text)

    def test_number_in_middle_not_extracted(self):
        # A number beyond the first/last 5 lines must not be extracted
        lines = ["line one", "line two", "line three", "line four", "line five",
                 "42",
                 "line seven", "line eight", "line nine", "line ten", "line eleven"]
        text = "\n".join(lines)
        label, _ = extract_page_number(text)
        self.assertIsNone(label)

    def test_number_removed_from_cleaned_text(self):
        text = "5\nBody text."
        _, cleaned = extract_page_number(text)
        self.assertNotIn("\n5\n", f"\n{cleaned}\n")


# ---------------------------------------------------------------------------
# apply_header_detection
# ---------------------------------------------------------------------------

class TestApplyHeaderDetection(unittest.TestCase):
    def _patterns(self):
        return [("^Chapter \\d+", 1), ("^Section \\d+", 2)]

    def test_matching_line_gets_header(self):
        text = "Chapter 3"
        result = apply_header_detection(text, self._patterns())
        self.assertTrue(result.startswith("# Chapter 3"))

    def test_matching_section_gets_h2(self):
        text = "Section 1"
        result = apply_header_detection(text, self._patterns())
        self.assertTrue(result.startswith("## Section 1"))

    def test_existing_header_skipped(self):
        text = "# Already a header"
        result = apply_header_detection(text, self._patterns())
        self.assertEqual(result, text)

    def test_line_over_120_chars_skipped(self):
        text = "Chapter 1 " + "x" * 115
        result = apply_header_detection(text, self._patterns())
        self.assertFalse(result.startswith("#"))

    def test_line_ending_with_comma_skipped(self):
        result = apply_header_detection("Chapter 1,", self._patterns())
        self.assertFalse(result.startswith("#"))

    def test_line_ending_with_semicolon_skipped(self):
        result = apply_header_detection("Chapter 1;", self._patterns())
        self.assertFalse(result.startswith("#"))

    def test_line_ending_with_colon_skipped(self):
        result = apply_header_detection("Chapter 1:", self._patterns())
        self.assertFalse(result.startswith("#"))

    def test_html_comment_line_skipped(self):
        text = "<!-- Page 3 -->"
        result = apply_header_detection(text, self._patterns())
        self.assertEqual(result, text)

    def test_non_matching_line_unchanged(self):
        text = "Just a regular paragraph."
        result = apply_header_detection(text, self._patterns())
        self.assertEqual(result, text)


# ---------------------------------------------------------------------------
# fix_image_paths
# ---------------------------------------------------------------------------

class TestFixImagePaths(unittest.TestCase):
    def test_rewrites_relative_path(self):
        text = '<img src="imgs/fig1.png">'
        result = fix_image_paths(text, "page_003", "figures")
        self.assertIn('src="figures/page_003/imgs/fig1.png"', result)

    def test_backslash_in_figures_rel_normalised(self):
        text = '<img src="imgs/fig1.png">'
        result = fix_image_paths(text, "page_001", "output\\figures")
        self.assertIn("output/figures/page_001/imgs/fig1.png", result)

    def test_no_img_tag_unchanged(self):
        text = "No images here."
        result = fix_image_paths(text, "page_001", "figures")
        self.assertEqual(result, text)


# ---------------------------------------------------------------------------
# format_page_block / format_error_block / extract_done_pages
# ---------------------------------------------------------------------------

class TestFormatters(unittest.TestCase):
    def test_format_page_block_with_page_number(self):
        result = format_page_block("page_001", "Content", "42")
        self.assertIn("<!-- Page page_001 (p. 42) -->", result)
        self.assertIn("Content", result)

    def test_format_page_block_without_page_number(self):
        result = format_page_block("page_001", "Content")
        self.assertIn("<!-- Page page_001 -->", result)
        self.assertNotIn("p. ", result)

    def test_format_error_block(self):
        result = format_error_block("page_002", "timeout")
        self.assertIn("<!-- Page page_002 — ERROR: timeout -->", result)

    def test_extract_done_pages_single(self):
        text = "<!-- Page page_001 -->\nsome text\n<!-- Page page_002 (p. 5) -->"
        pages = extract_done_pages(text)
        self.assertIn("page_001", pages)
        self.assertIn("page_002", pages)

    def test_extract_done_pages_empty(self):
        self.assertEqual(extract_done_pages("no pages here"), set())


if __name__ == "__main__":
    unittest.main()
