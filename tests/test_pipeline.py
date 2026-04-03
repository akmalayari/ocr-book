"""
test_pipeline.py — Tests unitaires pour pipeline.py

Stratégie :
  - VLM.from_() est mocké : aucun chargement de modèle réel.
  - ocr_image et collect_images sont mockés.
  - ocr_image retourne (text, metrics) comme dans la vraie implémentation.
  - On vérifie le comportement observable : contenu du .md, Stats, erreurs.

Couvre :
  - run_pipeline : succès, reprise, erreurs OCR partielles, toutes erreurs
  - run_pipeline : écriture incrémentale, en-tête Markdown, stats
  - run_pipeline : suppression du fichier si aucune page traitée
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config
from ocr_client import OCRError
from pipeline import run_pipeline
from postprocess import format_page_block

JPEG_BYTES = bytes([0xFF, 0xD8, 0xFF, 0xD9])


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_cfg(tmp_path):
    images_dir = tmp_path / "photos"
    images_dir.mkdir()
    return Config(
        model="NexaAI/TestModel",
        images_dir=str(images_dir),
        output_file=str(tmp_path / "output.md"),
        resume=False,
        log_file="",
        remove_isolated_page_numbers=False,
        rejoin_hyphenated_words=False,
        collapse_blank_lines=False,
    )


def make_images(cfg, count=3, prefix="page"):
    folder = Path(cfg.images_dir)
    paths = []
    for i in range(1, count + 1):
        p = folder / f"{prefix}_{i:03d}.jpg"
        p.write_bytes(JPEG_BYTES)
        paths.append(p)
    return paths


def run_with_mocks(cfg, texts_or_errors, image_paths=None):
    """
    Lance run_pipeline avec VLM.from_(), ocr_image et collect_images mockés.

    texts_or_errors : list de str (succès) ou OCRError (erreur).
    ocr_image retourne (text, metrics) en cas de succès.
    """
    if image_paths is None:
        image_paths = sorted(Path(cfg.images_dir).glob("*.jpg"))

    side_effects = []
    for item in texts_or_errors:
        if isinstance(item, OCRError):
            side_effects.append(item)
        else:
            side_effects.append((item, {"total_latency": 1.0}))

    mock_vlm = MagicMock()
    with patch("pipeline.VLM") as mock_vlm_class:
        mock_vlm_class.from_.return_value = mock_vlm
        with patch("pipeline.ocr_image", side_effect=side_effects):
            with patch("pipeline.collect_images", return_value=image_paths):
                return run_pipeline(cfg)


# ── Tests : succès complet ────────────────────────────────────────────────────

class TestPipelineSuccess:

    def test_returns_stats_object(self, tmp_cfg):
        make_images(tmp_cfg, 2)
        stats = run_with_mocks(tmp_cfg, ["T1", "T2"])
        assert stats is not None

    def test_stats_done_equals_image_count(self, tmp_cfg):
        make_images(tmp_cfg, 3)
        stats = run_with_mocks(tmp_cfg, ["T1", "T2", "T3"])
        assert stats.done == 3

    def test_stats_errors_zero_on_success(self, tmp_cfg):
        make_images(tmp_cfg, 2)
        stats = run_with_mocks(tmp_cfg, ["T1", "T2"])
        assert stats.errors == 0

    def test_stats_total_chars_accumulated(self, tmp_cfg):
        make_images(tmp_cfg, 2)
        stats = run_with_mocks(tmp_cfg, ["abc", "de"])
        assert stats.total_chars == 5

    def test_output_file_created(self, tmp_cfg):
        make_images(tmp_cfg, 1)
        run_with_mocks(tmp_cfg, ["Texte"])
        assert Path(tmp_cfg.output_file).exists()

    def test_output_contains_page_markers(self, tmp_cfg):
        make_images(tmp_cfg, 2)
        run_with_mocks(tmp_cfg, ["Texte 1", "Texte 2"])
        content = Path(tmp_cfg.output_file).read_text(encoding="utf-8")
        assert "<!-- Page" in content

    def test_output_contains_ocr_text(self, tmp_cfg):
        make_images(tmp_cfg, 1)
        run_with_mocks(tmp_cfg, ["Contenu unique de la page"])
        content = Path(tmp_cfg.output_file).read_text(encoding="utf-8")
        assert "Contenu unique de la page" in content

    def test_output_pages_in_order(self, tmp_cfg):
        paths = make_images(tmp_cfg, 3)
        run_with_mocks(tmp_cfg, ["Page A", "Page B", "Page C"], image_paths=paths)
        content = Path(tmp_cfg.output_file).read_text(encoding="utf-8")
        assert content.index("Page A") < content.index("Page B") < content.index("Page C")

    def test_new_file_has_markdown_header(self, tmp_cfg):
        make_images(tmp_cfg, 1)
        run_with_mocks(tmp_cfg, ["Texte"])
        content = Path(tmp_cfg.output_file).read_text(encoding="utf-8")
        assert content.startswith("# ")

    def test_stats_times_list_has_one_entry_per_page(self, tmp_cfg):
        make_images(tmp_cfg, 3)
        stats = run_with_mocks(tmp_cfg, ["T1", "T2", "T3"])
        assert len(stats.times) == 3

    def test_vlm_loaded_once(self, tmp_cfg):
        make_images(tmp_cfg, 3)
        mock_vlm = MagicMock()
        with patch("pipeline.VLM") as mock_vlm_class:
            mock_vlm_class.from_.return_value = mock_vlm
            with patch("pipeline.ocr_image", side_effect=[("T1", {"total_latency": 1.0})] * 3):
                with patch("pipeline.collect_images", return_value=sorted(Path(tmp_cfg.images_dir).glob("*.jpg"))):
                    run_pipeline(tmp_cfg)
        mock_vlm_class.from_.assert_called_once()

    def test_output_deleted_when_no_page_processed(self, tmp_cfg):
        """Si toutes les pages échouent sur un nouveau fichier, le fichier est supprimé."""
        make_images(tmp_cfg, 2)
        errors = [OCRError("E1"), OCRError("E2")]
        run_with_mocks(tmp_cfg, errors)
        assert not Path(tmp_cfg.output_file).exists()


# ── Tests : reprise (resume) ──────────────────────────────────────────────────

class TestPipelineResume:

    def _make_existing_output(self, cfg, done_page_ids: list):
        content = "# Livre OCR\n\n"
        for page_id in done_page_ids:
            content += format_page_block(page_id, f"Contenu de {page_id}")
        Path(cfg.output_file).write_text(content, encoding="utf-8")

    def test_skips_already_done_pages(self, tmp_cfg):
        tmp_cfg.resume = True
        paths = make_images(tmp_cfg, 3)
        self._make_existing_output(tmp_cfg, ["page_001"])
        stats = run_with_mocks(tmp_cfg, ["T2", "T3"], image_paths=paths)
        assert stats.skipped == 1
        assert stats.done == 2

    def test_all_pages_already_done(self, tmp_cfg):
        tmp_cfg.resume = True
        paths = make_images(tmp_cfg, 3)
        self._make_existing_output(tmp_cfg, ["page_001", "page_002", "page_003"])
        stats = run_with_mocks(tmp_cfg, [], image_paths=paths)
        assert stats.skipped == 3
        assert stats.done == 0

    def test_no_resume_rewrites_file(self, tmp_cfg):
        tmp_cfg.resume = False
        paths = make_images(tmp_cfg, 1)
        Path(tmp_cfg.output_file).write_text("# Ancien contenu\n\nTexte précédent", encoding="utf-8")
        run_with_mocks(tmp_cfg, ["Nouveau texte"], image_paths=paths)
        content = Path(tmp_cfg.output_file).read_text(encoding="utf-8")
        assert "Ancien contenu" not in content

    def test_resume_appends_without_duplicate_header(self, tmp_cfg):
        tmp_cfg.resume = True
        paths = make_images(tmp_cfg, 2)
        self._make_existing_output(tmp_cfg, ["page_001"])
        run_with_mocks(tmp_cfg, ["Nouveau texte"], image_paths=paths)
        content = Path(tmp_cfg.output_file).read_text(encoding="utf-8")
        assert content.count("# Livre OCR") == 1

    def test_stats_total_includes_skipped(self, tmp_cfg):
        tmp_cfg.resume = True
        paths = make_images(tmp_cfg, 5)
        self._make_existing_output(tmp_cfg, ["page_001", "page_002"])
        stats = run_with_mocks(tmp_cfg, ["T3", "T4", "T5"], image_paths=paths)
        assert stats.total == 5


# ── Tests : gestion des erreurs OCR ──────────────────────────────────────────

class TestPipelineOCRErrors:

    def test_continues_after_ocr_error(self, tmp_cfg):
        paths = make_images(tmp_cfg, 3)
        side_effects = ["Texte page 1", OCRError("Timeout"), "Texte page 3"]
        stats = run_with_mocks(tmp_cfg, side_effects, image_paths=paths)
        assert stats.done == 2
        assert stats.errors == 1

    def test_error_block_written_to_output(self, tmp_cfg):
        paths = make_images(tmp_cfg, 2)
        run_with_mocks(tmp_cfg, [OCRError("Connexion refusée"), "Texte OK"], image_paths=paths)
        content = Path(tmp_cfg.output_file).read_text(encoding="utf-8")
        assert "ERREUR" in content

    def test_subsequent_pages_processed_after_error(self, tmp_cfg):
        paths = make_images(tmp_cfg, 3)
        run_with_mocks(tmp_cfg, [OCRError("Crash"), "Page 2 OK", "Page 3 OK"], image_paths=paths)
        content = Path(tmp_cfg.output_file).read_text(encoding="utf-8")
        assert "Page 2 OK" in content
        assert "Page 3 OK" in content

    def test_all_errors_stats(self, tmp_cfg):
        paths = make_images(tmp_cfg, 3)
        stats = run_with_mocks(tmp_cfg, [OCRError("E1"), OCRError("E2"), OCRError("E3")], image_paths=paths)
        assert stats.errors == 3
        assert stats.done == 0

    def test_error_page_id_in_error_block(self, tmp_cfg):
        paths = make_images(tmp_cfg, 2)
        run_with_mocks(tmp_cfg, [OCRError("Timeout"), "Page 2 OK"], image_paths=paths)
        content = Path(tmp_cfg.output_file).read_text(encoding="utf-8")
        assert "page_001" in content
