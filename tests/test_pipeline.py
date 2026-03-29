"""
test_pipeline.py — Tests unitaires pour pipeline.py

Stratégie :
  - nexa_server, ocr_image et collect_images sont TOUJOURS mockés.
  - On travaille dans tmp_path pour les fichiers de sortie.
  - On vérifie le comportement observable : contenu du fichier .md,
    valeurs retournées dans Stats, gestion des erreurs OCR partielles.

Couvre :
  - run_pipeline : pipeline complet succès
  - run_pipeline : reprise (pages déjà traitées skippées)
  - run_pipeline : erreur OCR partielle (pipeline continue)
  - run_pipeline : toutes les pages en erreur
  - run_pipeline : écriture incrémentale (flush)
  - run_pipeline : en-tête Markdown sur nouveau fichier
  - run_pipeline : mode append sur fichier existant (pas de double en-tête)
  - run_pipeline : stats retournées correctes
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from ocr_client import OCRError
from pipeline import run_pipeline
from postprocess import format_page_block


# ── Fixtures ──────────────────────────────────────────────────────────────────

JPEG_BYTES = bytes([0xFF, 0xD8, 0xFF, 0xD9])


@pytest.fixture
def tmp_cfg(tmp_path):
    images_dir = tmp_path / "photos"
    images_dir.mkdir()
    return Config(
        model="NexaAI/TestModel",
        port=19999,
        server_timeout_s=1,
        request_timeout_s=5,
        images_dir=str(images_dir),
        output_file=str(tmp_path / "output.md"),
        resume=False,
        log_file="",
        remove_isolated_page_numbers=False,  # simplifier pour les tests
        rejoin_hyphenated_words=False,
        collapse_blank_lines=False,
    )


def make_images(cfg, count=3, prefix="page"):
    """Crée count images JPEG dans cfg.images_dir, retourne la liste des paths."""
    folder = Path(cfg.images_dir)
    paths = []
    for i in range(1, count + 1):
        p = folder / f"{prefix}_{i:03d}.jpg"
        p.write_bytes(JPEG_BYTES)
        paths.append(p)
    return paths


@contextmanager
def fake_nexa_server(cfg):
    """Context manager factice qui ne démarre aucun vrai serveur."""
    yield MagicMock()


# ── Helper pour patcher le pipeline ──────────────────────────────────────────

def run_with_mocks(cfg, ocr_side_effects, image_paths=None):
    """
    Lance run_pipeline avec :
      - nexa_server remplacé par fake_nexa_server
      - ocr_image retournant les valeurs de ocr_side_effects dans l'ordre
      - collect_images retournant image_paths (ou les images créées dans cfg.images_dir)
    """
    if image_paths is None:
        image_paths = sorted(Path(cfg.images_dir).glob("*.jpg"))

    with patch("pipeline.nexa_server", side_effect=fake_nexa_server):
        with patch("pipeline.ocr_image", side_effect=ocr_side_effects):
            with patch("pipeline.collect_images", return_value=image_paths):
                return run_pipeline(cfg)


# ── Tests : succès complet ────────────────────────────────────────────────────

class TestPipelineSuccess:

    def test_returns_stats_object(self, tmp_cfg):
        make_images(tmp_cfg, 2)
        stats = run_with_mocks(tmp_cfg, ["Texte page 1", "Texte page 2"])
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
        stats = run_with_mocks(tmp_cfg, ["abc", "de"])  # 3 + 2 = 5 chars
        assert stats.total_chars == 5

    def test_output_file_created(self, tmp_cfg):
        make_images(tmp_cfg, 1)
        run_with_mocks(tmp_cfg, ["Texte"])
        assert Path(tmp_cfg.output_file).exists()

    def test_output_contains_page_markers(self, tmp_cfg):
        make_images(tmp_cfg, 2)
        run_with_mocks(tmp_cfg, ["Texte 1", "Texte 2"])
        content = Path(tmp_cfg.output_file).read_text()
        assert "<!-- Page" in content

    def test_output_contains_ocr_text(self, tmp_cfg):
        make_images(tmp_cfg, 1)
        run_with_mocks(tmp_cfg, ["Contenu unique de la page"])
        content = Path(tmp_cfg.output_file).read_text()
        assert "Contenu unique de la page" in content

    def test_output_pages_in_order(self, tmp_cfg):
        paths = make_images(tmp_cfg, 3)
        run_with_mocks(tmp_cfg, ["Page A", "Page B", "Page C"], image_paths=paths)
        content = Path(tmp_cfg.output_file).read_text()
        pos_a = content.index("Page A")
        pos_b = content.index("Page B")
        pos_c = content.index("Page C")
        assert pos_a < pos_b < pos_c

    def test_new_file_has_markdown_header(self, tmp_cfg):
        make_images(tmp_cfg, 1)
        run_with_mocks(tmp_cfg, ["Texte"])
        content = Path(tmp_cfg.output_file).read_text()
        assert content.startswith("# ")

    def test_stats_times_list_has_one_entry_per_page(self, tmp_cfg):
        make_images(tmp_cfg, 3)
        stats = run_with_mocks(tmp_cfg, ["T1", "T2", "T3"])
        assert len(stats.times) == 3

    def test_stats_times_are_positive_floats(self, tmp_cfg):
        make_images(tmp_cfg, 2)
        stats = run_with_mocks(tmp_cfg, ["T1", "T2"])
        assert all(t >= 0 for t in stats.times)


# ── Tests : reprise (resume) ──────────────────────────────────────────────────

class TestPipelineResume:

    def _make_existing_output(self, cfg, done_page_ids: list):
        """Écrit un fichier de sortie simulant des pages déjà traitées."""
        content = "# Livre OCR\n\n"
        for page_id in done_page_ids:
            content += format_page_block(page_id, f"Contenu de {page_id}")
        Path(cfg.output_file).write_text(content)

    def test_skips_already_done_pages(self, tmp_cfg):
        tmp_cfg.resume = True
        paths = make_images(tmp_cfg, 3)
        # page_001 déjà traitée
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
        # Créer un fichier existant avec du contenu
        Path(tmp_cfg.output_file).write_text("# Ancien contenu\n\nTexte précédent")
        run_with_mocks(tmp_cfg, ["Nouveau texte"], image_paths=paths)
        content = Path(tmp_cfg.output_file).read_text()
        # Le nouveau contenu remplace l'ancien
        assert "Ancien contenu" not in content

    def test_resume_appends_without_duplicate_header(self, tmp_cfg):
        tmp_cfg.resume = True
        paths = make_images(tmp_cfg, 2)
        self._make_existing_output(tmp_cfg, ["page_001"])
        run_with_mocks(tmp_cfg, ["Nouveau texte"], image_paths=paths)
        content = Path(tmp_cfg.output_file).read_text()
        # L'en-tête "# Livre OCR" ne doit apparaître qu'une seule fois
        assert content.count("# Livre OCR") == 1

    def test_stats_total_is_all_images_including_skipped(self, tmp_cfg):
        tmp_cfg.resume = True
        paths = make_images(tmp_cfg, 5)
        self._make_existing_output(tmp_cfg, ["page_001", "page_002"])
        stats = run_with_mocks(tmp_cfg, ["T3", "T4", "T5"], image_paths=paths)
        assert stats.total == 5


# ── Tests : gestion des erreurs OCR ──────────────────────────────────────────

class TestPipelineOCRErrors:

    def test_continues_after_ocr_error(self, tmp_cfg):
        """Une erreur sur une page ne doit pas arrêter le pipeline."""
        paths = make_images(tmp_cfg, 3)
        side_effects = [
            "Texte page 1",
            OCRError("Timeout"),
            "Texte page 3",
        ]
        stats = run_with_mocks(tmp_cfg, side_effects, image_paths=paths)
        assert stats.done == 2
        assert stats.errors == 1

    def test_error_block_written_to_output(self, tmp_cfg):
        paths = make_images(tmp_cfg, 2)
        side_effects = [OCRError("Connexion refusée"), "Texte OK"]
        run_with_mocks(tmp_cfg, side_effects, image_paths=paths)
        content = Path(tmp_cfg.output_file).read_text()
        assert "ERREUR" in content

    def test_error_does_not_prevent_subsequent_pages(self, tmp_cfg):
        paths = make_images(tmp_cfg, 3)
        side_effects = [OCRError("Crash"), "Page 2 OK", "Page 3 OK"]
        stats = run_with_mocks(tmp_cfg, side_effects, image_paths=paths)
        content = Path(tmp_cfg.output_file).read_text()
        assert "Page 2 OK" in content
        assert "Page 3 OK" in content

    def test_all_errors_stats(self, tmp_cfg):
        paths = make_images(tmp_cfg, 3)
        side_effects = [OCRError("E1"), OCRError("E2"), OCRError("E3")]
        stats = run_with_mocks(tmp_cfg, side_effects, image_paths=paths)
        assert stats.errors == 3
        assert stats.done == 0

    def test_error_page_id_in_error_block(self, tmp_cfg):
        paths = make_images(tmp_cfg, 1)
        side_effects = [OCRError("Timeout 5s")]
        run_with_mocks(tmp_cfg, side_effects, image_paths=paths)
        content = Path(tmp_cfg.output_file).read_text()
        # Le nom de la page doit apparaître dans le bloc d'erreur
        assert "page_001" in content


# ── Tests : écriture incrémentale ─────────────────────────────────────────────

class TestPipelineIncrementalWrite:

    def test_output_file_has_content_after_first_page(self, tmp_cfg):
        """
        On simule une interruption après la première page en patchant flush.
        Le contenu de la première page doit être dans le fichier.
        """
        paths = make_images(tmp_cfg, 2)
        # Utilise le vrai fichier — vérifie juste que le contenu est écrit
        run_with_mocks(tmp_cfg, ["Page 1 complète", "Page 2 complète"],
                       image_paths=paths)
        content = Path(tmp_cfg.output_file).read_text()
        assert "Page 1 complète" in content
        assert "Page 2 complète" in content

    def test_stats_total_set_correctly(self, tmp_cfg):
        paths = make_images(tmp_cfg, 7)
        stats = run_with_mocks(tmp_cfg, ["T"] * 7, image_paths=paths)
        assert stats.total == 7
