"""
test_ocr_client.py — Tests unitaires pour ocr_client.py

Stratégie :
  - vlm.generate est TOUJOURS mocké : aucun appel VLM réel.
  - preprocess_image est mockée pour éviter les appels cv2.
  - On teste _is_looping exhaustivement (stop words, seuil, fenêtre).
  - On teste ocr_image : succès, erreurs, détection de boucle.

Couvre :
  - OCRError            : hiérarchie
  - _STOPWORDS_FR       : type, contenu
  - _is_looping         : fenêtre, seuil, stop words, texte varié
  - ocr_image           : succès, tuple retourné, fichier manquant,
                          exception VLM, résultat vide, arrêt sur boucle
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config
from ocr_client import OCRError, _STOPWORDS_FR, _is_looping, ocr_image

JPEG_BYTES = bytes([0xFF, 0xD8, 0xFF, 0xD9])


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_cfg(**kwargs):
    defaults = dict(
        preprocess_mode="none",
        log_file="",
        loop_check_every=200,
        loop_window_words=50,
        loop_divisor_threshold=0.7,
    )
    defaults.update(kwargs)
    return Config(**defaults)


def make_vlm_mock(tokens: list[str]):
    """VLM dont generate() appelle on_token pour chaque token de la liste."""
    vlm = MagicMock()
    vlm.apply_chat_template.return_value = "formatted_prompt"

    def fake_generate(formatted, config, on_token):
        for token in tokens:
            if not on_token(token):
                break

    vlm.generate.side_effect = fake_generate
    return vlm


@pytest.fixture
def jpeg_file(tmp_path):
    p = tmp_path / "page_001.jpg"
    p.write_bytes(JPEG_BYTES)
    return p


# ── OCRError ──────────────────────────────────────────────────────────────────

class TestOCRError:

    def test_is_runtime_error_subclass(self):
        assert issubclass(OCRError, RuntimeError)

    def test_message_preserved(self):
        assert "image introuvable" in str(OCRError("image introuvable"))

    def test_can_be_caught_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise OCRError("test")


# ── _STOPWORDS_FR ─────────────────────────────────────────────────────────────

class TestStopwordsFr:

    def test_is_frozenset(self):
        assert isinstance(_STOPWORDS_FR, frozenset)

    def test_not_empty(self):
        assert len(_STOPWORDS_FR) > 0

    def test_contains_determiners(self):
        for word in ("le", "la", "les", "un", "une"):
            assert word in _STOPWORDS_FR, f"'{word}' devrait être un stop word"

    def test_contains_prepositions(self):
        for word in ("de", "du", "des", "en", "dans", "sur", "pour"):
            assert word in _STOPWORDS_FR, f"'{word}' devrait être un stop word"

    def test_contains_conjunctions(self):
        for word in ("et", "ou", "que", "si"):
            assert word in _STOPWORDS_FR, f"'{word}' devrait être un stop word"

    def test_does_not_contain_content_words(self):
        for word in ("tableau", "chapitre", "résultat", "figure", "analyse"):
            assert word not in _STOPWORDS_FR, f"'{word}' ne devrait pas être un stop word"


# ── _is_looping ───────────────────────────────────────────────────────────────

class TestIsLooping:

    def test_returns_false_on_empty_string(self):
        assert _is_looping("", window_words=10, threshold=0.7) is False

    def test_returns_false_when_window_not_filled(self):
        # Seulement 5 mots de contenu, fenêtre = 20
        text = "tableau résultat analyse figure graphique"
        assert _is_looping(text, window_words=20, threshold=0.7) is False

    def test_detects_repetitive_content_words(self):
        # "tableau" répété 60 fois → fenêtre de 20 remplie d'un seul mot
        text = "tableau " * 60
        assert _is_looping(text, window_words=20, threshold=0.5) is True

    def test_no_false_positive_on_varied_text(self):
        text = (
            "Le premier chapitre présente les résultats de l'analyse statistique. "
            "La deuxième section discute des implications pour la recherche. "
            "En conclusion, les données montrent une corrélation significative."
        )
        assert _is_looping(text, window_words=20, threshold=0.7) is False

    def test_stop_words_only_do_not_trigger_loop(self):
        # Uniquement des stop words → pas assez de mots de contenu pour remplir la fenêtre
        text = "le la les de et en à au aux que qui est il elle " * 10
        assert _is_looping(text, window_words=20, threshold=0.5) is False

    def test_threshold_not_reached_no_loop(self):
        # 30 mots tous uniques → aucune répétition → ratio = 0
        words = [f"concept{i}" for i in range(30)]
        text = " ".join(words)
        assert _is_looping(text, window_words=20, threshold=0.5) is False

    def test_threshold_reached_loop_detected(self):
        # Deux mots qui se répètent très fréquemment
        text = "tableau résultat " * 60
        assert _is_looping(text, window_words=20, threshold=0.7) is True

    def test_stop_words_stripped_punctuation(self):
        # Stop words avec ponctuation attachée doivent être exclus
        text = "le, la. les; tableau tableau tableau tableau tableau " * 10
        # "tableau" répété → boucle détectée malgré la ponctuation sur les stop words
        assert _is_looping(text, window_words=10, threshold=0.5) is True

    def test_mixed_content_and_stop_words(self):
        # Mots de contenu répétés entrecoupés de stop words
        text = "le tableau de la figure les tableau de la figure " * 20
        # "tableau" et "figure" sont des mots de contenu répétés
        assert _is_looping(text, window_words=10, threshold=0.5) is True


# ── ocr_image ─────────────────────────────────────────────────────────────────

class TestOcrImageSuccess:

    def test_returns_tuple(self, jpeg_file):
        vlm = make_vlm_mock(["Texte ", "de ", "la ", "page."])
        cfg = make_cfg()
        result = ocr_image(jpeg_file, vlm, cfg)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_string(self, jpeg_file):
        vlm = make_vlm_mock(["Bonjour ", "monde."])
        text, _ = ocr_image(jpeg_file, vlm, cfg=make_cfg())
        assert isinstance(text, str)

    def test_second_element_is_dict_with_latency(self, jpeg_file):
        vlm = make_vlm_mock(["Texte."])
        _, metrics = ocr_image(jpeg_file, vlm, cfg=make_cfg())
        assert isinstance(metrics, dict)
        assert "total_latency" in metrics
        assert isinstance(metrics["total_latency"], float)

    def test_text_concatenated_from_tokens(self, jpeg_file):
        vlm = make_vlm_mock(["Cha", "pitre", " 1"])
        text, _ = ocr_image(jpeg_file, vlm, cfg=make_cfg())
        assert text == "Chapitre 1"

    def test_text_stripped(self, jpeg_file):
        vlm = make_vlm_mock(["  \n\nTexte propre\n\n  "])
        text, _ = ocr_image(jpeg_file, vlm, cfg=make_cfg())
        assert not text.startswith(" ")
        assert not text.endswith(" ")

    def test_accepts_path_object(self, jpeg_file):
        vlm = make_vlm_mock(["Texte"])
        text, _ = ocr_image(jpeg_file, vlm, cfg=make_cfg())
        assert text == "Texte"

    def test_accepts_string_path(self, jpeg_file):
        vlm = make_vlm_mock(["Texte"])
        text, _ = ocr_image(str(jpeg_file), vlm, cfg=make_cfg())
        assert text == "Texte"

    def test_preprocess_none_does_not_call_preprocess(self, jpeg_file):
        vlm = make_vlm_mock(["Texte"])
        cfg = make_cfg(preprocess_mode="none")
        with patch("ocr_client.preprocess_image") as mock_pre:
            ocr_image(jpeg_file, vlm, cfg)
        mock_pre.assert_not_called()

    def test_preprocess_binarize_calls_preprocess(self, jpeg_file):
        vlm = make_vlm_mock(["Texte"])
        cfg = make_cfg(preprocess_mode="binarize")
        fake_preprocessed = jpeg_file  # retourner le même fichier
        with patch("ocr_client.preprocess_image", return_value=fake_preprocessed):
            text, _ = ocr_image(jpeg_file, vlm, cfg)
        assert text == "Texte"


# ── ocr_image : gestion des erreurs ──────────────────────────────────────────

class TestOcrImageErrors:

    def test_raises_ocr_error_if_file_not_found(self, tmp_path):
        vlm = MagicMock()
        cfg = make_cfg()
        with pytest.raises(OCRError, match="introuvable"):
            ocr_image(tmp_path / "inexistant.jpg", vlm, cfg)

    def test_raises_ocr_error_on_vlm_exception(self, jpeg_file):
        vlm = MagicMock()
        vlm.apply_chat_template.return_value = "formatted"
        vlm.generate.side_effect = RuntimeError("VLM crash")
        with pytest.raises(OCRError):
            ocr_image(jpeg_file, vlm, cfg=make_cfg())

    def test_raises_ocr_error_on_empty_result(self, jpeg_file):
        vlm = make_vlm_mock(["   \n\n   "])
        with pytest.raises(OCRError, match="vide"):
            ocr_image(jpeg_file, vlm, cfg=make_cfg())

    def test_raises_ocr_error_on_no_tokens(self, jpeg_file):
        vlm = make_vlm_mock([])  # generate() ne produit rien
        with pytest.raises(OCRError, match="vide"):
            ocr_image(jpeg_file, vlm, cfg=make_cfg())

    def test_keyboard_interrupt_not_caught(self, jpeg_file):
        vlm = MagicMock()
        vlm.apply_chat_template.return_value = "formatted"
        vlm.generate.side_effect = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            ocr_image(jpeg_file, vlm, cfg=make_cfg())


# ── ocr_image : détection de boucle ──────────────────────────────────────────

class TestOcrImageLoopDetection:

    def test_loop_stops_generation(self, jpeg_file):
        # 500 tokens répétitifs → détection à 200 tokens
        repeated = "tableau " * 500
        tokens = list(repeated.split(" "))
        vlm = make_vlm_mock(tokens)
        cfg = make_cfg(
            loop_check_every=200,
            loop_window_words=50,
            loop_divisor_threshold=0.3,
        )
        text, _ = ocr_image(jpeg_file, vlm, cfg)
        # La génération s'est arrêtée avant d'avoir consommé les 500 tokens
        assert len(text.split()) < 450

    def test_no_false_positive_on_normal_text(self, jpeg_file):
        # 360 tokens tous uniques → aucune répétition → pas d'arrêt
        tokens = [f"mot{i} " for i in range(360)]
        vlm = make_vlm_mock(tokens)
        cfg = make_cfg(
            loop_check_every=200,
            loop_window_words=50,
            loop_divisor_threshold=0.7,
        )
        text, _ = ocr_image(jpeg_file, vlm, cfg)
        assert len(text.split()) >= 350

    def test_loop_check_disabled_when_check_every_exceeds_tokens(self, jpeg_file):
        # check_every > nb de tokens → jamais vérifié, pas d'arrêt
        tokens = ["boucle "] * 100
        vlm = make_vlm_mock(tokens)
        cfg = make_cfg(loop_check_every=1000)
        text, _ = ocr_image(jpeg_file, vlm, cfg)
        assert len(text.split()) == 100
