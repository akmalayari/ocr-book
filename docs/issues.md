# Issues — travaux en cours

## Version paddle (actuelle)

### Intégration PaddleOCR dans la pipeline principale

PaddleOCR VL 1.5 retenu (voir `docs/tested.md`). À intégrer en remplacement de DeepSeek-OCR/nexaai.

**Points à traiter :**
1. **Dépendances** — Python 3.10 (conda), paddlepaddle CPU, paddleocr depuis dépôt main, patch paddlex. Décider : `environment.yml` conda intégré au projet, ou env séparé documenté.
2. **Nouveau `ocr_client.py`** (ou réécriture) — remplacer nexaai.VLM par llama-server subprocess + PaddleOCRVL.
3. **Postprocess** — `postprocess.py` câblé sur les tokens DeepSeek (`<|ref|>`, `<|det|>`). À réécrire pour le format HTML PaddleOCR (pas de tokens grounding, tables en `<table>`, figures en `<img>`).
4. **Détection de boucle** — `_is_looping` dans `ocr_client.py` devient sans objet (PaddleOCR ne boucle pas). À évaluer si on garde pour sécurité ou supprime.
5. **Gestion des crops figures** — PaddleOCR sauvegarde les crops dans `imgs/` relatif au `save_path`. À intégrer dans la pipeline de sortie.

### Amélioration 3 — Tuning llama-server

Pistes pour réduire le temps de traitement (actuellement 35–40s/image) :
- `--n-gpu-layers` : décharger des couches sur GPU via Vulkan
- `--ctx-size` : réduire si 4096 tokens non nécessaires pour les pages courtes
- `--batch-size` : ajuster pour le throughput

---

## Prochaines versions

### Feature 2 — Support PDF multi-pages

Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.

## Architecture

OK

## Style

OK
