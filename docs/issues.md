# Issues — travaux en cours

## Version paddle (actuelle)

### Validation du patch parallèle sur images de test

Le patch `apply_paddlex_patch_parallel.py` (pool global, np=3) doit être testé sur l'ensemble des images de référence (page_1 à page_9) pour vérifier l'absence de crash et la qualité de l'output.

---

## Prochaines versions

### Feature 2 — Support PDF multi-pages

Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.
