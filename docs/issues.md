# Issues — travaux en cours

## Version paddle (actuelle)

### Intégration PaddleOCR dans la pipeline principale

**Points à traiter :**
1. **Gestion des crops figures** — PaddleOCR sauvegarde les crops dans `imgs/` relatif au `save_path`. À intégrer dans la pipeline de sortie.

---

## Prochaines versions

### Feature 2 — Support PDF multi-pages

Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.
