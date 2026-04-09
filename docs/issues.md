# Issues — travaux en cours

## Version paddle (actuelle)

OK

---

## Prochaines versions

### Feature 1 - Indication de pages

Actuellement les images traitées sont indiquées avec "<!-- Page page_5 -->". Mais une image n'est pas egale a une page. Il est peut etre possible de recuperer le numeros de la page avec `markdown_ignore_labels` (voir `docs\paddleocr\config.md`)

### Feature 2 — Support PDF multi-pages

Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.

