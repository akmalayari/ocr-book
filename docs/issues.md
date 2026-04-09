# Issues — travaux en cours

## Version paddle (actuelle)

### Parallelisation pour gain de vitesse
- Paralleliser le traitement des pages en lançant deux serveurs llama-server concurrents.

---

## Prochaines versions

### Feature 2 — Support PDF multi-pages

Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.
