# Issues — travaux en cours

## Version refined (actuelle)

OK

## Prochaines versions

### Feature 1 — Support PDF multi-pages

Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.

### Feature 2 - Postprocess du LaTeX

Rendre robuste l'OCR des documents académiques contenant beaucoup d''equations LateX. 

### Fragilité 1 (en suspens) - Liberation de ressources 

Les ressources ne sont pas toujours libérées lors de la fermeture du serveur: "Background thread did not terminate in time. Some resources may not be properly released."