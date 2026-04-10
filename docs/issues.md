# Issues — travaux en cours

## Version refined (actuelle)

### OCR (objectif final)
Quand cette version est complétée: lancer  `python src/main.py --images "C:\path\to\Documents\Livres\Leçons d'introduction à la sociologie\livre" --no-resume --mode obsidian` pour OCRiser le livre.

## Prochaines versions

### Feature 2 — Support PDF multi-pages

Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.

### Fragilité 1 (en suspens) - Liberation de ressources 

Les ressources ne sont pas toujours libérées lors de la fermeture du serveur: "Background thread did not terminate in time. Some resources may not be properly released." 2Go qui restent éternellement en Ram. 