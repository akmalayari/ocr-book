# Issues — travaux en cours

## Version refined (actuelle)

### Feature 1 - Indication de pages

Actuellement les images traitées sont indiquées avec "<!-- Page page_5 -->". Mais une image n'est pas egale a une page. Il est peut etre possible de recuperer le numeros de la page avec `markdown_ignore_labels` (voir `docs\paddleocr\config.md`)

### Fragilité 1 - Liberation de ressources

Les ressources ne sont pas toujours libérées lors de la fermeture du serveur: "Background thread did not terminate in time. Some resources may not be properly released." 2Go qui restent éternellement en Ram. 

### Amelioration 1 - Bypass check pour gagner des secondes

Examiner ce message et déterminer s'il est possible de gagner quelques secondes en contournant le check. Voici le message en debut de run: "Checking connectivity to the model hosters, this may take a while. To bypass this check, set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` to `True`."

## Prochaines versions

### Feature 2 — Support PDF multi-pages

Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.

