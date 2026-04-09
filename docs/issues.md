# Issues — travaux en cours

## Version refined (actuelle)

### Fragilité 1 - Liberation de ressources

Les ressources ne sont pas toujours libérées lors de la fermeture du serveur: "Background thread did not terminate in time. Some resources may not be properly released." 2Go qui restent éternellement en Ram. 

### Amelioration 1 - Bypass check pour gagner des secondes

Examiner ce message et déterminer s'il est possible de gagner quelques secondes en contournant le check. Voici le message en debut de run: "Checking connectivity to the model hosters, this may take a while. To bypass this check, set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` to `True`."

## Prochaines versions

### Feature 2 — Support PDF multi-pages

Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.

