# Issues — à traiter plus tard

## Features

### Détection automatique des pages avec images non textuelles
Appliquer un mode OCR différent selon le contenu de la page.

**Approches à tester (par ordre de priorité) :**

1. **Heuristique OpenCV (option retenue pour démarrer)** — ratio contours horizontaux / surface. Si densité faible → page image → mode `"describe"`. Configurable via un seuil dans `config.py`. Problème ouvert : pages mixtes (voir point 3).

2. **Pages mixtes** — deux sous-pistes :
   - Utiliser `rec` (`<image>\nLocate <|ref|>xxxx<|/ref|> in the image.`) pour détecter le bounding box de l'illustration, puis OCR `"plain"` sur le texte et `"describe"` sur la zone image. Nécessite de comprendre le format de sortie du mode `rec`.
   - Utiliser le mode `"figure"` (`Parse the figure.`) — à tester : couvre-t-il les illustrations non-techniques (photos, dessins) ou seulement les graphiques/diagrammes ?

3. **Prompt adaptatif (option 3)** — prompt unique couvrant texte et image, ex. `"If this is a figure or illustration, describe it. Otherwise, Free OCR."`. Non documenté par DeepSeek-OCR, comportement à tester.

**Prérequis :** tester chaque mode (`plain`, `markdown`, `figure`, `classic`, `describe`) sur des pages représentatives pour cartographier ce que chacun produit et ne produit pas.

### Rename images in order of creation date
rename_images par date de création: du plus vieux au plus récent. Permet de reconstruire le livre dans l'ordre.

## Bugs actifs

### Précision OCR imparfaite malgré binarize_adaptive
Le modèle commet des erreurs de transcription même après binarisation. Cause non
identifiée — peut être liée au modèle lui-même (quantization Q8_0), au prompt,
ou aux paramètres de génération.

**Pistes :** Tester d'autres valeurs de `blockSize`/`C` pour la binarisation.
Tester le prompt `"markdown"` vs `"plain"` sur les mêmes pages pour comparer.
Modifier les paramètres de génération (`GenerationConfig`).

## Architecture

### `PROMPTS` comme constante de module
`config.py` — `PROMPTS` est un dict statique dans un `dataclass` avec `field(default_factory=...)`.
Chaque instance crée son propre dict alors que la valeur ne change jamais.
Mieux placé comme constante au niveau module ou comme `ClassVar`.

## Style

### Padding du renommage non documenté
`images.py:71` — `width = len(str(len(images)))` donne un padding minimal basé sur
le nombre d'images au moment du renommage. Si on ajoute des images plus tard et qu'on
re-renomme, les anciens fichiers (`page_01.jpg`) et les nouveaux (`page_001.jpg`)
seront incohérents. À documenter ou à fixer avec un minimum de 3 chiffres.
