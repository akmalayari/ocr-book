# Issues — à traiter plus tard

## Features

rename_images par date de création: du plus vieux au plus récent. Permet de reconstruire le livre dans l'ordre.

## Bugs actifs

### Output tronqué avec marqueur `répéte`
`pipeline.py` / `ocr_client.py` — Le modèle interrompt sa transcription en cours de page
et émet un résumé en bullet points précédé du mot-clé `**répéte**`. Constaté sur des pages
denses (output/livre.md, page_2, lignes ~39-45).

**Cause probable :** Le modèle atteint la limite de son contexte ou sa fenêtre d'attention
en cours de génération. Il bascule sur un mode "résumé/répétition" au lieu de s'arrêter
proprement. Peut aussi être lié à la qualité de la binarisation sur certaines zones (moins probable).

**Pistes :**
- Tester `n_ctx` plus élevé (8192) dans `Config` (à tester en priorité).
- Vérifier si le texte tronqué se produit systématiquement sur les mêmes pages (densité
  de texte, mise en page complexe) ou de façon aléatoire.
- Tester un pré-traitement alternatif sur les pages problématiques.
- Ajouter détection du marqueur `répéte` dans `postprocess.py` pour lever une alerte.

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
