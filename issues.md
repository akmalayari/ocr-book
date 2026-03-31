# Issues — travaux en cours

## Features

### 1. Détection automatique des pages avec images non textuelles
Appliquer un mode OCR différent selon le contenu de la page.

**Comportement confirmé des prompts (tests sur pages 1–5) :**
- `"plain"` — retourne le texte selon la trame du document. Usage principal.
- `"layout"` — idem `"plain"` mais ajoute les grounding boxes des blocs de texte (`<|ref|>...<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>`).
- `"describe"` — décrit l'image en anglais (indépendamment de la langue du document).
- `"parse"` — analyse fine des éléments de l'image en anglais.
- `"classic"` (supprimé, non pertinent) — retournait chaque phrase accompagnée de sa grounding box ; trop verbeux, dépasse systématiquement `n_ctx`.

**Approches à tester (par ordre de priorité) :**

1. **Heuristique OpenCV (option retenue pour démarrer)** — ratio contours horizontaux / surface. Si densité faible → page image → mode `"describe"`. Configurable via un seuil dans `config.py`. Problème ouvert : pages mixtes (voir point 2).

2. **Pages mixtes** — deux sous-pistes :
   - Le mode `"layout"` produit des balises `<|ref|>texte<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>`. Exploitable pour localiser une illustration : extraire sa bbox, recadrer l'image, appliquer `"describe"` sur la zone. Nécessite un parser de ces balises + nettoyage en post-traitement pour le texte normal.
   - Utiliser le mode `"parse"` (`Parse the figure.`) — à tester : couvre-t-il les illustrations non-techniques (photos, dessins) ou seulement les graphiques/diagrammes ?

3. **Prompt adaptatif** — prompt unique couvrant texte et image, ex. `"If this is a figure or illustration, describe it. Otherwise, Free OCR."`. Non documenté par DeepSeek-OCR, comportement à tester.

**Langue des descriptions :** `"describe"` et `"parse"` répondent en anglais indépendamment de la langue du document. À tester : un prompt explicite comme `"Describe this image in French."` ou `"Describe this image in the same language as the document."`.

### 2. Rename images in order of creation date
rename_images par date de création: du plus vieux au plus récent. Permet de reconstruire le livre dans l'ordre.

## Bugs actifs

### 1. Précision OCR imparfaite malgré binarize_adaptive
Le modèle commet des erreurs de transcription même après binarisation. Cause non
identifiée — peut être liée au modèle lui-même (quantization Q8_0), au prompt,
ou aux paramètres de génération.

**Pistes :** Tester d'autres valeurs de `blockSize`/`C` pour la binarisation.
Tester le prompt `"plain"` vs `"layout"` sur les mêmes pages pour comparer.
Modifier les paramètres de génération (`GenerationConfig`).

### 2. Context length exceeded sur pages denses
Plusieurs pages échouent avec `[-200004] Context length exceeded` avec `n_ctx=8192` :
pages 2, 3, 4, 5 en mode `"plain"` ; pages 3, 4 en mode `"layout"`.
Cause : pages denses en texte ou en tableaux dépassent la fenêtre de contexte.

**Pistes :**
- Augmenter `n_ctx` (16384 ou 32768) — coût mémoire à évaluer.
- Découper les images hautes en deux avant OCR.
- Réduire `max_tokens` pour libérer de la place au prompt+image.

### 3. `"layout"` sur page_5 retourne une bbox globale
`page_5.jpg` (texte + tableau + graphe) produit `<|ref|>image<|/ref|><|det|>[[0, 0, 999, 997]]<|/det|>` — le modèle détecte l'image entière comme un seul bloc au lieu d'en décomposer les éléments. Probablement dû à la qualité/flou de la photo. À re-tester avec une image nette.

### 4. `"describe"` sur page floue hallucine la langue
`page_5.jpg` (floue) : `"describe"` évoque du bengali alors que le document est en français. Indique que la qualité d'image affecte fortement ce mode.

## Architecture

### 1. `PROMPTS` comme constante de module
`config.py` — `PROMPTS` est un dict statique dans un `dataclass` avec `field(default_factory=...)`.
Chaque instance crée son propre dict alors que la valeur ne change jamais.
Mieux placé comme constante au niveau module ou comme `ClassVar`.

## Style

### 1. Padding du renommage non documenté
`images.py:71` — `width = len(str(len(images)))` donne un padding minimal basé sur
le nombre d'images au moment du renommage. Si on ajoute des images plus tard et qu'on
re-renomme, les anciens fichiers (`page_01.jpg`) et les nouveaux (`page_001.jpg`)
seront incohérents. À documenter ou à fixer avec un minimum de 3 chiffres.
