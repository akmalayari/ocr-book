# Issues — travaux en cours

## Features

### 1. Détection automatique des pages avec images non textuelles
Appliquer un mode OCR différent selon le contenu de la page.

**Comportement confirmé des prompts (tests sur pages 1–5) :**
- `"plain"` — retourne le texte selon la trame du document. Usage principal. Boucle sur les pages denses (tableaux, texte serré).
- `"layout"` — idem `"plain"` mais ajoute les grounding boxes des blocs de texte (`<|ref|>...<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>`). Boucle sur les balises `<tr>`/`<td>` dans les pages avec tableaux.
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
Le modèle commet des erreurs de transcription même après binarisation.

**Résultats des tests de quantization (2026-04-01) :**
- Q8_0 → BF16 : légère amélioration (ex: "l'age" correctement transcrit vs "Page"), mais précision toujours imparfaite. F16 probablement équivalent à BF16.
- BF16 : ~60s/page vs ~20s/page en Q8_0. Rapport qualité/vitesse défavorable.

**Pistes restantes :**
- Prompt plus directif (ex: `"Transcribe the text exactly as it appears."` ou prompt en français).
- Qualité des photos sources — hors scope logiciel.
- Modèle alternatif (Qwen2-VL, etc.) potentiellement plus précis sur du français dense.

### 2. Boucles de génération sur pages denses ou avec tableaux
Le modèle entre en boucle (génération infinie répétitive) sur certaines pages.

**Résultats avec `plain` + binarize seul :** page_2, page_3, page_4 bouclent.

Tous les tests ci-dessous utilisent GaussianBlur(5,5) + binarize_adaptive.

**`plain` :**
- page_1 : succès
- page_2 : échec (boucle)
- page_3 : succès, y compris le tableau textuel
- page_4 : mitigé — pas de boucle, retranscription du tableau numérique incorrecte
- page_5 : échec — génération d'un tableau infini de chiffres (probablement axe Y du graphique)

**`layout` :**
- page_1 : pareil que `plain`
- page_2 : succès — plus de boucle
- page_3 : pareil que `plain`
- page_4 : mitigé — tableau détecté, noms des colonnes approximatifs, cases vides, texte sous le tableau oublié
- page_5 : échec — boucle sur balises `<tr>`/`<td>`

`repetition_penalty` testé, sans effet sur les boucles.

**Conclusion :** aucun prompt ne couvre tous les cas. page_4/5 (tableau numérique, graphique) sont des candidats pour la détection automatique (Feature 1) + mode `"describe"`/`"parse"`.

**Pistes restantes :**
- Limiter `max_tokens` pour couper la génération avant la boucle.
- Prompt adaptatif selon le contenu (Feature 1).

## Architecture

Tout est ok.

## Style

### 1. Padding du renommage non documenté
`images.py:71` — `width = len(str(len(images)))` donne un padding minimal basé sur
le nombre d'images au moment du renommage. Si on ajoute des images plus tard et qu'on
re-renomme, les anciens fichiers (`page_01.jpg`) et les nouveaux (`page_001.jpg`)
seront incohérents. À documenter ou à fixer avec un minimum de 3 chiffres.
