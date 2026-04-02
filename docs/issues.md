# Issues — travaux en cours

## Features

### 1. Détection automatique des pages avec images non textuelles
Appliquer un mode OCR différent selon le contenu de la page.

**Comportement confirmé des prompts (tests sur pages 1–5) :**
- `"plain"` — retourne le texte selon la trame du document. Usage principal. Boucle sur les pages denses (tableaux, texte serré).
- `"layout"` — idem `"plain"` mais ajoute les grounding boxes des blocs de texte (`<|ref|>...<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>`). Boucle sur les balises `<tr>`/`<td>` dans les pages avec tableaux.
- `"describe"` — décrit l'image en anglais (indépendamment de la langue du document).
- `"parse"` — analyse fine des éléments de l'image en anglais.

**Approches à tester (par ordre de priorité) :**

1. **Heuristique OpenCV (option retenue pour démarrer)** — ratio contours horizontaux / surface. Si densité faible → page image → mode `"describe"`. Configurable via un seuil dans `config.py`. Problème ouvert : pages mixtes (voir point 2).

2. **Pages mixtes** — deux sous-pistes :
   - Le mode `"layout"` produit des balises `<|ref|>texte<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>`. Exploitable pour localiser une illustration : extraire sa bbox, recadrer l'image, appliquer `"describe"` sur la zone. Nécessite un parser de ces balises + nettoyage en post-traitement pour le texte normal.
   - Utiliser le mode `"parse"` (`Parse the figure.`) — à tester : couvre-t-il les illustrations non-techniques (photos, dessins) ou seulement les graphiques/diagrammes ?

3. **`"OCR only the text, ignore any figures."`** (testé, résultat partiel) — ignore figures et tableaux, mais skip aussi certaines colonnes de texte. Piste : utiliser la longueur du résultat comme signal de détection (résultat court/vide → page dominée par une figure → repasser en `"describe"`). À combiner avec l'heuristique OpenCV plutôt qu'en remplacement.

4. **Prompts avec prefix grounding** — non testés : `"<|grounding|>Transcribe only text blocks."`, `<|grounding|>"OCR only the text, ignore any figures."`. Le prefix grounding permet au modèle de localiser les blocs — potentiellement plus précis sur les colonnes.

**Langue des descriptions :** le modèle ignore systématiquement les instructions de langue (testé avec 5 formulations différentes). `"describe"` et `"parse"` répondent toujours en anglais. 

### 2. Support PDF multi-pages
Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.

## Bugs actifs

### 1. Précision OCR imparfaite malgré binarize_adaptive
Le modèle commet des erreurs de transcription même après binarisation.

**Résultats des tests de quantization (2026-04-01) :**
- Q8_0 → BF16 : amélioration (ex: "l'age" correctement transcrit vs "Page"), mais précision toujours imparfaite. F16 commet plus d'erreurs que BF16.
- BF16 sur page_5 : hallucine toujours sur les données du graphe, mais parvient à retranscrire approximativement le tableau et quelques paragraphes — finit par boucler. Q8_0 bouclait immédiatement. Impact réel de la quantization sur les pages difficiles.
- BF16 : ~50s/page vs ~20s/page en Q8_0. Rapport qualité/vitesse défavorable mais vaut le coup pour le gain de précision.

**Pistes restantes :**
- Prompt plus directif (ex: `"Transcribe the text exactly as it appears."` ou prompt en français).
- Qualité des photos sources — hors scope logiciel.
- Modèle alternatif (Qwen2-VL, etc.) potentiellement plus précis sur du français dense.
- Amélioration du préprocessing pour les images floues (voir Bug 3).

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
- Prompt adaptatif selon le contenu (Feature 1).

### 3. Courbure de page
Courbure due à la reliure — déforme les lignes de texte géométriquement.

Provoque également des erreurs sur les mots coupés en fin de ligne (`distri-\nbution` → `"des distinctions biutique négligeant"` au lieu de `"une distribution inégale"`) : la déformation géométrique au niveau du pli perturbe la reconnaissance de la césure.

**Pistes à tester (par ordre de priorité) :**
1. **Lignes de texte + fit polynomial + `cv2.remap`** — dilatation horizontale `(30,1)` pour regrouper les caractères en blobs de ligne, fit polynomial sur les centroïdes, remap inverse. Pure OpenCV, aucune dépendance. À appliquer sur chaque moitié de l'image séparément (double page). Approche retenue en priorité.
2. **DocTr** — transformer léger de déwarpage de documents, tourne sur CPU. Dépendance PyTorch/ONNX (~200 MB). À tester si l'approche OpenCV est insuffisante.
3. **DewarpNet / DocUNet** — modèle neural, résultats solides sur courbures prononcées. GPU requis, dépendance lourde. Dernier recours.

### 4. Images floues
Unsharp Mask (standard et ordre inversé) testé — aucune amélioration, ajoute des granulés sombres sur certaines configs. Les approches amplificatrices de hautes fréquences sont inefficaces sur du flou de mise au point.

**Pistes à tester (par ordre de priorité) :**
1. **Bilateral filter** — lisse le bruit en préservant les contours, alternative à GaussianBlur avant binarisation.
2. **Sauvola binarization** (`scikit-image`) — conçue pour les documents dégradés, tient compte de la variance locale. Potentiellement meilleure qu'`adaptiveThreshold` sur texte mal contrasté.
3. **Real-ESRGAN** — super-résolution x2/x4 par modèle neural, récupère des caractères illisibles. ~2–5s/image sur GPU. Dépendance lourde (`basicsr`).

## Architecture

OK

## Style

OK