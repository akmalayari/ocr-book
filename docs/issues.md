# Issues — travaux en cours

## Features

### 1. Détection automatique des pages avec images non textuelles
Appliquer un mode OCR différent selon le contenu de la page.

**Comportement confirmé des prompts (tests sur pages 1–6) :**
- `"plain"` — retourne le texte selon la trame du document. Usage principal. Boucle sur les pages denses. Sur page_6 (texte + tableau + graphique, image nette) : texte et tableau transcrits correctement, graphique silencieusement ignoré.
- `"layout"` — idem `"plain"` mais ajoute les grounding boxes. Classe les régions en `text`, `sub_title`, `table`, `image`, `image_caption`. Sur page_6 : graphique correctement classé `image`, contenu vide — le modèle ne génère rien pour les régions `image`.
- `"describe"` — décrit l'image en anglais (indépendamment de la langue du document).
- `"parse"` — analyse fine des éléments de l'image en anglais.

**Approches à tester (par ordre de priorité) :**

1. **Pipeline deux passes sur les pages mixtes (option prioritaire)** — le mode `layout` produit des bboxes avec label `image`. Approche : passe 1 `layout` pour détecter les régions `image` + extraire leur bbox ; passe 2 `parse` sur le crop correspondant. Nécessite un parser de grounding boxes (déjà dans `viz_boxes.py`) + crop PIL + second appel VLM. Permettrait de gérer texte et figure sur la même page.

2. **Heuristique OpenCV** — ratio contours horizontaux / surface. Si densité faible → page image → mode `"describe"`. Configurable via un seuil dans `config.py`. Plus simple mais ne gère pas les pages mixtes.

3. **`"OCR only the text, ignore any figures."`** (testé, résultat partiel) — ignore figures et tableaux mais skip aussi certaines colonnes de texte. À combiner avec les approches ci-dessus plutôt qu'en remplacement.

4. **Prompts avec prefix grounding** — non testés : `"<|grounding|>Transcribe only text blocks."`. Le prefix grounding pourrait être plus précis sur les colonnes.

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
- page_6 (nette, même contenu que page_5) : succès partiel — texte et tableau transcrits correctement, graphique ignoré silencieusement

**`layout` :**
- page_1 : pareil que `plain`
- page_2 : succès — plus de boucle
- page_3 : pareil que `plain`
- page_4 : mitigé — tableau détecté, noms des colonnes approximatifs, cases vides, texte sous le tableau oublié
- page_5 : échec — boucle sur balises `<tr>`/`<td>` (graphique mal classé `table` à cause de l'image floue)
- page_6 : succès partiel — graphique correctement classé `image` avec bbox, mais contenu vide

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

## Améliorations

### 1. Performance — BF16 à ~50s/image

BF16 est 2.5× plus lent que Q8_0 (~20s). Le GPU est bien utilisé (confirmé via Task Manager). Le coût est purement inference.

**Pistes à tester (par ordre de priorité) :**
1. **Réduire la résolution des images** — le visual encoder scale avec le nombre de pixels. Redimensionner à ~1200px de large avant d'envoyer au VLM (dans `preprocess.py`). Hypothèse : gain de 30–50% si l'encodage visuel domine le temps total.
2. **Profiler le vrai bottleneck** — chronométrer séparément l'encodage visuel et le décodage texte pour savoir où vont les 50s. `vlm.generate()` fait les deux d'un coup.
3. **Changer de wrapper Python (llama-cpp-python)** — secondaire. Même DLL llama.cpp sous le capot, peu probable d'accélérer. À tester si les pistes 1–2 sont insuffisantes.

## Architecture

OK

## Style

OK