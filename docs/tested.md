# Tested — ce qui a été expérimenté dans le projet

Pages de référence utilisées pour les tests : `page_1` à `page_9`.
- page_1 : texte simple, mauvais éclairage
- page_2 : texte dense, éclairage normal
- page_3 : texte + tableau textuel
- page_4 : texte + tableau numérique
- page_5 : texte + tableau + graphique, image floue
- page_6 : même contenu que page_5, image nette (version de référence pour tests graphe)
- page_7 : début de chapitre, texte seul, image nette
- page_8 : texte seul, image nette
- page_9 : fin de chapitre, texte seul, image nette

---

## Stack d'inférence

### REST API `nexa serve`
**Statut : abandonné.**
Retournait systématiquement HTTP 500 sur les requêtes multimodales sous Windows. Le serveur REST de Nexa ne sait pas gérer le format GGUF + mmproj de DeepSeek-OCR.

### `nexaai.VLM` Python direct
**Statut : retenu.**
Charge les deux fichiers du modèle (GGUF + mmproj) via `nexa_bridge.dll`. Contourne le serveur REST entièrement. Interface : `VlmChatMessage` + `apply_chat_template` + `vlm.generate(GenerationConfig(image_paths=[...]))`.

**Limite :** sur Windows, `nexa_bridge.dll` retourne un `stop_reason` corrompu (byte `0xc0` invalide UTF-8) dans les métadonnées de profiling. Contourné par un monkey-patch dans `src/patch.py` — le texte OCR lui-même est intact.

---

## Quantization du modèle

| Quantization | Vitesse | Qualité |
|---|---|---|
| Q8_0 | ~20s/page | boucle immédiatement sur pages difficiles |
| BF16 | ~50s/page | plus fidèle que F16 sur les passages difficiles |
| F16  | ~50s/page | moins fidèle que BF16 — hallucinations et reformulations sur passages difficiles |

**Statut : BF16 retenu.** Comparaison BF16 vs F16 (2026-04-02, `compare.py` mode sentence, page_1) : BF16 retranscrit fidèlement plusieurs passages là où F16 hallucine ou reformule. BF16 reste imparfait (voir Bug #1 et #5).

---

## Prompts

Testés sur pages 1–6 avec `preprocess=binarize` (`draft/test_prompts.py`).

| Mode | Prompt | Résultat |
|---|---|---|
| `plain` | `"Free OCR."` | texte brut propre sur pages simples, boucle sur pages denses |
| `layout` | `"<\|grounding\|>Convert the document to markdown."` | ajoute grounding boxes, règle la boucle de page_2 mais boucle sur les tableaux (`<tr>`/`<td>`) |
| `describe` | `"Describe this image in detail."` | description en anglais, indépendamment de la langue du document |
| `parse` | `"Parse the figure."` | analyse fine des éléments visuels en anglais |
| `rec` | `"Locate <\|ref\|>{target}<\|/ref\|> in the image."` | retourne la/les bbox correspondant au target. Voir section dédiée ci-dessous. |
| `classic` | (supprimé) | une grounding box par phrase, dépasse systématiquement `n_ctx` |

**Statut :** `layout` retenu pour usage principal. Précision légèrement supérieure à `plain`. Permet de récupérer les bbox pour le traitement des images dans la deuxième passe.

**`repetition_penalty`** testé — apparemment sans effet sur les boucles de génération.

### Comportement du mode `rec` (2026-04-03, page_6, BF16, binarize)

`Locate <|ref|>{target}<|/ref|> in the image.`

| Prompt | Résultat |
|---|---|
| `Locate <\|ref\|>A figure or graph<\|/ref\|> in the image.` | 1 bbox exacte pour le graphique, s'arrête proprement (5.8s) |
| `Locate <\|ref\|>A figure or graph<\|/ref\|> in the image. Describe it.` | identique — instruction post-bbox ignorée |
| `Locate <\|ref\|>A figure or graph<\|/ref\|> in the image. Parse it.` | identique — instruction post-bbox ignorée |
| `Locate <\|ref\|>every element<\|/ref\|> in the image.` | toutes les bboxes de la page (même résultat que `layout`) |
| `Locate <\|ref\|>everything<\|/ref\|> in the image.` | toutes les bboxes de la page |
| `Locate <\|ref\|>pliure du livre<\|/ref\|> in the image.` | toutes les bboxes de la page |
| `Locate <\|ref\|>image<\|/ref\|> in the image.` | toutes les bboxes de la page + boucle `<td>` sur le tableau |

**Comportement observé :** quand le modèle ne trouve pas l'élément demandé (ou que le target est trop générique), il retourne toutes les bboxes de la page. Quand il trouve un élément spécifique, il retourne uniquement sa bbox et s'arrête. Toute instruction ajoutée après le `<|det|>` est ignorée — la deux passes est obligatoire pour obtenir le contenu d'une région.

**Usage retenu :** `Locate <|ref|>A figure or graph<|/ref|> in the image.` pour test rapide, mais non nécessaire dans la pipeline puisque `layout` repère déjà les bbox avec le texte en plus. 

### Vocabulaire de labels grounding (mode `layout` et `rec`)

Observé sur pages 1–6 (BF16, binarize) :

| Label | Contenu |
|---|---|
| `text` | bloc de texte courant |
| `title` | titre |
| `sub_title` | sous-titre / intertitre |
| `table` | tableau (y compris figures mal classées sur image floue) |
| `table_caption` | légende de tableau |
| `table_footnote` | note de bas de tableau |
| `image` | figure / graphique (label correct sur image nette) |
| `image_caption` | légende de figure |

**Effet de la qualité d'image sur la classification :** page_5 (floue) → graphique classé `table`, hallucination `<td>30</td>` en boucle. Page_6 (nette, même contenu) → graphique correctement classé `image`, contenu vide.

**Contenu des régions `image` :** toujours vide — le modèle détecte et délimite la région mais ne génère aucune description. Voir section ci-dessous pour les résultats des deux passes sur le crop.

### Deuxième passe sur crop de graphique (2026-04-05, page_6, BF16, `draft/test_two_pass.py`)

| Combinaison | Résultat |
|---|---|
| `parse` + raw | tableau structuré, notes de bas de tableau absentes |
| `parse` + binarize | tableau structuré, notes de bas de tableau présentes mais traitées comme des lignes du tableau |
| `describe` + raw | description générale + bruit d'interprétation |
| `describe` + binarize | description générale + bruit d'interprétation |

**Verdict : `parse` + `binarize` retenu** pour la deuxième passe sur les régions `image`. Meilleure extraction des éléments visuels. Limitation connue : les notes de bas de tableau sont incluses dans le tableau au lieu d'être séparées.

### Prompts custom testés (2026-04-03, BF16, binarize)

| Prompt | Résultat |
|---|---|
| `"Describe this image in detail in french."` | description en anglais (instruction de langue ignorée) |
| `"Describe this image in detail in the language of the document."` | description en anglais |
| `"Décrit cette image en détail."` | description en anglais |
| `"Décrit cette image en détail en français."` | description en anglais |
| `"What is the language of the document?"` | répond `"pt"` puis fait l'OCR de la page |
| `"Figure or text?"` | boucle sur le titre |
| `"Does this document contain a figure?"` | décrit le document en anglais |
| `"Does this document contain a figure? Yes \| No"` | écrit les titres puis un court paragraphe incohérent en français |
| `"If this is a figure or illustration, describe it. Otherwise, Free OCR."` | décrit l'image en anglais même sur page de texte |
| `"Transcribe the text exactly as it appears."` | OCR instable et désordonné, qualité inférieure à `"Free OCR."` |
| `"Is there a figure in this document?"` | pages 1–3 : description générale ; page_4 : description détaillée du tableau ; page_5 : boucle |
| `"OCR only the text, ignore any figures."` | ignore figures et tableaux, mais skip aussi certaines colonnes de texte (comportement partiel — voir issues Feature 1) |
| `"<\|grounding\|>Describe this image in detail."` | équivalent exact de `layout` (grounding boxes + classification régions) — testé page_6 (43.5s) |
| `"<\|grounding\|>Parse the figure."` | **freeze terminal** — à éviter |

**Conclusion :** le modèle ignore systématiquement les instructions de langue. Les prompts de classification (yes/no, figure?) ne produisent pas de réponse structurée utilisable. `"OCR only the text, ignore any figures."` est le seul prompt qui filtre réellement les figures, mais de façon incomplète.

---

## Prétraitement des images

### Image originale (aucun prétraitement)
**Statut : abandonné.** Provoque des boucles de génération sur plusieurs pages.

### Exposure boost Pillow (contrast ×1.8 + brightness ×1.2)
**Statut : abandonné.** Évite les boucles mais produit moins de mots (~830 vs ~1000 pour binarize_adaptive), plus lent (~25s), davantage d'hallucinations.

### CLAHE (égalisation adaptative du contraste, OpenCV LAB)
**Statut : abandonné.** Testé visuellement (`draft/viz_preprocess2.py`) et en OCR (`draft/preprocess_test.py`). Produit ~1519 mots mais avec des boucles en fin de génération. Pas d'amélioration nette sur la précision.

### Binarisation Otsu (GaussianBlur(5,5) + seuil global Otsu)
**Statut : abandonné.** Boucle immédiatement, hallucinations massives (page_1 : répétitions de "Ils sont des mots utilisés dans la société").

### EqualizeHist seul / EqualizeHist + binarize adaptive
**Statut : abandonné.** Testé visuellement (`draft/viz_preprocess2.py`). EqualizeHist amplifie le bruit de fond et les gradients d'éclairage — contre-productif avant binarisation adaptative.

### bg_divide + binarize adaptive
**Statut : non retenu pour l'instant.** Testé visuellement (`draft/viz_preprocess2.py`). Normalise l'illumination en divisant par un fond estimé (GaussianBlur 101×101). Résultat visuel intéressant sur les pages avec éclairage très inégal, mais non testé en OCR.

### Binarisation adaptive (GAUSSIAN_C, blockSize=31, C=10)
**Statut : remplacé par blockSize=31, C=15.** Provoque des boucles de génération sur pages floues ou bruitées.

**Avantages :** rapide, supprime le fond, robuste aux variations d'éclairage locales.  
**Limites :** C=10 efface les traits mous sur images floues → boucles `<td>` HTML.

### GaussianBlur(5,5) + binarize adaptive (blockSize=31, C=15)
**Statut : retenu, paramètres par défaut mis à jour.** Grid test blockSize ∈ {21,31} × C ∈ {10,15} sur page_5 (bruitée, Laplacien=58) et page_6 (nette, Laplacien=134) via `draft/test_binarize_grid.py` + `draft/compare_grid.py`.

- C=10 (ancien défaut) : boucles HTML `<tr><td>` sur pages floues/bruitées. Détection de boucle par fréquence de mots insuffisante pour ce type de boucle — ajout de `_has_char_repeat` dans `ocr_client.py`.
- C=15 : aucune boucle sur page_5 ni page_6. blockSize=31 >≈ blockSize=21 sur page bruitée (91 % vs 85 % de similarité word-level vs référence page_6).
- Paramètre `C` augmenté de 10 à 15 dans `config.py`.

**Limites :** précision dégradée sur pages très bruitées (page_5) — amélioration supplémentaire à explorer.

### Unsharp Mask (standard : `img + alpha*(img - blurred)`)
**Statut : abandonné.** Testé dans `draft/test_unsharp.py`. Amplifie les hautes fréquences — ajoute des granulés sombres, dégrade la binarisation sur la plupart des configs. Inefficace sur du flou de mise au point (l'information haute fréquence est physiquement perdue).

### Unsharp Mask inversé (`blurred - alpha*(img - blurred)`)
**Statut : abandonné.** Aucune amélioration sur la binarisation, introduce des artefacts sur certaines configs.

### page-dewarp (lmmx, `pip install page-dewarp[jax]`)
**Statut : abandonné.**

Nos images sont des doubles pages (deux pages par photo). page-dewarp est conçu pour une seule page — il ne détecte pas correctement les contours ni les lignes de texte sur des images deux pages et produit un résultat dégradé.

Tentative de contournement : couper l'image en deux moitiés (mi-largeur), dewarp chaque moitié séparément, recombinaison. Problème : page-dewarp recadre la sortie selon sa détection de contours de page, ce qui coupe les bords des demi-images artificielles et rend la recombinaison incohérente.

---

## Paramètres de génération

| Paramètre | Valeur testée | Effet |
|---|---|---|
| `repetition_penalty` | 1.1 | apparemment sans effet sur les boucles de génération |
| `max_tokens` | 2048 | coupe la génération avant qu'elle parte en boucle infinie (à affiner) |
| `temperature` | 0.0 | déterministe, retenu |
