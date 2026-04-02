# Tested — ce qui a été expérimenté dans le projet

Pages de référence utilisées pour les tests : `page_1` à `page_5`.
- page_1 : texte simple, mauvais éclairage
- page_2 : texte dense, éclairage normal
- page_3 : texte + tableau textuel
- page_4 : texte + tableau numérique
- page_5 : texte + tableau + graphique

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
| BF16 | ~60s/page | légèrement meilleure sur pages difficiles |

**Statut : BF16 non retenu.** Rapport qualité/vitesse défavorable (×3 sur le temps, gain marginal). Q8_0 utilisé par défaut.

---

## Prompts

Testés sur pages 1–5 avec `preprocess=binarize` (`draft/prompt_test.py`).

| Mode | Prompt | Résultat |
|---|---|---|
| `plain` | `"Free OCR."` | texte brut propre sur pages simples, boucle sur pages denses |
| `layout` | `"<|grounding|>Convert the document to markdown."` | ajoute grounding boxes, règle la boucle de page_2 mais boucle sur les tableaux (`<tr>`/`<td>`) |
| `describe` | `"Describe this image in detail."` | description en anglais, indépendamment de la langue du document |
| `parse` | `"Parse the figure."` | analyse fine des éléments visuels en anglais |
| `classic` | (supprimé) | une grounding box par phrase, dépasse systématiquement `n_ctx` |

**Statut :** `plain` retenu pour usage principal. `layout` utile sur certaines pages (page_2), contre-productif sur les tableaux. Aucun prompt universel ne couvre tous les types de page.

**`repetition_penalty`** testé — sans effet sur les boucles de génération.

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

### Binarisation adaptive seule (GAUSSIAN_C, blockSize=31, C=10)
**Statut : retenu (version initiale).** Meilleur compromis vitesse/qualité parmi les premiers tests : ~19s/page, ~1000 mots, évite les boucles. Perd la mise en forme Markdown (titres, italiques) mais améliore la précision du texte.

**Avantages :** rapide, supprime le fond, robuste aux variations d'éclairage locales, réduit les hallucinations.  
**Limites :** granulés sur les zones à fort bruit (plis, textures de papier), sensible aux images floues.

### GaussianBlur(5,5) + binarize adaptive
**Statut : retenu, à intégrer dans `preprocess.py`.** Validé sur pages 1–5 (`draft/test_blur_binarize.py`). Supprime les granulés avant binarisation, texte plus net. Améliore les résultats sur `plain` et `layout`. Pas encore intégré dans `src/preprocess.py`.

**Avantages :** même rapidité que binarize seule, réduit les artefacts de grain.  
**Limites :** n'aide pas sur du flou de mise au point prononcé.

### Unsharp Mask (standard : `img + alpha*(img - blurred)`)
**Statut : abandonné.** Testé dans `draft/test_unsharp.py`. Amplifie les hautes fréquences — ajoute des granulés sombres, dégrade la binarisation sur la plupart des configs. Inefficace sur du flou de mise au point (l'information haute fréquence est physiquement perdue).

### Unsharp Mask inversé (`blurred - alpha*(img - blurred)`)
**Statut : abandonné.** Aucune amélioration sur la binarisation, introduce des artefacts sur certaines configs.

---

## Paramètres de génération

| Paramètre | Valeur testée | Effet |
|---|---|---|
| `repetition_penalty` | 1.1 | sans effet sur les boucles de génération |
| `max_tokens` | 2048 | coupe la génération avant qu'elle parte en boucle infinie (à affiner) |
| `temperature` | 0.0 | déterministe, retenu |
