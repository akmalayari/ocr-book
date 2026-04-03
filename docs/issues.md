# Issues — travaux en cours

## Version actuelle — à finaliser

Objectif : précision 100% sur texte et tableaux, photos nettes uniquement.

### Bug 1 — Boucles de génération sur pages denses ou avec tableaux
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

**Pistes restantes :**
- Prompt adaptatif selon le contenu (voir Feature 1 — prochaines versions).

### Bug 2 — Précision OCR imparfaite malgré binarize_adaptive

**Résultats des tests de quantization (2026-04-01) :**
- Q8_0 → BF16 : amélioration (ex: "l'age" correctement transcrit vs "Page"), mais précision toujours imparfaite. F16 commet plus d'erreurs que BF16.
- BF16 : ~50s/page vs ~20s/page en Q8_0. Rapport qualité/vitesse défavorable mais vaut le coup pour le gain de précision.

**Pistes restantes :**
- Prompt plus directif (ex: `"Transcribe the text exactly as it appears."` ou prompt en français).
- Modèle alternatif (Qwen2-VL, etc.) potentiellement plus précis sur du français dense.

---

## Prochaines versions

### Feature 1 — Détection et traitement des graphiques
Appliquer un mode OCR différent selon le contenu de la page.

**Comportement confirmé des prompts (tests sur pages 1–6) :**
- `"plain"` — retourne le texte selon la trame du document. Usage principal. Sur page_6 : texte et tableau transcrits correctement, graphique silencieusement ignoré.
- `"layout"` — idem `"plain"` mais ajoute les grounding boxes. Classe les régions en `text`, `sub_title`, `table`, `image`, `image_caption`. Sur page_6 : graphique correctement classé `image`, contenu vide — le modèle ne génère rien pour les régions `image`.
- `"describe"` — décrit l'image en anglais (indépendamment de la langue du document).
- `"parse"` — analyse fine des éléments de l'image en anglais.

**Langue des descriptions :** le modèle ignore systématiquement les instructions de langue. `"describe"` et `"parse"` répondent toujours en anglais.

**Approches à tester (par ordre de priorité) :**

1. **Pipeline deux passes sur les pages mixtes (option prioritaire)** — passe 1 `layout` pour détecter les régions `image` + extraire leur bbox ; passe 2 `parse` sur le crop correspondant. Nécessite un parser de grounding boxes (déjà dans `viz_boxes.py`) + crop PIL + second appel VLM.

2. **Heuristique OpenCV** — ratio contours horizontaux / surface. Si densité faible → page image → mode `"describe"`. Configurable via un seuil dans `config.py`. Plus simple mais ne gère pas les pages mixtes.

3. **Prompts avec prefix grounding** — non testés : `"<|grounding|>Transcribe only text blocks."`. Le prefix grounding pourrait être plus précis sur les colonnes.

### Feature 2 — Support PDF multi-pages
Splitter un PDF en images (une par page) avant de l'envoyer au pipeline, via `pdf2image` ou `pymupdf`. À intégrer dans `collect_images` ou en amont.

### Bug 3 — Courbure de page
Courbure due à la reliure — déforme les lignes de texte géométriquement. Provoque des erreurs sur les mots coupés en fin de ligne.

**Pistes à tester (par ordre de priorité) :**
1. **Lignes de texte + fit polynomial + `cv2.remap`** — pure OpenCV, aucune dépendance. À appliquer sur chaque moitié de l'image séparément (double page). Approche retenue en priorité.
2. **DocTr** — transformer léger de déwarpage de documents, tourne sur CPU. Dépendance PyTorch/ONNX (~200 MB).
3. **DewarpNet / DocUNet** — modèle neural, résultats solides sur courbures prononcées. GPU requis, dépendance lourde. Dernier recours.

### Bug 4 — Robustesse du préprocessing (bruit, flou, éclairage inégal)
Objectif : rendre l'OCR plus tolérant aux images imparfaites. Unsharp Mask testé — abandonné (amplifie les granulés).

**Pistes à tester (par ordre de priorité) :**
1. **bg_divide + binarize adaptive** — normalise l'illumination en divisant par un fond estimé (GaussianBlur 101×101). Résultat visuel prometteur (déjà dans `draft/viz_preprocess2.py`). Non encore testé en OCR — piste prioritaire.
2. **Opérations morphologiques après binarisation** — `opening` supprime les artefacts de bruit, `closing` rebouche les trous dans les lettres.
3. **fastNlMeansDenoising avant binarisation** — débruitage non-local, préserve mieux les bords. Lent (~×10 vs Gaussian).
4. **Gris sans binarisation + denoising fort** — le modèle reçoit peut-être plus d'information depuis une image grise bien débruitée.
5. **Sauvola binarization** (`scikit-image`) — conçue pour les documents dégradés, tient compte de la variance locale.
6. **Real-ESRGAN** — super-résolution x2/x4, récupère des caractères illisibles. ~2–5s/image sur GPU. Dépendance lourde.

### Amélioration 1 — Performance BF16 (~50s/image)
BF16 est 2.5× plus lent que Q8_0 (~20s). Coût purement inference.

**Pistes à tester (par ordre de priorité) :**
1. **Réduire la résolution des images** — redimensionner à ~1200px de large avant VLM. Hypothèse : gain 30–50%.
2. **Profiler le vrai bottleneck** — chronométrer séparément l'encodage visuel et le décodage texte.
3. **Changer de wrapper Python (llama-cpp-python)** — secondaire, peu probable d'accélérer.

## Architecture

OK

## Style

OK
