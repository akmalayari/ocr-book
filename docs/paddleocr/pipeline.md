# PaddleOCR-VL — Architecture interne du pipeline

## Séquence d'exécution par image

```
image
  └─► [Doc Preprocessor]  ← désactivé par défaut (use_doc_preprocessor: False)
        ├─ orientation classify (PP-LCNet_x1_0_doc_ori)
        └─ unwarping (UVDoc)
  └─► [Layout Detection]  ← PP-DocLayoutV3 (local, PaddlePaddle CPU)
        └─ détecte les blocs et leur type (text, table, image, ...)
  └─► pour chaque bloc détecté :
        └─► [VLM Recognition]  ← PaddleOCR-VL-1.5-0.9B via llama-server (HTTP)
              └─ génère le contenu du bloc en markdown/HTML
  └─► assemblage des blocs en markdown final
```

**Conséquence clé :** chaque bloc = un appel HTTP séparé à llama-server. Une page avec 6 blocs (titre, 2 textes, tableau, légende, figure) génère 6 appels VLM séquentiels. Les pages avec plus de blocs sont proportionnellement plus lentes.

## Comportement GPU

GPU en pics, pas en continu :
1. Layout detection → pic GPU (PaddlePaddle local)
2. Encodage image + appel HTTP llama-server → pic GPU (Vulkan)
3. Idle entre blocs (préparation, HTTP)

## Traitement spécial des tableaux

PaddleOCR utilise un pipeline OTSL pour les tableaux complexes :
1. `ppdoclayout` détecte et extrait les cellules via OCR traditionnel
2. Encode le contenu en format OTSL (`<fcel>col<fcel>col<nl>...`)
3. Envoie l'OTSL au VLM pour reconstruction en HTML

**Problème avec llama-cpp-server :** llama-server ne sait pas parser l'OTSL comme image → erreur 500. Contournement : patch paddlex qui intercepte l'erreur, extrait l'OTSL du message d'erreur, et le convertit directement via `convert_otsl_to_html()`. Voir `docs/dev/paddlex_patch.md`.

## Modèles impliqués

| Composant | Modèle | Backend |
|---|---|---|
| Layout detection | PP-DocLayoutV3 | PaddlePaddle CPU |
| VLM recognition | PaddleOCR-VL-1.5-0.9B (GGUF F16) | llama-server Vulkan |
| Orientation classify | PP-LCNet_x1_0_doc_ori | PaddlePaddle (désactivé) |
| Unwarping | UVDoc | PaddlePaddle (désactivé) |

## Labels de blocs (PP-DocLayoutV3)

Labels reconnus et traitement dans le markdown :

| Label | Description | Dans markdown |
|---|---|---|
| `text` | Bloc de texte courant | texte brut |
| `paragraph_title` | Titre de section | header markdown |
| `doc_title` | Titre du document | header markdown |
| `figure_title` / `table_caption` | Légende figure/tableau | `<div style="text-align: center;">` |
| `table` | Tableau | `<table>...</table>` HTML |
| `image` / `chart` | Figure / graphique | `<img src="...">` |
| `formula` / `display_formula` | Formule mathématique | contenu brut |
| `abstract` | Résumé | texte brut |

Labels ignorés par défaut (via `markdown_ignore_labels`) :
`number`, `footnote`, `header`, `header_image`, `footer`, `footer_image`, `aside_text`
