# PaddleOCR-VL — Format de sortie et post-processing

## Flux de génération du markdown

```
VLM génère contenu brut (HTML pour tableaux, texte pour blocs texte)
    └─► save_to_markdown(pretty=True)   ← notre choix
          └─► _to_markdown(pretty=True)
                ├─ blocs texte/légende  → format_centered_by_html()  → <div style="text-align: center;">...</div>
                ├─ blocs tableau        → format_table_center_func()  → styles inline sur <td>, <th>, <table>
                └─ blocs image/figure   → format_image_scaled_by_html_func() → <img width="X%">
```

## Différence `pretty=True` vs `pretty=False`

### `pretty=True` (défaut)

```html
<!-- Légende tableau -->
<div style="text-align: center;">Tableau 1 : Titre du tableau</div>

<!-- Tableau -->
<table border=1 style='margin: auto; word-wrap: break-word;'>
  <tr><td style='text-align: center; word-wrap: break-word;'>Col1</td></tr>
</table>

<!-- Figure -->
<div style="text-align: center;">
  <img src="imgs/img_xxx.jpg" alt="Image" width="31%" />
</div>
```

### `pretty=False`

```html
<!-- Légende tableau -->
Tableau 1 : Titre du tableau

<!-- Tableau -->
<table><tr><td>Col1</td></tr></table>

<!-- Figure -->
<img src="imgs/img_xxx.jpg" alt="Image" />
```

## Notre approche : `pretty=True` + strip des styles table

**Choix retenu :** `save_to_markdown(pretty=True)` + regex dans `postprocess.py` pour supprimer les styles inline des balises table uniquement. Les `<div style="text-align: center;">` (légendes, titres) et le scaling des figures (`width="X%"`) sont conservés.

```python
# Dans postprocess.py / clean_page()
text = re.sub(r"<table\b[^>]*\bstyle='[^']*'", "<table border=1", text)
text = re.sub(r"<(t[dh])\b[^>]*\bstyle='[^']*'>", r"<\1>", text)
```

Résultat :
```html
<table border=1>
  <tr><td>Col1</td><td>Col2</td></tr>
</table>
```

## Ce que génère réellement le VLM

Le VLM génère le contenu des blocs, pas le wrapper HTML :
- **Blocs texte** : texte markdown brut
- **Blocs tableau** : HTML `<table><tr><td>...</td></tr></table>` (sans styles)
- **Blocs figure** : vide (le chemin de l'image est géré par PaddleOCR, pas le VLM)

Les styles inline (`style='...'`) sont ajoutés par PaddleOCR en post-processing dans `_to_markdown()` — ils ne sont **pas** générés par le VLM. Les désactiver ne change pas la vitesse de génération.

## `format_block_content` vs `pretty`

Ces deux paramètres sont indépendants et souvent confondus :

| Paramètre | Où | Effet |
|---|---|---|
| `format_block_content` | constructeur / predict | Formatage des blocs dans la sortie `.json()`. Défaut : `False`. |
| `pretty` | `save_to_markdown()` | Ajout de styles HTML dans le fichier `.md`. Défaut : `True`. |

`format_block_content=True` applique le même type de formatage mais dans la représentation JSON des blocs (utilisé pour le serving/API), pas pour le fichier markdown.

## Crops de figures

PaddleOCR sauvegarde les crops des régions image dans un sous-dossier `imgs/` relatif au `save_path` passé à `save_to_markdown()`. Le markdown référence ces crops par chemin relatif.

Dans notre pipeline : `save_path = output/figures/<page_stem>/` → crops dans `output/figures/<page_stem>/imgs/`.
