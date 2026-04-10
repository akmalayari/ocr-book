# Patch paddlex — per-region VLM error recovery

## Fichier modifié

```
C:\path\to\miniforge3\envs\py-3.10\Lib\site-packages\paddlex\inference\pipelines\paddleocr_vl\pipeline.py
```

## Problème

Pour les tableaux complexes (ex. page_4), `tokenize_figure_of_table()` retourne du contenu
OTSL (`<fcel>col1<fcel>col2<nl>...`) comme champ `image` au lieu d'un numpy array.
llama-server ne sait pas parser ce format et retourne une erreur 500 :

```
Exception from the 'vlm' worker: Error code: 500 - {'error': {'message': "Failed to parse input at pos 0: <fcel>..."}}
```

L'appel VLM étant **batché** (toutes les régions de même pixel_key en un seul appel),
une région défaillante fait planter l'image entière — texte compris.

## Localisation

Méthode `get_layout_parsing_results()`, boucle sur `batch_dict_by_pixel`, ~ligne 374.

## Changement

**Avant** — appel batché, pas de gestion d'erreur par région :

```python
images = batch_dict_by_pixel[pixel_key]["images"]
queries = batch_dict_by_pixel[pixel_key]["queries"]
batch_results = list(
    self.vl_rec_model.predict(
        [
            {
                "image": image,
                "query": query,
            }
            for image, query in zip(images, queries)
        ],
        skip_special_tokens=False if has_spotting else True,
        **kwargs,
    )
)
del images, queries
batch_dict_by_pixel[pixel_key]["vlm_results"] = batch_results
```

**Après** — appels individuels avec fallback OTSL :

```python
images = batch_dict_by_pixel[pixel_key]["images"]
queries = batch_dict_by_pixel[pixel_key]["queries"]
batch_results = []
for image, query in zip(images, queries):
    try:
        result = list(
            self.vl_rec_model.predict(
                [{"image": image, "query": query}],
                skip_special_tokens=False if has_spotting else True,
                **kwargs,
            )
        )[0]
    except Exception as _vlm_err:
        err_msg = str(_vlm_err)
        otsl_start = err_msg.find("<fcel>")
        if otsl_start != -1:
            # OTSL content echoed back by llama-server; convert directly
            result = {"result": err_msg[otsl_start:]}
        else:
            result = {"result": ""}
    batch_results.append(result)
del images, queries
batch_dict_by_pixel[pixel_key]["vlm_results"] = batch_results
```

## Logique du fallback OTSL

Quand llama-server renvoie une erreur 500 sur du contenu OTSL, il **echo le contenu
reçu** dans le message d'erreur. On extrait ce contenu depuis `err_msg` (recherche de
`<fcel>`) et on le place dans `result["result"]`. Le pipeline appelle ensuite
`convert_otsl_to_html(result_str)` à la ligne ~452, ce qui convertit l'OTSL en table
HTML normalement — comme si le VLM avait répondu correctement.

## Usage

```bash
# Appliquer le patch
python docs/dev/apply_paddlex_patch_otsl.py

# Vérifier sans modifier
python docs/dev/apply_paddlex_patch_otsl.py --check

# Restaurer l'original
python docs/dev/apply_paddlex_patch_otsl.py --revert
```
