# Patch parallélisme VLM intra-page

## Contexte

PaddleOCR-VL traite les blocs d'une page séquentiellement : pour chaque bloc détecté par PP-DocLayoutV3, un appel HTTP est envoyé à llama-server. Avec 5-8 blocs par page, ces appels s'enchaînent les uns après les autres alors que le GPU est souvent idle entre deux.

## Solution

`docs/dev/apply_paddlex_patch_parallel.py` patche `paddlex/inference/pipelines/paddleocr_vl/pipeline.py` pour remplacer la boucle séquentielle par un `ThreadPoolExecutor` global.

**Doit être appliqué après `apply_paddlex_patch.py` (patch OTSL).**

## Fonctionnement

La boucle originale :
```python
for pixel_key in batch_dict_by_pixel:
    for image, query in zip(images, queries):
        result = self.vl_rec_model.predict(...)  # séquentiel
```

Le patch collecte tous les blocs (toutes pixel_keys confondues) en une seule liste, les soumet au pool, puis redistribue les résultats :
```python
_all_tasks = [(img, qry, kwargs), ...]   # tous les blocs
_all_results = pool.map(_infer_block, _all_tasks)  # parallèle
# redistribution par pixel_key après
```

**Pourquoi thread-safe** : PaddleX utilise `asyncio.run_coroutine_threadsafe()` sur un event loop global unique (background thread). Plusieurs threads peuvent appeler `predict()` simultanément — leurs coroutines s'empilent sur le même loop, qui les exécute en I/O concurrent vers llama-server.

## Paramètres cohérents

`_VLM_PARALLEL` dans le patch doit correspondre à `-np` dans llama-server, et `-c` doit être dimensionné en conséquence :

| VLM_PARALLEL / -np | -c recommandé | Tokens/slot |
|---|---|---|
| 2 | 4096 | 2048 |
| 3 | 6144 | 2048 — **retenu** |
| 4 | 8192 | 2048 — crash vision encoder |

## Limites

- **Vision encoder Vulkan saturé à 4 workers** : 4 encodages image simultanés crashent le driver. Plancher stable à 3 workers.
- **Rendements décroissants** : 60s → 49s → 43.4s texte / 37s graphe (gain de 2.4s entre np=2 et np=3). Au-delà de 3, crash.
- **Pages avec peu de blocs** : une page avec 2 blocs n'utilise que 2 workers même avec np=3. Le gain est proportionnel au nombre de blocs.

## Usage

```bash
# Appliquer (patch OTSL doit déjà être actif)
python docs/dev/apply_paddlex_patch_parallel.py

# Vérifier
python docs/dev/apply_paddlex_patch_parallel.py --check

# Retirer (retour patch OTSL seul)
python docs/dev/apply_paddlex_patch_parallel.py --revert
```

## Config src/ associée

`src/pipeline.py` : `-np 3`
`src/config.py` : `n_ctx = 6144`
