# Patch paddlex — per-region VLM error recovery

## Modified File

The patch script discovers the active PaddleX package instead of assuming a
platform-specific site-packages directory. Typical locations are:

```text
# Windows
C:\path\to\miniforge3\envs\ocr-livre\Lib\site-packages\paddlex\inference\pipelines\paddleocr_vl\pipeline.py

# Linux
/path/to/miniforge3/envs/ocr-livre/lib/python3.10/site-packages/paddlex/inference/pipelines/paddleocr_vl/pipeline.py
```

## Problem

For complex tables (ex. page_4), `tokenize_figure_of_table()` returns OTSL content
(`<fcel>col1<fcel>col2<nl>...`) as an `image` field instead of a numpy array.
llama-server cannot parse this format and returns a 500 error:

```
Exception from the 'vlm' worker: Error code: 500 - {'error': {'message': "Failed to parse input at pos 0: <fcel>..."}}
```

The VLM call being **batched** (all regions of the same pixel_key in a single call),
one failing region crashes the entire image — text included.

## Location

Method `get_layout_parsing_results()`, loop over `batch_dict_by_pixel`, ~line 374.

## Change

**Before** — batched call, no per-region error handling:

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

**After** — individual calls with OTSL fallback:

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

## OTSL Fallback Logic

When llama-server returns a 500 error on OTSL content, it **echoes the received
content** in the error message. We extract this content from `err_msg` (search for
`<fcel>`) and place it in `result["result"]`. The pipeline then calls
`convert_otsl_to_html(result_str)` at ~line 452, which converts the OTSL to an HTML
table normally — as if the VLM had responded correctly.

## Usage

```bash
# Apply the patch
python docs/dev/apply_paddlex_patch_otsl.py

# Check without modifying
python docs/dev/apply_paddlex_patch_otsl.py --check

# Restore original
python docs/dev/apply_paddlex_patch_otsl.py --revert
```
