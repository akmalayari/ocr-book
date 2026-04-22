# PaddleOCR-VL — Internal Pipeline Architecture

## Per-image Execution Sequence

```
image
  └─► [Doc Preprocessor]  ← disabled by default (use_doc_preprocessor: False)
        ├─ orientation classify (PP-LCNet_x1_0_doc_ori)
        └─ unwarping (UVDoc)
  └─► [Layout Detection]  ← PP-DocLayoutV3 (local, PaddlePaddle CPU)
        └─ detects blocks and their type (text, table, image, ...)
  └─► for each detected block:
        └─► [VLM Recognition]  ← PaddleOCR-VL-1.5-0.9B via llama-server (HTTP)
              └─ generates block content in markdown/HTML
  └─► assembly of blocks into final markdown
```

**Key consequence:** each block = separate HTTP call to llama-server. A page with 6 blocks (title, 2 texts, table, caption, figure) generates 6 sequential VLM calls. Pages with more blocks are proportionally slower.

## GPU Behavior

GPU in bursts, not continuous:
1. Layout detection → GPU burst (local PaddlePaddle)
2. Image encoding + llama-server HTTP call → GPU burst (Vulkan)
3. Idle between blocks (preparation, HTTP)

## Special Table Processing

PaddleOCR uses an OTSL pipeline for complex tables:
1. `ppdoclayout` detects and extracts cells via traditional OCR
2. Encodes content in OTSL format (`<fcel>col<fcel>col<nl>...`)
3. Sends OTSL to the VLM for HTML reconstruction

**Problem with llama-cpp-server:** llama-server cannot parse OTSL as image → 500 error. Workaround: paddlex patch that intercepts the error, extracts OTSL from the error message, and converts it directly via `convert_otsl_to_html()`. See `docs/dev/paddlex_patch_otsl.md`.

## Involved Models

| Component | Model | Backend |
|---|---|---|
| Layout detection | PP-DocLayoutV3 | PaddlePaddle CPU |
| VLM recognition | PaddleOCR-VL-1.5-0.9B (GGUF F16) | llama-server Vulkan |
| Orientation classify | PP-LCNet_x1_0_doc_ori | PaddlePaddle (disabled) |
| Unwarping | UVDoc | PaddlePaddle (disabled) |

## Block Labels (PP-DocLayoutV3)

Recognized labels and processing in markdown:

| Label | Description | In markdown |
|---|---|---|
| `text` | Regular text block | plain text |
| `paragraph_title` | Section title | markdown header |
| `doc_title` | Document title | markdown header |
| `figure_title` / `table_caption` | Figure/table caption | `<div style="text-align: center;">` |
| `table` | Table | `<table>...</table>` HTML |
| `image` / `chart` | Figure / chart | `<img src="...">` |
| `formula` / `display_formula` | Mathematical formula | raw content |
| `abstract` | Abstract | plain text |

Labels ignored by default (via `markdown_ignore_labels`):
`number`, `footnote`, `header`, `header_image`, `footer`, `footer_image`, `aside_text`
