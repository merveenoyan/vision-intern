# vision-intern

A Python library for building vision pipelines: label datasets with VLMs,
evaluate annotations with a VLM-as-judge, and train object detection models —
all from a few function calls.

## Recommended architecture (model roles)

This workflow uses **separate models for separate roles**. Keep them on
separate processes/endpoints — do not collapse them into one model.

| Role | Size | What it does | Where it runs |
|---|---|---|---|
| **Orchestrator** | ~10-12B (thinking) | Drives the workflow, babysits the run, fixes errors, decides next steps. **Never labels or judges itself.** | The agent/subagent |
| **Labeller** | ~7B-8B VLM | Generates detections on the dataset (one model, the larger of the workers). | Remote (HF Inference Providers) or local server, it costs $0.5/1k images |
| **Judges** | 2-5B VLMs (several) | Score/verify the labeller's detections. Use multiple small judges for an ensemble/voting signal. | Remote or local servers |

Why the separation matters:

- **The orchestrator must not run inference.** Doing labelling or judging
  inside the orchestrator process thrashes its KV cache and degrades (or
  crashes) the agent that is supposed to be supervising the run. The
  orchestrator only calls the workflow functions and watches their output.
- **Labeller > judges in capacity.** Labelling is the harder generative task,
  so give it the larger worker (~7B). Judging each detection is a narrower
  verification task that smaller models (4-5B) handle well and cheaply.
- **Judges are pluggable and parallelizable.** Because the judge step takes a
  `model_id` (and optional `base_url`), you can run several small judges and
  combine their verdicts, rather than relying on a single large model.

Concretely, in the `label_dataset` / `judge_labels` / `train` calls:

- Point `label_dataset(..., model_id=<7B VLM>)` at your labeller endpoint.
- Point `judge_labels(..., model_id=<4-5B VLM>)` at one or more judge endpoints.
- Keep the orchestrating agent on a ~12B thinking model that only issues these
  calls and monitors progress — it should never be passed as a `model_id`.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

### 1. Label a dataset

Use any VLM to auto-detect objects in images. Works with local image
directories or Hugging Face datasets.

```python
from workflows import label_dataset

# Local images → COCO JSON
label_dataset(
    source="data/images",
    classes=["table", "image", "chart"],
    output="annotations.json",
)

# HF dataset → HF dataset with detections column
label_dataset(
    source="lmms-lab/DocVQA",
    classes=["table", "image", "chart"],
    output="merve/docvqa-labeled",
    dataset_config="DocVQA",
    split="test",
    max_samples=1000,
    push_to_hub=True,
    backend="openai",
    base_url="http://localhost:8083/v1",   # llama-server
    api_key="sk-local",
)
```

### 2. Judge labels with a VLM

Score each detection and keep only the ones above a threshold.

```python
from workflows import judge_labels

judge_labels(
    source="merve/docvqa-labeled",
    output="merve/docvqa-judged",
    threshold=0.5,
    push_to_hub=True,
    backend="openai",
    base_url="http://localhost:8084/v1",   # small 4B judge server
    model_id="Qwen3-VL-4B-Instruct-Q8_0.gguf",
    api_key="sk-local-judge",
)
```

### 3. Train RF-DETR

Fine-tune a detection model on the curated dataset. This is a generalized
version of the [HF object-detection tutorial](https://huggingface.co/docs/transformers/tasks/object_detection):
lazy preprocessing, optional [Albumentations](https://albumentations.ai/)
augmentation, and COCO-style **mAP / mAR** evaluation via `torchmetrics`.

```python
from workflows import train

train(
    source="merve/docvqa-judged",  # or a local COCO directory
    model_id="Roboflow/rf-detr-base",
    epochs=10,
    batch_size=8,
    augment=True,          # Albumentations (no-op if not installed)
    val_split="test",      # held-out split for mAP/mAR; None to skip eval
    push_to_hub=True,
    hub_model_id="merve/rfdetr-docvqa",
    report_to="trackio",   # live metric tracking
)
```

Supported input formats (auto-detected):

- **HF dataset with an `objects` column** — the standard HF detection layout.
- **HF dataset with a `detections` column** — produced by `label_dataset` /
  `judge_labels` (Pascal-VOC boxes, converted automatically).
- **Local COCO directory** — `train/images/` + `train/labels.json`
  (and optional `val/`).

Any `AutoModelForObjectDetection` checkpoint works (RF-DETR, DETR, etc.); the
default is `Roboflow/rf-detr-base`. RF-DETR requires `transformers>=5.10` and
`timm`.

## Inference backends

All VLM-powered tools (`vlm_detect`, `document_ocr`, `ocr_judge`) and
workflows (`label_dataset`, `judge_labels`) support two backends:

### `backend="openai"` (recommended for serving)

Uses the OpenAI Python client. A single code path that works with:

| Provider | `base_url` | `api_key` |
|---|---|---|
| **HF Inference Providers** | `https://router.huggingface.co/v1` (default) | Your HF token |
| **vLLM** | `http://localhost:8000/v1` | Server API key |
| **llama-server** | `http://localhost:8084/v1` | Server API key |
| Any OpenAI-compatible endpoint | Custom URL | Custom key |

```python
from tools import vlm_detect

vlm_detect(
    "photo.jpg",
    classes=["person", "car"],
    backend="openai",
    base_url="https://router.huggingface.co/v1",
    api_key="hf_...",
    model_id="Qwen/Qwen3-VL-8B-Instruct",
)
```

### `backend="transformers"` (local GPU)

Loads the model directly with `transformers` + `torch`. Models are cached
after the first load.

```python
from tools import vlm_detect

vlm_detect(
    "photo.jpg",
    classes=["person", "car"],
    backend="transformers",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
)
```

## Tools reference

| Tool | Description |
|---|---|
| `detect` | RF-DETR closed-set detection (COCO 80 classes) |
| `instance_segment` | RF-DETR-Seg instance segmentation |
| `segment_from_bbox` | SAM3 — bbox prompt to high-quality masks |
| `segment_from_text` | Falcon-Perception — text prompt to zero-shot masks |
| `estimate_depth` | Depth Anything V2 — monocular relative depth |
| `estimate_pose` | Sapiens2 — dense 308-keypoint human pose |
| `grounded_detect` | MM-Grounding-DINO — open-vocabulary detection |
| `fast_segment` | EdgeTAM — lightweight bbox to mask |
| `ocr` | PaddleOCR-VL — vision-language OCR |
| `vlm_detect` | VLM instruction-prompted detection (any VLM) |
| `document_ocr` | Document OCR to markdown (text, tables, formulas) |
| `ocr_judge` | Pairwise OCR quality evaluation with ELO rating |
| `convert_bbox` | Convert bboxes between 6 formats |
| `validate_annotations` | Validate detection annotations for issues |
| `compute_stats` | Statistics for COCO annotation files |

## Workflows reference

| Workflow | Description |
|---|---|
| `label_dataset` | Auto-label images with a VLM for object detection |
| `judge_labels` | Score and filter labels with a VLM-as-judge |
| `train` | Fine-tune RF-DETR on labeled data |

All workflows support both **local directories** (COCO format) and
**Hugging Face datasets** as input and output.

## CLI usage

Each workflow can also be run from the command line:

```bash
# Label (7-8B labeller, here via HF Inference Providers)
python -m workflows.vlm_label \
    --source lmms-lab/DocVQA --dataset-config DocVQA \
    --classes "table,image,chart,diagram,figure" \
    --output merve/docvqa-labeled --push-to-hub \
    --backend openai --base-url https://router.huggingface.co/v1 \
    --api-key hf_... --model Qwen/Qwen3-VL-8B-Instruct \
    --split test --max-samples 100

# Judge (small 4B judge, here on a local llama-server)
python -m workflows.vlm_judge \
    --source merve/docvqa-labeled \
    --output merve/docvqa-judged --push-to-hub \
    --backend openai --base-url http://localhost:8084/v1 \
    --api-key sk-local-judge --model Qwen3-VL-4B-Instruct-Q8_0.gguf \
    --threshold 0.5

# Train (mAP/mAR eval on the test split, push to the Hub)
python -m workflows.train_rfdetr \
    --source merve/docvqa-judged --train-split test --val-split none \
    --model Roboflow/rf-detr-base --epochs 10 --batch-size 8 \
    --output-dir checkpoints/rfdetr-docvqa
```

## Example: full pipeline with role-separated models

Each role uses its own model (see "Recommended architecture" above). The
labeller is the larger worker; the judge is a small, cheap verifier. The
~12B orchestrator (e.g. Gemma-4) drives this script and is **never** used
as a `model_id` here.

Serve a small judge locally with llama-server (runs alongside the
orchestrator with little VRAM):

```bash
llama-server \
    --model Qwen3-VL-4B-Instruct-Q8_0.gguf \
    --mmproj mmproj-F16.gguf \
    --host 0.0.0.0 --port 8084 \
    --n-gpu-layers 99 --ctx-size 8192 \
    --api-key sk-local-judge
```

Then run the pipeline from Python:

```python
from workflows import label_dataset, judge_labels, train

# Labeller: ~7-8B VLM via HF Inference Providers (remote)
LABELLER = dict(
    backend="openai",
    base_url="https://router.huggingface.co/v1",
    api_key="hf_...",
    model_id="Qwen/Qwen3-VL-8B-Instruct",
)

# Judge: small ~4B VLM on a local llama-server
JUDGE = dict(
    backend="openai",
    base_url="http://localhost:8084/v1",
    api_key="sk-local-judge",
    model_id="Qwen3-VL-4B-Instruct-Q8_0.gguf",
)

# Step 1: label
label_dataset(
    source="lmms-lab/DocVQA",
    dataset_config="DocVQA",
    classes=["table", "image", "chart", "diagram", "figure"],
    output="merve/docvqa-media-labeled",
    split="test",
    max_samples=1000,
    push_to_hub=True,
    **LABELLER,
)

# Step 2: judge
judge_labels(
    source="merve/docvqa-media-labeled",
    output="merve/docvqa-media-judged",
    threshold=0.5,
    push_to_hub=True,
    **JUDGE,
)

# Step 3: train
train(
    source="merve/docvqa-media-judged",
    epochs=10,
    batch_size=4,
)
```

## License

MIT
