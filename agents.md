# AGENTS.md — vision-agent

## What this is
Python library at `/home/merve/Desktop/vision-agent`: a 3-stage vision pipeline —
**label datasets with VLMs → judge labels with VLM-as-judge → train RF-DETR**.
Supports local dirs (COCO) + HF datasets, with `openai`-compatible and local
`transformers` backends.

## Model architecture (locked in, see README "Recommended architecture")
Three separate roles — never collapse into one model, never pass the orchestrator as a `model_id`:

| Role | Model | Where |
|---|---|---|
| **Orchestrator** | Qwen3.6-27B (`Qwen3.6-27B-UD-Q8_K_XL.gguf`) | llama-server `:8081`, key `sk-local-qwen36`. Drives/babysits the run. Never labels/judges. |
| **Labeller** | `Qwen/Qwen3-VL-8B-Instruct` (~8B) | HF Inference Providers (remote), `https://router.huggingface.co/v1`, HF token. |
| **Judge** | `Qwen3-VL-4B-Instruct-Q8_0.gguf` (~4B) | local llama-server `:8084`, key `sk-local-judge`. |

Judge GGUF + mmproj live at:
`~/.cache/huggingface/hub/models--unsloth--Qwen3-VL-4B-Instruct-GGUF/snapshots/00c00da0690c4b14b5539b02c4ea5d7c9102b35e/`
(`Qwen3-VL-4B-Instruct-Q8_0.gguf` + `mmproj-F16.gguf`). Start with:
```bash
~/llama.cpp/build/bin/llama-server \
  --model <.../Qwen3-VL-4B-Instruct-Q8_0.gguf> --mmproj <.../mmproj-F16.gguf> \
  --alias Qwen3-VL-4B-Instruct-Q8_0.gguf --host 0.0.0.0 --port 8084 \
  --n-gpu-layers 99 --ctx-size 8192 --api-key sk-local-judge
```

## Environment notes
- Use **`python3`** (no `python` on PATH).
- `transformers` 5.10.2, `datasets` 4.8.5, `torchmetrics` 1.9.0, `trackio`, `pycocotools`, `accelerate` installed.
- **`timm` and `albumentations` are NOT installed.** RF-DETR (`Roboflow/rf-detr-base`) needs `timm`; augmentation needs `albumentations`. `pip3 install timm albumentations` before training.
- HF token at `~/.cache/huggingface/token`.

## Key files
- `workflows/vlm_label.py` — `label_dataset()` (filters detections to requested classes in Hub mode).
- `workflows/vlm_judge.py` — `judge_labels()`. `threshold` gates by judge score; `threshold=0.0` keeps everything and just records `judge_verdicts`.
- `workflows/train_rfdetr.py` — `train()`. **Generalized from the HF object-detection tutorial**: lazy `with_transform`, optional Albumentations augmentation, COCO mAP/mAR via `torchmetrics`, `push_to_hub`, `trackio`. Auto-detects 3 input formats: HF `objects` column, pipeline `detections` column (xyxy→xywh), local COCO dir.
- `tools/vlm_client.py` — `run_vlm()` (image resize→1280px + JPEG, 3× retry w/ backoff).
- `tools/vlm_detect.py` — detection + bbox parsing.
- `run_docvqa_pipeline.py` — end-to-end (label 8B remote → judge 4B local:8084 → train). Judge threshold now `0.0`.
- `run_judge_zeroconf.py` — standalone re-judge at zero confidence → `merve/docvqa-media-judged`.
- `run_judge_train.py` / `run_train_only.py` — judge+train / train-only on the judged set.
- `visualize_detections.py` — viz labeled dataset → `viz_output/`.
- `visualize_judged.py` — viz judged dataset (boxes + judge scores) → `viz_judged/`. CLI: `--dataset/--split/--num/--output-dir`.
- `README.md` — full docs.

## Pipeline status (DocVQA, 1000 images, split = `test`)
- **Labeling — DONE.** Qwen3-VL-8B → `merve/docvqa-media-labeled` (split `test`), 997/1000 have detections.
- **Judging — DONE (zero confidence).** Re-ran with `threshold=0.0`: kept **all 1400 detections** → `merve/docvqa-media-judged` (split `test`), with a `judge_verdicts` column. The single-judge score is unreliable (mostly `0.0`, occasional `0.98`) — confirmed visually in `viz_judged/`, so we keep everything and defer to a future ensemble.
- **Training — NOT DONE.** No model trained yet. Was about to run; blocked only on `timm` install.

## Datasets (HF, account `merve`)
- `merve/docvqa-media-labeled` — labeller output (split `test`). May contain a few stray labels from before the class-filter fix.
- `merve/docvqa-media-judged` — zero-confidence judged output (split `test`), `detections` + `judge_verdicts`. Classes seen: `table`, `image` (subset of `["table","image","chart","diagram","figure"]`).

## How to run training (next step)
```bash
pip3 install timm albumentations
cd /home/merve/Desktop/vision-agent
python3 -m workflows.train_rfdetr \
  --source merve/docvqa-media-judged --train-split test --val-split none \
  --val-size 0.15 \                # hold out 15% for mAP (judged set has only `test`)
  --model Roboflow/rf-detr-base --epochs 10 --batch-size 8 \
  --output-dir checkpoints/rfdetr-docvqa
```
- The judged set has **only a `test` split (1400 boxes)**. Use `--val-size 0.15` to get mAP, or `--val-split none --val-size 0` to train on everything with no eval.
- Check GPU headroom before training: the orchestrator (Qwen3.6-27B) and judge (4B) also sit on the GPU. Lower `--batch-size` if OOM.
- `checkpoints/rfdetr-docvqa/` already has stale `checkpoint-*` dirs from an earlier (transformers-`Trainer`) attempt — safe to ignore/clean.

## Known open issues / next steps
1. **Run training** (install `timm` first).
2. **Multi-judge ensemble** not implemented — `judge_labels` takes a single `model_id`. The preserved `judge_verdicts` column enables per-judge voting; this is the planned next improvement.
3. **Judge server `:8084` is currently DOWN** — restart it (command above) before any judging.
4. `push_to_hub` preserves the **source split name** (e.g. `test`) — UX footgun; downstream code must pass `train_split="test"`.
5. Labeled dataset may contain stray labels (pre class-filter) — re-label if clean image/table-only data is wanted.

## Conventions
- Keep the three roles on separate processes/endpoints. The orchestrator must never run inference (KV-cache thrash).
- `viz_output/` and `viz_judged/` are local artifacts.

## what I want you to do
- make a visualization function and whenever you push a dataset to Hub, overlay bounding boxes on images so I don't have to run them. 
- if the job at hand requires multiple read/writes, use HF Buckets.
- I want you to implement multiple VLMs as judge support, specifically 2 models.
- I want you to run this pipeline with different models (for labelling and judging) on Hugging Face Jobs and babysit training runs. You need to write uv scripts and poll the Jobs until you get them to work. For detection, they have different prompts btw so make sure to check the model card. The labeller should be larger than both of the judge models, and all models should be of different families. Collect the uv scripts in this repository. For this run, you as Claude Opus will orchestrate. Later, we will try this with another open-source model (orchestration)
Here's the models:
    - https://huggingface.co/google/gemma-4-E4B-it
    - https://huggingface.co/google/gemma-4-E2B-it
    - https://huggingface.co/LiquidAI/LFM2.5-VL-450M
    - https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B
    - https://huggingface.co/moondream/moondream3-preview
    - https://huggingface.co/Qwen/Qwen3.5-4B
    - https://huggingface.co/Qwen/Qwen3.5-9B 



