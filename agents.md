# AGENTS.md — vision-agent

## What this is
A Python toolkit for **VLM-driven object-detection pipelines**, meant to be
*driven by an orchestrating agent*. Three stages:

1. **Label** a dataset with a VLM (open-vocabulary detection from class names).
2. **Judge** the labels with one or more smaller VLMs-as-judge + cheap geometric
   checks, and keep an auditable verdict per box.
3. **Train** a detection model (RF-DETR by default) on the curated result.

Every stage works on either a **local directory** (COCO format) or a **Hugging
Face dataset**, with an OpenAI-compatible backend (HF Inference Providers, vLLM,
llama-server, …) or a local `transformers` backend. The task domain is
arbitrary — documents, road signs, products, anything you can name as classes.

**Worked use cases live in `examples/`** — one subfolder per use case, each with
its own README (goal, data, classes, model-per-role, exact commands, outputs,
gotchas). Before building a new pipeline, **list `examples/` and read the
closest one** — it shows a real class set and model wiring to imitate. Copy
`examples/_template/` to start a new one. The seed example
(`examples/docvqa-media/`) is *one* demo, not the only use case.

## Model architecture (the core design constraint)
The pipeline uses **separate models for separate roles** — keep them on separate
processes/endpoints, and **never pass the orchestrator as a worker `model_id`**:

| Role | Size | Job | Notes |
|---|---|---|---|
| **Orchestrator** | ~10-30B (thinking) | Drives/babysits the run, fixes errors, decides next steps. **Never labels or judges itself.** | Runs inference inside the orchestrator process thrashes its KV cache and degrades the agent supervising the run. |
| **Labeller** | the largest worker | Proposes detections on each image. | One model; labelling is the hard generative task, so give it the most capacity. |
| **Judge(s)** | smaller than the labeller | Score/verify the labeller's boxes. | Use **2+ judges of different families** for an ensemble/voting signal. Pluggable via `model_id` (+ optional `base_url`). |

Rules that hold regardless of which specific models you pick:
- **Labeller strictly larger than every judge.**
- **All models from different families** (so judge errors are uncorrelated).
- **Orchestrator only issues the workflow calls and watches output** — it is not
  in the `LABELLER`/`JUDGE` config dicts.

The specific models are a per-run choice. The reference run on the
`multimodel-jobs` branch uses a Qwen labeller with Google-Gemma + LiquidAI-LFM
judges on HF Jobs — see `jobs/README.md` for that concrete wiring and the model
cards' detection-prompt differences.

## Environment & setup
- This is a **`uv` project** (`pyproject.toml` + `uv.lock`, Python pinned in
  `.python-version`). The flat layout (`tools/`, `workflows/`, `jobs/`) is kept
  importable via `pythonpath=["."]`; it is **not** a built wheel
  (`[tool.uv] package = false`).
- **Light core vs. training extra.** The default deps cover the label/judge/merge
  path with **no torch**. RF-DETR training and local-GPU inference need the
  `train` extra:
  ```bash
  uv sync                 # core: label / judge / merge (no torch)
  uv sync --extra train   # adds torch, transformers, timm, albumentations, …
  uv sync --extra all     # train + dev (pytest, ruff)
  ```
  The lazy `tools/__init__.py` and `workflows/__init__.py` only import torch when
  a training symbol is actually touched, so the light path stays light.
- Use **`python3`** if invoking directly (no `python` on PATH); prefer
  `uv run <cmd>` so the environment is the locked one.
- An HF token is read from the environment / `hf auth login` (`~/.cache/huggingface/token`).

## Repository layout
- `tools/` — composable, model-backed vision tools + CPU-only helpers (bbox
  conversion/validation/stats, dataset grouping, viz). Each returns plain
  dicts/lists. See `tools/__init__.py` for the public surface.
- `workflows/` — the three pipeline stages as functions **and** CLIs:
  `label_dataset` / `judge_labels` / `train`.
- `tools/registry.py` + `tools/config.py` — the **agent tool layer**: a lazy,
  JSON-schema'd registry over the `tools/` + `workflows/` functions
  (`get_tools` / `as_json_schema` / `call`) with credential/endpoint params
  hidden behind per-role `ToolConfig` (`configure(default=/labeller=/judge=)`).
  Re-exported from `tools/__init__.py` and the top-level `vision_agent.py` shim.
  `get_tools()` is torch-free by default (skips `requires_train` tools).
- `jobs/` — self-contained PEP-723 `uv` scripts that run the pipeline on **HF
  Jobs**, one model per role. They clone this repo for the shared `tools/` +
  `workflows/` helpers, so **push your branch before launching**. See
  `jobs/README.md`.
- `tests/` — `tests/unit/` (offline smoke tests: bbox round-trips, dataset
  grouping, job-script PEP-723 validity) and `tests/integration/` (opt-in,
  needs an HF token).
- `examples/` — worked use cases, **one subfolder per use case** (each a README
  recipe: goal/data/classes/models/commands/outputs/gotchas). Read-and-imitate;
  nothing imports from here. New use cases get dumped here — copy
  `examples/_template/`. See `examples/README.md`.
- `README.md` — user-facing docs and API reference.

## Key modules
- `workflows/vlm_label.py` — `label_dataset()`. Filters detections to the
  requested classes; stores a numbered `detections_overlay` image per row for
  the judges to look at.
- `workflows/vlm_judge.py` — `judge_labels()` and the ensemble helpers
  (`score_detections`, `ensemble_row`, `generate_object_specs`). Judges score the
  *numbered overlay* (VLMs reason about drawn boxes far better than raw pixel
  coords). `threshold` gates by judge score; `0.0` keeps everything and just
  records `judge_verdicts`.
- `workflows/train_rfdetr.py` — `train()`. Generalized from the HF
  object-detection tutorial: lazy `with_transform`, optional Albumentations
  augmentation, COCO mAP/mAR via `torchmetrics`, `push_to_hub`, `trackio`.
  **Auto-detects 3 input formats**: HF `objects` column, pipeline `detections`
  column (xyxy→xywh), and a local COCO directory.
- `tools/vlm_client.py` — `run_vlm()` (image resize→1280px + JPEG, retry w/ backoff).
- `tools/vlm_detect.py` — VLM detection + bbox parsing.
- `tools/bbox_utils.py`, `tools/dataset_utils.py` — CPU-only, no model load.
- `tools/hub_viz.py` — `push_dataset_with_viz()`: every dataset push includes an
  auto-generated box-overlay gallery + README, so boxes never need re-rendering.

## Running it
Quickest path is the CLIs (also see `README.md` and `jobs/README.md`):
```bash
# Label (point --backend/--base-url/--model at your labeller endpoint)
uv run python -m workflows.vlm_label  --source <hf-dataset-or-dir> \
  --classes "a,b,c" --output <out> [--push-to-hub] --backend openai \
  --base-url <url> --api-key <key> --model <labeller-model>

# Judge (small judge endpoint; run several and merge for an ensemble)
uv run python -m workflows.vlm_judge  --source <labeled> --output <judged> \
  --backend openai --base-url <url> --api-key <key> --model <judge-model>

# Train (mAP/mAR eval optional; needs the `train` extra)
uv run --extra train python -m workflows.train_rfdetr --source <judged> \
  --model Roboflow/rf-detr-base --epochs 10 --batch-size 8 --output-dir <dir>
```
For the full multi-model run **on HF Jobs**, follow `jobs/README.md`
(label → 2 judges in parallel → merge → train), passing artifacts between stages
through an `hf` bucket mounted at `/data`.

## Tests
```bash
uv run --extra dev pytest tests/unit        # offline, fast, no token
uv run --extra dev pytest -m integration    # opt-in: Hub auth + data path, needs HF_TOKEN
```
The unit suite includes a **job-script guard** that compile-checks every
`jobs/*.py` and validates its PEP-723 header — run it before pushing the branch
that HF Jobs will clone.

## Operating discipline (conventions)
- **Be decisive — act over ask.** Pick the sensible default, state the assumption
  in one line, and proceed. Ask the human only when genuinely blocked, the action
  is destructive/irreversible/outward-facing (force-push, delete, paid fan-out),
  or a wrong guess wastes real time or money — and batch unavoidable questions
  into one. Don't survey options for paths you won't take.
- **Check `examples/` first.** Before building a pipeline for a new domain, read
  the closest use case in `examples/`; when a run is worth keeping, write it up
  there (copy `examples/_template/`) and add it to the index.
- **Keep the three roles on separate endpoints**; the orchestrator never runs
  inference.
- **Smoke-test small, then scale.** Run any new stage with `--max-samples 20`
  first; confirm it SUCCEEDED before launching the full run. On HF Jobs, submit
  **one** job, verify, then fan out the rest.
- **Pick the Jobs flavor by the compute bottleneck, not the stage name.** The
  heavy step is **judging with the larger VLM** (an ~8B judge over thousands of
  images is the long pole) → give it `l40sx1`. A **small judge** (≤2B) runs on
  `l4x1`. **RF-DETR training on a small curated set** (~1–2K images, ~10 epochs)
  is light + single-GPU → `l4x1` is plenty; reserve `l40sx1`/multi-GPU for large
  data or long schedules. CPU stages (router labelling, merge) → `cpu-upgrade`.
  So the big GPU usually goes to the *large judge*, not to training.
- **Never lose artifacts.** Training always sets `push_to_hub=True` + an explicit
  `hub_model_id`. Give Jobs a generous `--timeout` (a long final push can run
  past the default and get flagged ERROR *after* the data uploaded fine).
- **Don't silently change the user's approach** (dataset/model/method/classes).
  If data is unavailable, ask — don't substitute.
- **Visualize on push.** Dataset pushes go through `push_dataset_with_viz()` so a
  box-overlay gallery ships with the data — no need to re-render to inspect.
- **Use a bucket for multi-read/write** flows (the `jobs/` pipeline passes
  verdicts between stages via `hf://buckets/.../`). On a Job the bucket is
  mounted at `/data`; locally the scripts address it directly over `hf://` (via
  `tools/run_store.py`), so a relative `--out`/`--verdicts` name works in both —
  see `jobs/README.md` §"Running the same scripts locally". Run a stage against
  your working tree with `REPO_DIR=$(pwd)` instead of the clone.
- `viz_output/`, `viz_judged/`, `viz_test_predictions/` are local artifacts
  (gitignored).

## Extension points / known gaps
- **Judge ensemble**: implemented for the Jobs path (`merge_judges.py`,
  `--min-agree`, `--max-area-frac`). The single-judge score is an *uncalibrated*
  VLM confidence — gate keeps with cheap non-vibe checks (vote agreement + a
  page-spanning area guard), recorded per box in `judge_verdicts`.
- **Split-name footgun**: `push_to_hub` preserves the *source* split name, so
  downstream calls may need an explicit `train_split=<name>`.
- **Agent tool layer**: done — `tools/registry.py` exposes the functions as a
  JSON-schema'd, role-configured registry for an in-process agent
  (`get_tools` / `as_json_schema` / `call`, see `README.md` §"Using it as an
  agent toolkit"). Returns stay rich Python objects (in-process); a
  wire/serialization layer (e.g. RLE-encoding `instance_segment`'s mask tensor)
  is a thin future add.
- **Packaging**: still a flat-layout `uv` project. The remaining deferred step is
  moving `tools/`/`workflows/` under a single `vision_agent/` import namespace so
  it can be `pip install`ed and imported from anywhere (would touch every import +
  the `jobs/` clone bootstrap; the top-level `vision_agent.py` shim gives the
  `import vision_agent` ergonomics in the meantime).
