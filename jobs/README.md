# HF Jobs — multi-model labelling / judging / training

Self-contained `uv` scripts that run the pipeline on Hugging Face Jobs with a
**different model per role** (all different families; labeller larger than both
judges):

| Stage | Script | Model | Family | Size | Flavor |
|---|---|---|---|---|---|
| 1 Label | `label_qwen.py` | `Qwen/Qwen3.5-9B` | Qwen | 9.65B | `cpu-upgrade`¹ |
| 2 Judge A | `judge_one.py` | `google/gemma-4-E4B-it` | Google | 8.0B | `l40sx1`² |
| 2 Judge B | `judge_one.py` | `LiquidAI/LFM2.5-VL-1.6B` | Liquid | 1.6B | `l4x1`² |
| 3 Merge | `merge_judges.py` | — (ensemble) | — | — | `cpu-upgrade` |
| 4 Train | `train_rfdetr_job.py` | `Roboflow/rf-detr-base` | — | — | `l4x1`² |

> **² Pick the flavor by the compute bottleneck, not the stage name.** The
> heavy step is **judging with the larger VLM** (an 8B judge over ~1.4K images is
> the long pole) — give *that* the big GPU (`l40sx1`). A **small judge**
> (≤2B, e.g. LFM-1.6B) runs fine on `l4x1`. **RF-DETR training on a small
> curated set** (~1–2K images, ~10 epochs) is light and single-GPU — `l4x1` is
> plenty; reserve `l40sx1`/multi-GPU for large datasets or long schedules. So in
> practice the big GPU goes to the *large judge*, not to training. CPU stages
> (router labelling, merge) stay on `cpu-upgrade`.

¹ Qwen3.5-9B has live HF Inference Providers, so `label_qwen.py` labels through
the HF router (`openai` backend) on a CPU job — no GPU load. It uses the
per-object `bbox_2d` (0-1000) prompt (`tools/vlm_detect`), giving tight boxes
(median area ≈ 0.09 of the page).

> **Swapping the labeller.** Any VLM with a detection prompt works — the choice
> is per-domain. A detection specialist like `moondream/moondream3-preview`
> (native `.detect()`) is great on natural imagery but tends to return
> page-spanning boxes on *scanned document pages* (median area ≈ 0.98), which is
> why this DocVQA run uses Qwen's tight `bbox_2d` boxes instead.

Each script clones this repo (`REPO_REF` env, default `multimodel-jobs`) for the
shared `tools/` + `workflows/` helpers, so **push your branch before launching**.
Intermediate verdicts pass between jobs through the bucket
`merve/vision-agent-runs` (mounted at `/data`) — the multi-read/write case
called for in `agents.md`.

Datasets / artifacts (separate from the originals):
`merve/docvqa-media-labeled-qwen` → `merve/docvqa-media-judged-ensemble` →
`merve/rfdetr-docvqa-qwen`. Every dataset push includes an auto-generated
box-overlay gallery (`viz/` + README), so boxes never need re-rendering.

### How judging works (and its limits)
The labeller stores a `detections_overlay` column — each proposed box drawn and
**numbered** (`#0`, `#1`, …) on the image. Judges score that overlay directly,
evaluating each numbered box by *looking at it*, instead of being handed raw
`bbox` coordinates over the bare image (VLMs reason about pixel coordinates
poorly, the main hallucination source). The per-judge `score` is still an
**uncalibrated VLM confidence**, not a metric — so the keep decision is gated
by two cheap, non-vibe checks: `--min-agree 2` (both judges must vote
`correct`) and a `--max-area-frac 0.9` page-spanning guard (a pure geometric
filter, no VLM). Both checks are recorded per detection in `judge_verdicts`
(`area_frac`, `geom_keep`). Judges fall back to rendering the overlay on the fly
when the column is absent (e.g. datasets labelled before this change).

## One-time setup
```bash
hf buckets create vision-agent-runs          # run bucket (idempotent)
git push origin multimodel-jobs              # jobs clone this ref
```

## Running the same scripts locally
Nothing here is Jobs-only. Two knobs make a stage run on your machine against
your working tree:

- **Code: `REPO_DIR`.** Each script imports `tools/` + `workflows/` from a repo.
  Unset, it clones `REPO_REF` (the Jobs default — no checkout there). Set
  `REPO_DIR=$(pwd)` to use your **local edits** and skip the clone:
  ```bash
  HF_TOKEN=$(hf auth token) REPO_DIR=$(pwd) uv run jobs/judge_one.py -- \
    --model google/gemma-4-E4B-it --dataset merve/roadsign-labeled-qwen \
    --split train --out roadsigns/verdicts_gemma.parquet --max-samples 20
  ```
- **Data: the bucket, addressed directly.** The cross-stage verdicts live in the
  bucket (canonical). On a Job it's FUSE-mounted at `/data`; locally there's no
  mount, so the scripts address it over `hf://buckets/<id>` automatically. The
  `--out` / `--verdicts` paths are **relative names** placed under the data root,
  resolved as: `--data-root` → `$DATA_ROOT` → `/data` (if mounted) → the bucket
  over `hf://`. So the *same* `--out roadsigns/verdicts_gemma.parquet` writes to
  `/data/...` on a Job and to `hf://buckets/merve/vision-agent-runs/...` locally.
  Override with `--data-root ./runs` (a plain local dir, fully offline) or
  `--data-root hf://buckets/<id>` / `--bucket <id>`. Absolute paths and `hf://`
  URIs are still accepted verbatim.

Merge reads the same way — relative `--verdicts "label::roadsigns/v.parquet"`
resolves under the data root, so a local merge can read verdicts a Job wrote to
the bucket (and vice-versa).

## Smoke test (≈20 images, cheap)
Append `--max-samples 20` to stages 1–3.

## Full run
```bash
BUCKET="-v hf://buckets/merve/vision-agent-runs:/data"
REF="-e REPO_REF=multimodel-jobs"
DIR=/data/docvqa-qwen

# 1 — label (router, CPU). --timeout 3h: the full 1000-row push runs past the
# default timeout and gets flagged ERROR even though the data uploaded fine.
# --dedupe: DocVQA has ~3.4 question rows per page, so dedupe by docId first
# (1000 rows → 296 unique pages) to avoid labelling each page several times.
hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN --timeout 3h $REF -d \
  jobs/label_qwen.py -- --output merve/docvqa-media-labeled-qwen --dedupe

# 2 — judges (after stage 1 SUCCEEDED; run both in parallel). The LARGE judge is
# the long pole, so it gets l40sx1; the small judge runs on l4x1 (see flavor note
# above). --timeout 3h: the 8B judge over ~1.4K rows runs past the default job
# timeout; without it the job is killed AFTER writing verdicts and flagged ERROR.
hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN --timeout 3h $BUCKET $REF -d \
  jobs/judge_one.py -- --model google/gemma-4-E4B-it \
  --dataset merve/docvqa-media-labeled-qwen --out $DIR/verdicts_gemma.parquet
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN --timeout 3h $BUCKET $REF -d \
  jobs/judge_one.py -- --model LiquidAI/LFM2.5-VL-1.6B \
  --dataset merve/docvqa-media-labeled-qwen --out $DIR/verdicts_lfm.parquet

# 3 — merge (after both judges SUCCEEDED)
hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN $BUCKET $REF -d \
  jobs/merge_judges.py -- --dataset merve/docvqa-media-labeled-qwen \
  --output merve/docvqa-media-judged-ensemble \
  --verdicts "google/gemma-4-E4B-it::$DIR/verdicts_gemma.parquet" \
  --verdicts "LiquidAI/LFM2.5-VL-1.6B::$DIR/verdicts_lfm.parquet" \
  --min-agree 2 --max-area-frac 0.9

# 4 — train RF-DETR (after merge SUCCEEDED). Small curated set + ~10 epochs is
# light and single-GPU, so l4x1 suffices (bump to l40sx1 only for large data /
# long schedules). The val split is grouped by image
# (tools.dataset_utils.grouped_train_val_split), so repeated images can't leak
# across train/val — no need to pre-split.
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN $REF --timeout 6h -d \
  jobs/train_rfdetr_job.py -- --epochs 10 --batch-size 8 \
  --source merve/docvqa-media-judged-ensemble \
  --hub-model-id merve/rfdetr-docvqa-qwen
```

## Babysitting
```bash
hf jobs ps                      # list running jobs + ids
hf jobs logs <job_id>           # fetch logs (add -f to follow)
hf jobs inspect <job_id>        # status / exit code
hf jobs cancel <job_id>         # stop
hf buckets ls merve/vision-agent-runs/docvqa-qwen -h        # check artifacts
```
Each stage prints `STAGE N DONE` on success — grep logs for that plus
`Traceback|Error|OOM|Killed` to catch failures.
