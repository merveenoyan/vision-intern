# HF Jobs — multi-model labelling / judging / training

Self-contained `uv` scripts that run the pipeline on Hugging Face Jobs with a
**different model per role** (all different families; labeller larger than both
judges):

| Stage | Script | Model | Family | Size | Flavor |
|---|---|---|---|---|---|
| 1 Label | `label_moondream.py` | `moondream/moondream3-preview` | moondream | 9.3B | `l4x1` |
| 2 Judge A | `judge_one.py` | `google/gemma-4-E4B-it` | Google | 8.0B | `l4x1` |
| 2 Judge B | `judge_one.py` | `LiquidAI/LFM2.5-VL-1.6B` | Liquid | 1.6B | `l4x1` |
| 3 Merge | `merge_judges.py` | — (ensemble) | — | — | `cpu-upgrade` |
| 4 Train | `train_rfdetr_job.py` | `Roboflow/rf-detr-base` | — | — | `l40sx1` |

Each script clones this repo (`REPO_REF` env, default `multimodel-jobs`) for the
shared `tools/` + `workflows/` helpers, so **push your branch before launching**.
Intermediate verdicts pass between jobs through the bucket
`merve/vision-agent-runs` (mounted at `/data`) — the multi-read/write case
called for in `agents.md`.

Datasets / artifacts (separate from the originals):
`merve/docvqa-media-labeled-moondream` → `merve/docvqa-media-judged-ensemble` →
`merve/rfdetr-docvqa-moondream`. Every dataset push includes an auto-generated
box-overlay gallery (`viz/` + README), so boxes never need re-rendering.

## One-time setup
```bash
hf buckets create vision-agent-runs          # run bucket (idempotent)
git push origin multimodel-jobs              # jobs clone this ref
```

## Smoke test (≈20 images, cheap)
```bash
BUCKET="-v hf://buckets/merve/vision-agent-runs:/data"
REF="-e REPO_REF=multimodel-jobs"
N="--max-samples 20"

# 1 — label
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN $BUCKET $REF -d \
  jobs/label_moondream.py -- $N

# 2 — judges (after stage 1 SUCCEEDED)
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN $BUCKET $REF -d \
  jobs/judge_one.py -- --model google/gemma-4-E4B-it \
  --out /data/docvqa-moondream/verdicts_gemma.parquet $N
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN $BUCKET $REF -d \
  jobs/judge_one.py -- --model LiquidAI/LFM2.5-VL-1.6B \
  --out /data/docvqa-moondream/verdicts_lfm.parquet $N

# 3 — merge (after both judges SUCCEEDED)
hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN $BUCKET $REF -d \
  jobs/merge_judges.py -- \
  --verdicts "google/gemma-4-E4B-it::/data/docvqa-moondream/verdicts_gemma.parquet" \
  --verdicts "LiquidAI/LFM2.5-VL-1.6B::/data/docvqa-moondream/verdicts_lfm.parquet" \
  --min-agree 1 $N
```

## Full run
Drop `--max-samples` from stages 1–3 (defaults to the whole `test` split), then:
```bash
hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN $REF --timeout 6h -d \
  jobs/train_rfdetr_job.py -- --epochs 10 --batch-size 8
```

## Babysitting
```bash
hf jobs ps                      # list running jobs + ids
hf jobs logs <job_id>           # fetch logs (add -f to follow)
hf jobs inspect <job_id>        # status / exit code
hf jobs cancel <job_id>         # stop
hf buckets ls merve/vision-agent-runs/docvqa-moondream -h   # check artifacts
```
Each stage prints `STAGE N DONE` on success — grep logs for that plus
`Traceback|Error|OOM|Killed` to catch failures.
