# Handoff — DocVQA media detection, **scaled-up round (val+test, self-made splits)**

Continuation of the chart/image/signature detection project on document pages.
Pipeline is unchanged in method (VLM-label → ensemble-judge → train RF-DETR); this
round **doubled the labelled data** by adding DocVQA `validation` to `test`,
deduping across the union, and cutting our **own** train/val/test splits. Read
top-to-bottom before launching anything.

---

## 1. Where the project stands (current round)

**Source:** `lmms-lab/DocVQA`, config `DocVQA`. That config has **no `train`
split** — only `validation` (5,349 rows → 1,286 unique pages) and `test`
(5,188 → 1,287). We labelled **both**, concatenated, and **cross-deduped by
image content** → **2,572 unique pages / 2,062 boxes** (exactly 1 page
overlapped val/test — they're essentially disjoint). This is ~2× the prior
round's 1,287 pages.

**Classes:** `chart, image, signature` (tables still excluded).

**Datasets produced this round:**
| Dataset | What |
|---|---|
| `merve/docvqa-media3-labeled-qwen-trainval` | combined+deduped labelled (2,572 pages / 2,062 boxes), split `test` |
| `merve/docvqa-media3-judged-trainval-agree1` | judged `min-agree 1` → 1,648 boxes |
| `merve/docvqa-media3-judged-trainval-agree2` | judged `min-agree 2` → 947 boxes |
| `merve/docvqa-media3-judged-splits-agree1` | DatasetDict train/val/test = 1800/386/386 pages, 1171/229/248 boxes |
| `merve/docvqa-media3-judged-splits-agree2` | same page partition, 675/132/140 boxes |
| `merve/rfdetr-docvqa-media3-trainval-agree1-medium` | RF-DETR-**medium**, agree-1, 30 ep |
| `merve/rfdetr-docvqa-media3-trainval-agree2-medium` | RF-DETR-medium, agree-2, 30 ep |

The two split datasets share an **identical page partition** (fixed seed +
`docId` grouping in `make_splits.py`), so agree-1 vs agree-2 differ only in box
density — a clean read on the `--min-agree` dial.

---

## 2. Results — big jump, but read the caveats

**vs prior baselines** (prior round: RF-DETR base/large, ~1,287 pages, tiny noisy val):
- prior best: mAP ~0.004, mAP@50 ~0.019, mAR@100 ~0.32.

**This round, RF-DETR-medium, eval on the held-out `test` split (386 pages, untouched):**
| Metric | agree-1-medium | agree-2-medium |
|---|---|---|
| mAP | **0.225** | 0.173 |
| mAP@50 | **0.344** | 0.248 |
| mAP@75 | 0.223 | 0.200 |
| mAR@100 | 0.572 | **0.645** |
| chart map/mar100 | 0.124 / 0.596 | 0.128 / **0.768** |
| image map/mar100 | **0.447** / 0.783 | 0.365 / **0.842** |
| signature map/mar100 | **0.105** / 0.337 | 0.027 / 0.326 |

(Trainer's per-epoch eval on the **validation** split read slightly higher,
~0.27–0.28 mAP, and ranked agree-2 ≥ agree-1 — that was val-split noise. Trust
the test numbers.)

Takeaways:
1. **~50× mAP jump** (0.004 → 0.17–0.23) and 2× recall. Levers that landed
   together: 2× data, a right-sized model (medium), 30 epochs (was 10), and a
   **fixed leak-free test split** instead of the old tiny val.
2. **agree-1 vs agree-2 is a precision/recall trade**, not a clear win:
   agree-1 higher mAP/precision, agree-2 higher recall. No reason to prefer
   agree-1's extra single-judge boxes on quality grounds.
3. **Signature is the weak class** (mAP 0.03–0.10) — rare, as predicted.
   Infographic pages won't add many; needs a targeted signature source.
4. **GT is VLM pseudo-labels**, so these metrics measure agreement with the
   judged labels, not human truth. **Vibe-check the overlays** (see §4).

---

## 3. How this round was run (scripts + commands)

Labelling is still **local + atomic** (Qwen on the router → free/resumable),
one `docId`-keyed checkpoint per split. New glue scripts handle the combine and
the self-made splits. Full chain:

```bash
# 1. Label each split locally into its OWN docId-keyed checkpoint (free, resumable)
HF_TOKEN=$(hf auth token) python3 jobs/label_local.py \
  --source lmms-lab/DocVQA --dataset-config DocVQA --split test \
  --checkpoint /tmp/docvqa-media3-test-dets.jsonl
HF_TOKEN=$(hf auth token) python3 jobs/label_local.py \
  --source lmms-lab/DocVQA --dataset-config DocVQA --split validation \
  --checkpoint /tmp/docvqa-media3-val-dets.jsonl

# 2. NEW: combine both checkpoints, cross-dedupe by image content, push one labelled set
HF_TOKEN=$(hf auth token) python3 jobs/build_combined_labeled.py \
  --input test::/tmp/docvqa-media3-test-dets.jsonl \
  --input validation::/tmp/docvqa-media3-val-dets.jsonl \
  --output merve/docvqa-media3-labeled-qwen-trainval

# 3. Judge x2 on l4x1 (GPU), then merge at BOTH thresholds (CPU). Verdicts in bucket /data.
hf jobs uv run -d --flavor l4x1 --secrets HF_TOKEN \
  -v hf://buckets/merve/vision-agent-runs:/data -e REPO_REF=multimodel-jobs \
  jobs/judge_one.py -- --model google/gemma-4-E4B-it \
  --dataset merve/docvqa-media3-labeled-qwen-trainval --split test \
  --out /data/docvqa-media3-trainval/verdicts_gemma.parquet
# ...repeat for LiquidAI/LFM2.5-VL-1.6B → verdicts_lfm.parquet
for AG in 1 2; do
  hf jobs uv run -d --flavor cpu-upgrade --secrets HF_TOKEN \
    -v hf://buckets/merve/vision-agent-runs:/data -e REPO_REF=multimodel-jobs \
    jobs/merge_judges.py -- \
    --dataset merve/docvqa-media3-labeled-qwen-trainval --split test \
    --verdicts "google/gemma-4-E4B-it::/data/docvqa-media3-trainval/verdicts_gemma.parquet" \
    --verdicts "LiquidAI/LFM2.5-VL-1.6B::/data/docvqa-media3-trainval/verdicts_lfm.parquet" \
    --min-agree $AG --output merve/docvqa-media3-judged-trainval-agree$AG
done

# 4. NEW: cut leak-free train/val/test (grouped by docId, fixed seed → identical partition)
for AG in 1 2; do
  HF_TOKEN=$(hf auth token) python3 jobs/make_splits.py \
    --source merve/docvqa-media3-judged-trainval-agree$AG --split test \
    --output merve/docvqa-media3-judged-splits-agree$AG
done

# 5. Train (l4x1 is plenty for medium; batch 16, 30 epochs). Use the EXPLICIT splits.
for AG in 1 2; do
  hf jobs uv run -d --flavor l4x1 --secrets HF_TOKEN --timeout 6h \
    -e REPO_REF=multimodel-jobs \
    jobs/train_rfdetr_job.py -- \
    --source merve/docvqa-media3-judged-splits-agree$AG \
    --train-split train --val-split validation \
    --model Roboflow/rf-detr-medium --epochs 30 --batch-size 16 \
    --hub-model-id merve/rfdetr-docvqa-media3-trainval-agree$AG-medium
done
```

---

## 4. Eval + eyeball the predictions

`jobs/eval_overlay.py` (NEW) runs a pushed model on any split **locally on
GPU**, reports torchmetrics mAP/mAR against the pseudo-labels, and writes one
side-by-side PNG per page (left = GT, right = prediction + confidence):

```bash
HF_TOKEN=$(hf auth token) python3 jobs/eval_overlay.py \
  --model merve/rfdetr-docvqa-media3-trainval-agree2-medium \
  --source merve/docvqa-media3-judged-splits-agree2 --split test \
  --out-dir viz_test_predictions/agree2-medium
```
Overlays for this round: `viz_test_predictions/agree{1,2}-medium/` (386 each;
filename ends `_Npreds` = #boxes the model fired ≥0.3 conf).

---

## 5. Infra notes / gotchas (cost us time before)

1. **`hf jobs uv run` streams attached** — pass **`-d`/`--detach`** or it blocks
   the next submission. Use `-d` for fan-out.
2. **`hf jobs inspect` JSON has multiple `stage` fields** — naive grep for the
   first one can misread a long-running job as ERROR. Confirm via the job's
   **logs** (look for `STAGE x DONE`), not just the stage string.
3. **Right-size the GPU/batch to the model.** `rf-detr-medium` trains fine on
   `l4x1` (24 GB) at batch 16 in ~15–20 min for 1,800 imgs/30 ep. `l40sx1` and
   batch 8 (the old job defaults) are oversized/slow for medium — don't copy
   them blindly.
4. **Verdicts/parquets written to the mounted bucket `/data` persist** (it's
   not the ephemeral job disk). `STAGE 2 judge DONE` prints only after the
   write, so that line = the file is safe on the bucket.
5. You are **not the only agent** hitting `hf jobs` — don't cancel a job you
   didn't launch just because it looks related. Match by job ID.

## 6. Lessons that still apply (don't relearn them)

1. **Label locally when the model is on the router** (free, parallel, atomic).
   [[labelling-atomic-local]]. Atomic checkpoint+resume is local-only; on a Job
   you must push.
2. **One checkpoint per (config, split); dedupe across the union later.**
   `docId` is unique only within a config; `build_combined_labeled.py` does the
   cross-split dedupe by **image hash** (never collides).
3. **Overlay Arrow bug:** encode overlays as `{"bytes": png, "path": None}` then
   `cast_column(HFImage())` — never `add_column` raw PIL.
4. **Keep labeller (`tools/vlm_detect.py`) and judge (`workflows/vlm_judge.py`
   `_CLASS_HINTS`) class definitions in sync.**
5. **`--min-agree` is the recall/precision dial.** This round confirmed it's a
   *trade* at this data scale, not a strict win either way.

## 7. Next levers (priority order)

1. **Fix signature recall** — it's the floor on mAP. Needs a signature-rich
   source (the doc pages here just don't have many); or label at higher input
   res (`tools/vlm_client.py:_MAX_DIMENSION = 1280`) since signatures are tiny.
2. **More epochs / larger model now that data is bigger** — medium at 30 ep may
   still be under-trained on charts (chart mAP ~0.12 despite mAR ~0.6–0.77,
   i.e. recall ok, localization/precision weak).
3. **Add InfographicVQA** (config `InfographicVQA`, test+val, chart/image-dense)
   via the same combine flow — own checkpoints, then `build_combined_labeled.py`
   with all four `split::checkpoint` inputs.
4. **Spot-check pseudo-label quality from the overlays** before trusting mAP;
   if GT is noisy, real accuracy is higher/lower than the numbers suggest.
