# Road-sign detection

Train a closed-set road-sign detector from VLM pseudo-labels — the e2e pipeline
with the **labelling stage already done**, run by the
[`vision-e2e-runner`](../../.claude/agents/vision-e2e-runner.md) subagent.

## Goal
Detect 21 road-sign / traffic-light classes in dashcam-style images. Labelling
was done with a VLM (Qwen); this example covers the **judge → train** half, and
exists mainly to exercise the **human-approved judge descriptions** gate: the
class names are terse and coded (`do_not_turn_l`, `t_intersection_l`,
`ped_zebra_cross`, `enter_left_lane`, …), so a human reviewing the meta-prompt's
expansion of each into a clear visual definition materially improves judging.

## Data
- **Original (human GT):** `Francesco/road-signs-6ih4y` (RF100; train/val/test)
- **Labelled (VLM, ready):** `merve/roadsign-labeled-qwen` (split `train`, 1.4K
  rows) — has `detections` + `detections_overlay`; also carries the original
  human `objects` ClassLabel column.

## Classes
The 21 classes are the dataset's `objects.category` ClassLabel names **minus the
`road-signs` supercategory** at index 0 (an RF100 convention):
```
bus_stop, do_not_enter, do_not_stop, do_not_turn_l, do_not_turn_r,
do_not_u_turn, enter_left_lane, green_light, left_right_lane, no_parking,
parking, ped_crossing, ped_zebra_cross, railway_crossing, red_light, stop,
t_intersection_l, traffic_light, u_turn, warning, yellow_light
```

## Models per role
| Role | Model | Family | Size | Where |
|---|---|---|---|---|
| Orchestrator | the driving agent | — | — | issues calls, never labels/judges |
| Labeller | `Qwen/Qwen3.5-9B` | Qwen | 9.65B | HF router (already run) |
| Judge A | `google/gemma-4-E4B-it` | Google | 8.0B | `l40sx1` (the heavy judge — the long pole) |
| Judge B | `LiquidAI/LFM2.5-VL-1.6B` | Liquid | 1.6B | `l4x1` (small judge) |
| Train | `Roboflow/rf-detr-large` | — | — | `l40sx1` (large model, 20–30 epochs) |

## Commands
The **only human-approval gate** is the judge descriptions. Generate them as a
separate step, review, then judge with the approved file (it is used verbatim —
no regeneration):

The descriptions are the only human gate; everything after runs unattended.
Stages 2–4 emit **both** ensemble policies (`agree1` recall, `agree2` precision)
as separate repos — see [emit-both-agree convention](../../agents.md).

```bash
# 1 — generate judge descriptions  → review/edit descriptions.json  ← HUMAN GATE
uv run python -m workflows.gen_descriptions \
  --source merve/roadsign-labeled-qwen --drop road-signs \
  --backend openai --base-url https://router.huggingface.co/v1 \
  --model Qwen/Qwen3.5-9B --output descriptions.json   # the approved file lives in this folder

# 2 — judge into BOTH policies (each pushes with a bbox-overlay gallery)
for AG in 1 2; do
  uv run python -m workflows.vlm_judge --source merve/roadsign-labeled-qwen \
    --output merve/roadsign-judged-ensemble-agree$AG --push-to-hub \
    --judges "google/gemma-4-E4B-it,LiquidAI/LFM2.5-VL-1.6B" \
    --backend openai --base-url https://router.huggingface.co/v1 \
    --class-descriptions descriptions.json --min-agree $AG --split train
done

# 3 — strip the human-GT objects column (it shadows our detections) from each
for AG in 1 2; do
  HF_TOKEN=$(hf auth token) python3 jobs/strip_objects.py \
    merve/roadsign-judged-ensemble-agree$AG merve/roadsign-judged-ensemble-agree$AG-trainready
done

# 4 — train RF-DETR on each. This data trains far better with --no-augment.
for AG in 1 2; do
  uv run --extra train python -m workflows.train_rfdetr \
    --source merve/roadsign-judged-ensemble-agree$AG-trainready --val-split test \
    --model Roboflow/rf-detr-large --epochs 30 --batch-size 8 --no-augment \
    --hub-model-id merve/rfdetr-roadsign-agree$AG-large-noaug
done
```

## Outputs
- **Labeled:** `merve/roadsign-labeled-qwen`
- **Judged:** `merve/roadsign-judged-ensemble-agree1` / `-agree2` (+ `…-trainready`)
  — each with an auto bbox-overlay gallery on its dataset page
- **Models:** `merve/rfdetr-roadsign-agree1-large-noaug` (best) /
  `-agree2-large-noaug`

## Results
mAP on the held-out split, vs the **VLM-judged** labels (not human GT — run
`jobs/eval_vs_gt.py` against `Francesco/road-signs-6ih4y` test for the true number):

| config | agree1 mAP | agree1 mAP@50 | agree2 mAP |
|---|---|---|---|
| **rf-detr-large, 30 ep, no-aug** | **0.685** | 0.772 | 0.623 |
| rf-detr-base, 10 ep, +aug (baseline) | 0.298 | 0.335 | 0.282 |

The 0.30 → 0.68 jump came entirely from three hparams: **base→large, 10→30 epochs,
augmentation off** (lr/batch/cosine schedule were already matched). `agree1`
(higher recall) beat `agree2` here. Strong classes: `do_not_enter` 0.93,
`ped_crossing` 0.87, `stop` 0.82; weakest `yellow_light` 0.18 (amber-vs-red, the
case the judges also flagged).

## Notes / gotchas
- **`gen_descriptions` is the only approval gate.** `judge_labels` regenerates
  descriptions *only* when none are supplied; passing `--class-descriptions`
  pins the human-approved text for the whole run.
- **Strip `objects` before training** — the RF-DETR trainer prioritizes a human
  `objects` column over our VLM `detections`, and non-0-based ids crash it.
- **Augmentation hurts here — train with `--no-augment`.** The clean, centered
  sign crops don't benefit from heavy augmentation (see Results: 0.685 no-aug vs
  0.30 with aug). Augmentation is use-case dependent — assess per dataset.
- **VLM labels do contain errors** worth catching (e.g. a `yellow_light` sign
  labelled `red_light`, a `red_light` labelled `traffic_light`) — the point of
  the judge pass.
