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
| Judge A | `google/gemma-4-E4B-it` | Google | 8.0B | HF router / `l4x1` |
| Judge B | `LiquidAI/LFM2.5-VL-1.6B` | Liquid | 1.6B | HF router / `l4x1` |
| Train | `Roboflow/rf-detr-base` | — | — | local GPU / `l40sx1` |

## Commands
The **only human-approval gate** is the judge descriptions. Generate them as a
separate step, review, then judge with the approved file (it is used verbatim —
no regeneration):

```bash
# 1 — generate judge descriptions  → review/edit descriptions.json  ← HUMAN GATE
uv run python -m workflows.gen_descriptions \
  --source merve/roadsign-labeled-qwen --drop road-signs \
  --backend openai --base-url https://router.huggingface.co/v1 \
  --model Qwen/Qwen3.5-9B --output descriptions.json

# 2 — judge with the approved descriptions (ensemble, both must agree)
uv run python -m workflows.vlm_judge \
  --source merve/roadsign-labeled-qwen \
  --output merve/roadsign-judged-ensemble --push-to-hub \
  --judges "google/gemma-4-E4B-it,LiquidAI/LFM2.5-VL-1.6B" \
  --backend openai --base-url https://router.huggingface.co/v1 \
  --class-descriptions descriptions.json --min-agree 2 --split train

# 3 — strip the human-GT objects column (it shadows our detections in training)
HF_TOKEN=$(hf auth token) python3 jobs/strip_objects.py \
  merve/roadsign-judged-ensemble merve/roadsign-judged-ensemble-trainready

# 4 — train RF-DETR (val split grouped by image to avoid leakage)
uv run --extra train python -m workflows.train_rfdetr \
  --source merve/roadsign-judged-ensemble-trainready --val-split test \
  --model Roboflow/rf-detr-base --epochs 10 --batch-size 8 \
  --hub-model-id merve/rfdetr-roadsign --output-dir checkpoints/rfdetr-roadsign
```

## Outputs
- Labeled:  `merve/roadsign-labeled-qwen` (done)
- Judged:   `merve/roadsign-judged-ensemble`
- Model:    `merve/rfdetr-roadsign`

## Notes / gotchas
- **`gen_descriptions` is the only approval gate.** `judge_labels` regenerates
  descriptions *only* when none are supplied; passing `--class-descriptions`
  pins the human-approved text for the whole run.
- **Strip `objects` before training** — the RF-DETR trainer prioritizes a human
  `objects` column over our VLM `detections`, and non-0-based ids crash it.
- **VLM labels do contain errors** worth catching (e.g. a `yellow_light` sign
  labelled `red_light`, a `red_light` labelled `traffic_light`) — the point of
  the judge pass.
