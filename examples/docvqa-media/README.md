# DocVQA media-region detection

The reference end-to-end run for this toolkit: auto-label media regions on
scanned document pages, verify them with a small judge ensemble, and fine-tune
RF-DETR on the result — all on HF Jobs, one model per role.

## Goal
Detect **media regions** (tables, embedded images, charts, diagrams, figures)
in scanned document pages, so a fast closed-set detector can be trained without
any hand-labeling. DocVQA pages have these regions but no detection annotations,
so a VLM proposes boxes and an ensemble of smaller VLMs verifies them.

## Data
- **Source:** `lmms-lab/DocVQA` (config `DocVQA`, split `test`)
- **Preprocessing:** dedupe by `docId` — DocVQA has ~3.4 question rows per page,
  so `--dedupe` collapses 1000 rows → ~296 unique pages before labelling, so
  each page is labelled once.

## Classes
```
table,image,chart,diagram,figure
```

## Models per role
| Role | Model | Family | Size | Where |
|---|---|---|---|---|
| Orchestrator | (the driving agent) | — | — | issues the job calls, never labels/judges |
| Labeller | `Qwen/Qwen3.5-9B` | Qwen | 9.65B | HF Inference Providers (router), CPU job |
| Judge A | `google/gemma-4-E4B-it` | Google | 8.0B | `l4x1` GPU job |
| Judge B | `LiquidAI/LFM2.5-VL-1.6B` | Liquid | 1.6B | `l4x1` GPU job |
| Train | `Roboflow/rf-detr-large` | — | — | `l40sx1` GPU job |

Why this wiring: the labeller is the largest worker and the two judges are from
different families (uncorrelated errors). Qwen's prompt-based `bbox_2d` (0–1000)
convention gives **tight** per-object boxes (median area ≈ 0.09 of the page),
where a detection specialist like moondream tends to return page-spanning boxes
on scanned documents (median area ≈ 0.98) — see `jobs/README.md`.

## Commands
The exact, ordered `hf jobs uv run` invocations (label → 2 judges in parallel →
merge → train), the bucket setup, smoke-test flags, and babysitting commands
live in **[`../../jobs/README.md`](../../jobs/README.md)** — that *is* this
example's runbook. The use-case-specific choices captured here are the dataset,
the dedupe, the class set, and the model-per-role table above.

Verification gates at the merge step: `--min-agree 2` (both judges must vote
`correct`) and `--max-area-frac 0.9` (a geometric guard dropping page-spanning
boxes) — the single-judge VLM score is uncalibrated, so keeps are gated by
these cheap non-vibe checks, recorded per box in `judge_verdicts`.

## Outputs
- Labeled:  `merve/docvqa-media-labeled-qwen`
- Judged:   `merve/docvqa-media-judged-ensemble`
- Model:    `merve/rfdetr-docvqa-qwen`

Every dataset push includes an auto-generated box-overlay gallery (`viz/` +
README), so boxes never need re-rendering to inspect.

## Notes / gotchas
- **Give jobs a generous `--timeout`.** Labelling 1000 rows and the gemma judge
  over 1000 rows both run past the default job timeout; the final push completes
  but the job is then flagged ERROR. `--timeout 3h` avoids the false failure.
- **Strip the inherited `objects` column before training.** DocVQA rows carry a
  human-GT `objects` column; the RF-DETR trainer prioritizes `objects` over our
  VLM `detections`, so it must be dropped first (`jobs/strip_objects.py`).
- **Group the train/val split by image.** Pages can repeat; an image-grouped
  split (`tools.dataset_utils.grouped_train_val_split`) keeps a page out of both
  splits, otherwise mAP is inflated by leakage.
