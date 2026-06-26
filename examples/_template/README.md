# <Use case name>

> Copy this folder to `examples/<your-use-case>/` and fill in the sections.
> Delete this quote block when done, and add a row to `examples/README.md`'s
> index.

## Goal
<What is being detected, in what imagery, and why this pipeline fits.>

## Data
- **Source:** `<hf-dataset-or-local-dir>` (config `<...>`, split `<...>`)
- **Preprocessing:** <dedupe by key / resize / filtering — or "none">

## Classes
```
<comma,separated,class,list passed to the labeller>
```

## Models per role
| Role | Model | Family | Size | Where |
|---|---|---|---|---|
| Orchestrator | <model> | — | — | drives the run, never labels/judges |
| Labeller | <model> | <family> | <size> | <remote router / local server> |
| Judge A | <model> | <family> | <size> | <...> |
| Judge B | <model> | <family> | <size> | <...> |

(Labeller strictly larger than every judge; all different families — see
[`../../agents.md`](../../agents.md).)

## Commands
```bash
# 1 — label
# 2 — judge (one per judge)
# 3 — merge / filter
# 4 — train
```

## Outputs
- Labeled:  `<hub-id>`
- Judged:   `<hub-id>`
- Model:    `<hub-id>`

## Notes / gotchas
- <anything non-obvious that bit during the run>
