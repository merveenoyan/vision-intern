# Examples — worked use cases

Each subfolder here is **one use case** built with this toolkit: a concrete run
of the label → judge → train pipeline (or just a subset of it) on a specific
dataset, with specific classes and a specific model per role. They are recipes
to **read and imitate**, not library code — nothing in the rest of the repo
imports from here.

> **For an orchestrating agent:** before building a new pipeline, list this
> folder and skim the closest existing example — it shows the class set, the
> model-per-role wiring, the exact commands, and the gotchas already hit. Copy
> `_template/` to start a new one.

## Convention

```
examples/
  _template/         # skeleton — copy this to start a new use case
  <use-case>/        # one folder per use case, kebab-case
    README.md        # REQUIRED: goal, data, classes, models, commands, outputs
    *.py             # OPTIONAL: scripts specific to this use case only
```

Each use-case `README.md` should answer, in order:

1. **Goal** — what is being detected, in what kind of imagery, and why.
2. **Data** — source dataset(s)/dir, split, any dedupe or preprocessing.
3. **Classes** — the exact class list passed to the labeller.
4. **Models per role** — orchestrator / labeller / judge(s), with sizes and
   why each was chosen (recall the [role-separation rules](../agents.md) —
   labeller strictly larger than every judge, all different families).
5. **Commands** — the exact CLI / job invocations run, in order.
6. **Outputs** — the Hub datasets and model produced (labeled → judged → model).
7. **Notes / gotchas** — anything non-obvious that bit during the run.

Keep the *generic* mechanics in `README.md` / `jobs/README.md`; put only the
**use-case-specific choices** here.

## Index

| Use case | Goal | Status |
|---|---|---|
| [`docvqa-media`](docvqa-media/) | Detect media regions (table/image/chart/…) in scanned document pages | done |
| [`roadsign-detection`](roadsign-detection/) | Detect 21 road-sign / traffic-light classes; judge → train (best mAP 0.685) | done |

Some use cases are run by the [`vision-e2e-runner`](../.claude/agents/vision-e2e-runner.md)
subagent, which executes the pipeline autonomously except for the one human gate:
approving the generated judge descriptions.
