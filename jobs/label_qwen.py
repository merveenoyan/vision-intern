# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai>=1.40",
#   "pillow",
#   "datasets>=3.0",
#   "huggingface_hub>=0.26",
#   "requests",
#   "numpy",
# ]
# ///
"""HF Job — STAGE 1 (Qwen variant): label DocVQA media with Qwen3.5-9B.

Qwen uses the prompt-based ``bbox_2d`` (0-1000) detection convention, which
gives tight per-object boxes — unlike moondream's page-spanning output on
scanned documents. Qwen3.5-9B has live HF Inference Providers, so we label
through the HF router (``openai`` backend) on a cheap CPU job instead of loading
9.6B on a GPU. Reuses ``workflows.vlm_label.label_dataset`` (viz-on-push built
in).

    hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN \
      -e REPO_REF=multimodel-jobs jobs/label_qwen.py -- --max-samples 20
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = os.environ.get("REPO_URL", "https://github.com/merveenoyan/vision-intern.git")
REPO_REF = os.environ.get("REPO_REF", "multimodel-jobs")
# Set REPO_DIR to a local checkout (e.g. `REPO_DIR=$(pwd)`) to run against your
# working tree; left unset it clones REPO_REF (the default on HF Jobs).
REPO_DIR = Path(os.environ.get("REPO_DIR", "/tmp/vision-intern"))
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF,
                    REPO_URL, str(REPO_DIR)], check=True)
sys.path.insert(0, str(REPO_DIR))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="lmms-lab/DocVQA")
    p.add_argument("--dataset-config", default="DocVQA")
    p.add_argument("--split", default="test")
    p.add_argument("--classes", default="table,image,chart,diagram,figure")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--output", default="merve/docvqa-media-labeled-qwen")
    p.add_argument("--backend", default="openai", choices=["openai", "transformers"])
    p.add_argument("--base-url", default=None, help="Defaults to the HF router")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--dedupe", action="store_true",
                   help="Collapse repeated images to one row before labelling "
                        "(DocVQA has ~3.4 question rows per page).")
    p.add_argument("--dedupe-key-columns", default="docId",
                   help="Comma-separated column(s) identifying the same image "
                        "for --dedupe (default: docId; '' = image-content hash).")
    args = p.parse_args()

    from workflows.vlm_label import label_dataset

    token = os.environ["HF_TOKEN"]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    print(f"Labelling with {args.model} via {args.backend} backend, classes {classes}")

    label_dataset(
        source=args.source,
        classes=classes,
        output=args.output,
        model_id=args.model,
        backend=args.backend,
        base_url=args.base_url,
        api_key=token,
        image_column="image",
        split=args.split,
        max_samples=args.max_samples,
        push_to_hub=True,
        hf_token=token,
        dataset_config=args.dataset_config,
        dedupe=args.dedupe,
        dedupe_key_columns=(
            [c.strip() for c in args.dedupe_key_columns.split(",") if c.strip()]
            if args.dedupe_key_columns else None
        ),
    )
    print("STAGE 1 (qwen) DONE")


if __name__ == "__main__":
    main()
