"""Re-run the judge stage at *zero* confidence.

The judge score has proven unreliable (near-hallucinated), so for this first
stage we keep **every** detection (``threshold=0.0``) and only record the
verdicts for later analysis / a future multi-judge ensemble. Nothing is
dropped based on the judge score.

Judge: Qwen3-VL-4B served locally via llama-server on :8084.
Input:  merve/docvqa-media-labeled  (split: test)
Output: merve/docvqa-media-judged   (all detections kept + judge_verdicts)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

HF_TOKEN = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
os.environ["HF_TOKEN"] = HF_TOKEN

LABELED_ID = "merve/docvqa-media-labeled"
JUDGED_ID = "merve/docvqa-media-judged"

JUDGE_MODEL = "Qwen3-VL-4B-Instruct-Q8_0.gguf"
JUDGE_BASE_URL = "http://localhost:8084/v1"
JUDGE_API_KEY = "sk-local-judge"

from workflows.vlm_judge import judge_labels

print("=" * 60)
print("Re-judging at ZERO confidence (threshold=0.0 → keep all dets)")
print(f"  in:  {LABELED_ID} [test]")
print(f"  out: {JUDGED_ID}")
print(f"  judge: {JUDGE_MODEL} @ {JUDGE_BASE_URL}")
print("=" * 60)

judge_labels(
    source=LABELED_ID,
    output=JUDGED_ID,
    model_id=JUDGE_MODEL,
    threshold=0.0,
    backend="openai",
    base_url=JUDGE_BASE_URL,
    api_key=JUDGE_API_KEY,
    image_column="image",
    detections_column="detections",
    split="test",
    push_to_hub=True,
    hf_token=HF_TOKEN,
)

print("\nDONE — zero-confidence judged dataset pushed.")
