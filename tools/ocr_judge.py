"""Pairwise OCR quality evaluation using a VLM as judge.

Adapted from `uv-scripts/ocr (ocr-vllm-judge.py)
<https://huggingface.co/datasets/uv-scripts/ocr>`_.
Given a document image and two OCR outputs, a VLM judge decides which is
better on faithfulness, completeness, accuracy, reading order, and
formatting.  Includes ELO computation for ranking multiple models.

Supports two backends (see :mod:`tools.vlm_client`):

- ``"openai"``  — HF Inference Providers, vLLM server, llama-server
- ``"transformers"`` — local GPU with ``transformers``

CLI usage::

    python -m tools.ocr_judge image.png \\
        --text-a "OCR output from model A" \\
        --text-b "OCR output from model B"
"""

from __future__ import annotations

import argparse
import json
import re

from PIL import Image

from .utils import load_image
from .vlm_client import run_vlm

DEFAULT_JUDGE = "Qwen/Qwen2.5-VL-7B-Instruct"

PAIRWISE_PROMPT = """\
You are an expert OCR quality evaluator. You are given a document image and \
TWO OCR outputs (A and B) extracted from that same image.

Compare them and decide which extraction is better overall.

Evaluation criteria (in priority order):
1. Faithfulness — only text from the document, no commentary
2. Completeness — all visible text captured
3. Accuracy — correct characters, no hallucinations
4. Reading order — natural flow
5. Formatting — clean structure (plain accurate > fancy incomplete)

If both are equal, respond with "tie".

Output A:
---
{text_a}
---

Output B:
---
{text_b}
---

Respond with JSON only: {{"winner": "A"|"B"|"tie", "reason": "brief explanation"}}"""

INITIAL_ELO = 1500
K = 32


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------

def _parse_verdict(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
        winner = result.get("winner", "tie").upper().strip()
        if winner not in ("A", "B", "TIE"):
            winner = "tie"
        return {"winner": winner, "reason": result.get("reason", "")}
    except json.JSONDecodeError:
        match = re.search(r'"winner"\s*:\s*"([ABab]|tie)"', text, re.IGNORECASE)
        if match:
            return {"winner": match.group(1).upper(), "reason": ""}
        return {"winner": "tie", "reason": "parse error"}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def ocr_judge(
    image: str | Image.Image,
    text_a: str,
    text_b: str,
    model_id: str = DEFAULT_JUDGE,
    backend: str = "transformers",
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Compare two OCR outputs for the same image.

    Parameters
    ----------
    image : str or PIL.Image
        The source document image.
    text_a, text_b : str
        OCR outputs from two different models.
    model_id : str
        VLM used as judge.
    backend : ``"openai"`` | ``"transformers"``
        Inference backend.
    base_url : str, optional
        API endpoint for the ``openai`` backend.
    api_key : str, optional
        API key / HF token for the ``openai`` backend.

    Returns
    -------
    dict
        ``{"winner": "A"|"B"|"tie", "reason": str}``
    """
    image = load_image(image)

    prompt = PAIRWISE_PROMPT.format(
        text_a=text_a[:3000], text_b=text_b[:3000],
    )
    response = run_vlm(
        image, prompt, model_id,
        backend=backend, base_url=base_url, api_key=api_key,
        max_tokens=512,
    )
    return _parse_verdict(response)


def update_elo(
    elo_a: float, elo_b: float, winner: str,
) -> tuple[float, float]:
    """Update ELO ratings given a pairwise outcome."""
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    score_a = {"A": 1.0, "B": 0.0}.get(winner, 0.5)
    elo_a += K * (score_a - expected_a)
    elo_b += K * ((1 - score_a) - (1 - expected_a))
    return elo_a, elo_b


def elo_leaderboard(
    results: list[dict],
    model_names: list[str],
) -> dict[str, float]:
    """Compute ELO ratings from a list of comparison results.

    Each result should have ``model_a``, ``model_b``, and ``winner``
    keys (as returned by :func:`ocr_judge` wrapped with model info).

    Returns a dict mapping model name -> ELO rating.
    """
    elo = {m: INITIAL_ELO for m in model_names}
    for r in results:
        a, b, w = r["model_a"], r["model_b"], r.get("winner", "tie")
        elo[a], elo[b] = update_elo(elo[a], elo[b], w)
    return dict(sorted(elo.items(), key=lambda x: -x[1]))


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise OCR quality judge")
    parser.add_argument("image", help="Document image path or URL")
    parser.add_argument("--text-a", required=True)
    parser.add_argument("--text-b", required=True)
    parser.add_argument("--model", default=DEFAULT_JUDGE)
    parser.add_argument("--backend", default="transformers",
                        choices=["openai", "transformers"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    verdict = ocr_judge(
        args.image, args.text_a, args.text_b, model_id=args.model,
        backend=args.backend, base_url=args.base_url, api_key=args.api_key,
    )
    print(json.dumps(verdict, indent=2))
