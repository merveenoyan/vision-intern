"""Document OCR — convert images to markdown using Vision Language Models.

Adapted from `uv-scripts/ocr <https://huggingface.co/datasets/uv-scripts/ocr>`_.
Supports multiple OCR models and task modes (text, formula, table).

Supports two backends (see :mod:`tools.vlm_client`):

- ``"openai"``  — HF Inference Providers, vLLM server, llama-server
- ``"transformers"`` — local GPU with ``transformers``

CLI usage::

    python -m tools.document_ocr image.png
    python -m tools.document_ocr image.png --task table
    python -m tools.document_ocr image.png --backend openai
"""

from __future__ import annotations

import argparse
from typing import Any

from PIL import Image

from .utils import load_image
from .vlm_client import run_vlm

DEFAULT_MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"

TASK_PROMPTS: dict[str, dict[str, str]] = {
    "PaddlePaddle/PaddleOCR-VL-1.6": {
        "ocr": "OCR:",
        "formula": "Formula Recognition:",
        "table": "Table Recognition:",
    },
    "zai-org/GLM-OCR": {
        "ocr": "Text Recognition:",
        "formula": "Formula Recognition:",
        "table": "Table Recognition:",
    },
    "_default": {
        "ocr": "Perform OCR on this image. Return the text in markdown format.",
        "formula": "Recognize the mathematical formula in this image. Return LaTeX.",
        "table": "Recognize the table in this image. Return it in markdown format.",
    },
}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def document_ocr(
    image: str | Image.Image,
    model_id: str = DEFAULT_MODEL,
    task: str = "ocr",
    max_new_tokens: int = 4096,
    backend: str = "transformers",
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Extract text from *image* as markdown.

    Parameters
    ----------
    image : str or PIL.Image
        Path, URL, or PIL Image of a document / page.
    model_id : str
        Hugging Face model ID.
    task : str
        One of ``"ocr"`` (text), ``"formula"`` (LaTeX), ``"table"``
        (HTML / markdown table).
    max_new_tokens : int
        Generation budget.
    backend : ``"openai"`` | ``"transformers"``
        Inference backend.
    base_url : str, optional
        API endpoint for the ``openai`` backend.
    api_key : str, optional
        API key / HF token for the ``openai`` backend.

    Returns
    -------
    str
        Extracted text in markdown (or LaTeX / HTML depending on task).
    """
    image = load_image(image)

    prompts = TASK_PROMPTS.get(model_id, TASK_PROMPTS["_default"])
    prompt_text = prompts.get(task, prompts["ocr"])

    return run_vlm(
        image, prompt_text, model_id,
        backend=backend, base_url=base_url, api_key=api_key,
        max_tokens=max_new_tokens,
    ).strip()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document OCR")
    parser.add_argument("image", help="Image path or URL")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", default="ocr", choices=["ocr", "formula", "table"])
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--backend", default="transformers",
                        choices=["openai", "transformers"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    result = document_ocr(
        args.image, model_id=args.model, task=args.task,
        max_new_tokens=args.max_tokens,
        backend=args.backend, base_url=args.base_url, api_key=args.api_key,
    )
    print(result)
