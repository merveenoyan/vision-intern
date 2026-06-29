"""Shared VLM inference layer with two backends.

Backend ``"openai"``
    Uses the OpenAI Python client.  A single code-path that works with:

    - **HF Inference Providers** — ``base_url="https://router.huggingface.co/v1"``
      (default when no *base_url* is given), ``api_key`` = your HF token.
    - **vLLM server** — ``base_url="http://localhost:8000/v1"``
    - **llama-server** — ``base_url="http://localhost:8080/v1"``
    - Any other OpenAI-compatible endpoint.

Backend ``"transformers"``
    Loads the model locally with ``transformers`` + ``torch``.
    Requires a CUDA GPU.  Models are cached after the first call.
"""

from __future__ import annotations

import base64
import io
import os
import threading
from typing import Any

from PIL import Image

# Cached OpenAI clients keyed by (base_url, api_key).
_openai_clients: dict[tuple[str, str], Any] = {}
_client_lock = threading.Lock()

HF_INFERENCE_URL = "https://router.huggingface.co/v1"


# ------------------------------------------------------------------
# Image encoding
# ------------------------------------------------------------------

_MAX_DIMENSION = 1280


def _pil_to_data_url(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 data URL, resizing if needed.

    Large images are downscaled to fit within _MAX_DIMENSION and saved as
    JPEG to avoid hitting API request-size limits.
    """
    w, h = image.size
    if max(w, h) > _MAX_DIMENSION:
        scale = _MAX_DIMENSION / max(w, h)
        image = image.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS,
        )

    buf = io.BytesIO()
    if image.mode == "RGBA":
        image.save(buf, format="PNG")
        mime = "image/png"
    else:
        image.save(buf, format="JPEG", quality=85)
        mime = "image/jpeg"

    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


# ------------------------------------------------------------------
# OpenAI-compatible backend
# ------------------------------------------------------------------

def _get_client(base_url: str, api_key: str) -> Any:
    key = (base_url, api_key)
    if key not in _openai_clients:
        with _client_lock:
            if key not in _openai_clients:
                from openai import OpenAI
                _openai_clients[key] = OpenAI(
                    base_url=base_url, api_key=api_key,
                )
    return _openai_clients[key]


def _resolve_api_key(api_key: str | None) -> str:
    return (
        api_key
        or os.environ.get("HF_TOKEN")
        or os.environ.get("OPENAI_API_KEY")
        or "no-key"
    )


def _run_openai(
    image: Image.Image,
    prompt: str,
    model_id: str,
    base_url: str | None,
    api_key: str | None,
    max_tokens: int,
) -> str:
    import time

    base_url = base_url or HF_INFERENCE_URL
    api_key = _resolve_api_key(api_key)
    client = _get_client(base_url, api_key)

    data_url = _pil_to_data_url(image)

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise last_err


# ------------------------------------------------------------------
# Transformers backend
# ------------------------------------------------------------------

def _load_vlm(model_id: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def _run_transformers_batch(
    images: list[Image.Image],
    prompts: list[str],
    model_id: str,
    max_tokens: int,
) -> list[str]:
    """Generate for a *batch* of (image, prompt) pairs in one forward pass.

    This is the throughput path for the GPU judge jobs: the small judge models
    (e.g. ``LiquidAI/LFM2.5-VL-1.6B``, ``google/gemma-4-E4B-it``) are not on HF
    Inference Providers, so they run locally with ``transformers`` — feeding one
    image at a time wastes the GPU.

    Uses the standard multimodal batching path: a list of conversations (image
    embedded per turn) through ``apply_chat_template`` with ``padding=True`` and
    ``padding_side="left"`` (left padding is what decoder-only ``generate`` wants,
    so every row's new tokens start at the same offset and one slice serves the
    whole batch).
    """
    import torch

    from .utils import get_or_load

    model, processor = get_or_load(f"vlm:{model_id}", lambda: _load_vlm(model_id))

    conversations = [
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        for img, prompt in zip(images, prompts)
    ]
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
        )

    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)


def _run_transformers(
    image: Image.Image,
    prompt: str,
    model_id: str,
    max_tokens: int,
) -> str:
    return _run_transformers_batch([image], [prompt], model_id, max_tokens)[0]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def run_vlm(
    image: Image.Image,
    prompt: str,
    model_id: str,
    backend: str = "transformers",
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 4096,
) -> str:
    """Send *image* + *prompt* to a VLM and return the response text.

    Parameters
    ----------
    image : PIL.Image
        RGB image (already loaded).
    prompt : str
        User message text.
    model_id : str
        Model identifier — a HF repo id for both backends, or whatever
        your OpenAI-compatible server expects.
    backend : ``"openai"`` | ``"transformers"``
        ``"openai"`` routes through the OpenAI Python client and works
        with HF Inference Providers, vLLM, llama-server, or any
        compatible endpoint.  ``"transformers"`` loads the model locally.
    base_url : str, optional
        API base URL for the ``openai`` backend.  Defaults to
        ``https://router.huggingface.co/v1`` (HF Inference Providers).
        Set to ``http://localhost:8000/v1`` for a local vLLM server,
        ``http://localhost:8080/v1`` for llama-server, etc.
    api_key : str, optional
        API key / token.  Falls back to ``HF_TOKEN`` then
        ``OPENAI_API_KEY`` environment variables.
    max_tokens : int
        Maximum tokens to generate.

    Returns
    -------
    str
        The model's response text.
    """
    if backend == "openai":
        return _run_openai(image, prompt, model_id, base_url, api_key, max_tokens)
    if backend == "transformers":
        return _run_transformers(image, prompt, model_id, max_tokens)
    raise ValueError(
        f"Unknown backend {backend!r}. Use 'openai' or 'transformers'."
    )


def run_vlm_batch(
    images: list[Image.Image],
    prompts: list[str],
    model_id: str,
    backend: str = "transformers",
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 4096,
) -> list[str]:
    """Run a VLM over a *batch* of (image, prompt) pairs; one response each.

    The ``transformers`` backend runs the whole batch through a single
    ``generate`` call (the throughput path for the GPU judge jobs). The
    ``openai`` backend has no batched chat endpoint, so it falls back to
    sequential per-item calls — same results, just not faster.

    *images* and *prompts* must be the same length; the returned list is aligned
    with them.
    """
    if len(images) != len(prompts):
        raise ValueError(
            f"images ({len(images)}) and prompts ({len(prompts)}) must match",
        )
    if not images:
        return []
    if backend == "transformers":
        return _run_transformers_batch(images, prompts, model_id, max_tokens)
    if backend == "openai":
        return [
            _run_openai(img, prompt, model_id, base_url, api_key, max_tokens)
            for img, prompt in zip(images, prompts)
        ]
    raise ValueError(
        f"Unknown backend {backend!r}. Use 'openai' or 'transformers'."
    )
