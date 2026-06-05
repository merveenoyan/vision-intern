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

def _run_transformers(
    image: Image.Image,
    prompt: str,
    model_id: str,
    max_tokens: int,
) -> str:
    import torch

    from .utils import get_or_load

    def _load() -> tuple[Any, Any]:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        return model, processor

    model, processor = get_or_load(f"vlm:{model_id}", _load)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
        )

    return processor.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )


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
