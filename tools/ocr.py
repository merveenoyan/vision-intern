"""Vision-language OCR with selectable model size.

Four models, keyed by *size*:
    "large"  → datalab-to/chandra-ocr-2     (~5.3B, AutoModelForMultimodalLM)
    "medium" → rednote-hilab/dots.mocr       (~3B,  AutoModelForCausalLM, custom code)
    "small"  → datalab-to/surya-ocr-2        (~686M, AutoModelForMultimodalLM)
    "mini"   → PaddlePaddle/PaddleOCR-VL-1.6 (~0.9B, PaddleOCRVLForConditionalGeneration)
"""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .utils import get_or_load, load_image

# size -> (model_id, loader_kind). The loader_kind selects the inference path:
#   "multimodal_lm" — chandra/surya: AutoModelForMultimodalLM + chat-template tokenize
#   "dots"          — dots.mocr: AutoModelForCausalLM + qwen_vl_utils vision processing
#   "paddle"        — PaddleOCR-VL: PaddleOCRVLForConditionalGeneration, same chat flow
MODELS: dict[str, dict[str, str]] = {
    "large": {"id": "datalab-to/chandra-ocr-2", "kind": "multimodal_lm"},
    "medium": {"id": "rednote-hilab/dots.mocr", "kind": "dots"},
    "small": {"id": "datalab-to/surya-ocr-2", "kind": "multimodal_lm"},
    "mini": {"id": "PaddlePaddle/PaddleOCR-VL-1.6", "kind": "paddle"},
}

# dots.mocr ships a layout-aware prompt; the other models take a bare "OCR:".
_DEFAULT_PROMPTS = {
    "multimodal_lm": "OCR:",
    "paddle": "OCR:",
    "dots": "Please output the layout information from the image, including bounding "
            "boxes and text content, as markdown.",
}


def _load_multimodal_lm(model_id: str) -> tuple[Any, Any]:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, processor


def _load_paddle(model_id: str) -> tuple[Any, Any]:
    from transformers import AutoProcessor, PaddleOCRVLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = PaddleOCRVLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, processor


def _load_dots(model_id: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, processor


def _generate_multimodal_lm(model, processor, image, prompt, max_new_tokens) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    # Weights are bf16; cast only float tensors so integer inputs keep dtype.
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(model.dtype)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def _generate_dots(model, processor, image, prompt, max_new_tokens) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def ocr(
    image: str | Image.Image,
    prompt: str | None = None,
    size: str = "medium",
    max_new_tokens: int = 4096,
) -> dict:
    """Extract text from an image using a vision-language OCR model.

    Args:
        image: input image (path, URL, or PIL Image)
        size: which model to use — "large" (chandra-ocr-2),
               "medium" (dots.mocr), "small" (surya-ocr-2),
               or "mini" (PaddleOCR-VL-1.6)
        prompt: instruction for the model; defaults to a size-appropriate prompt
        max_new_tokens: generation budget (raise for dense, multi-page documents)

    Returns a dict with:
        text (str): extracted text content
        model (str): the Hugging Face model id used
    """
    if size not in MODELS:
        raise ValueError(f"size must be one of {sorted(MODELS)}, got {size!r}")

    spec = MODELS[size]
    model_id, kind = spec["id"], spec["kind"]
    if prompt is None:
        prompt = _DEFAULT_PROMPTS[kind]

    image = load_image(image)

    if kind == "dots":
        model, processor = get_or_load(f"ocr_{size}", lambda: _load_dots(model_id))
        text = _generate_dots(model, processor, image, prompt, max_new_tokens)
    else:
        # multimodal_lm (chandra/surya) and paddle share the chat-template flow;
        # only the model class differs at load time.
        loader = _load_paddle if kind == "paddle" else _load_multimodal_lm
        model, processor = get_or_load(f"ocr_{size}", lambda: loader(model_id))
        text = _generate_multimodal_lm(model, processor, image, prompt, max_new_tokens)

    return {"text": text, "model": model_id}
