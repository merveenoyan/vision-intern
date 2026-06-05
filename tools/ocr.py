"""PaddleOCR-VL — vision-language OCR."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .utils import get_or_load, load_image

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"


def _load() -> tuple[Any, Any]:
    from transformers import AutoProcessor, PaddleOCRVLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = PaddleOCRVLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, processor


def ocr(
    image: str | Image.Image,
    prompt: str = "OCR:",
) -> dict:
    """Extract text from an image using PaddleOCR-VL.

    Args:
        image: input image (path, URL, or PIL Image)
        prompt: instruction for the model (default "OCR:")

    Returns a dict with:
        text (str): extracted text content
    """
    model, processor = get_or_load("paddleocr_vl", _load)
    image = load_image(image)

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
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024)

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return {"text": text}
