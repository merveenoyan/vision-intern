"""OCR adapter contracts without downloading model weights or requiring a GPU."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


@pytest.fixture
def ocr_module(monkeypatch):
    # Isolated module: leave the real torch/module imports untouched after each test.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        bfloat16="bf16", is_floating_point=lambda value: False,
    ))
    spec = importlib.util.spec_from_file_location(
        "tools._ocr_contract", Path(__file__).parents[2] / "tools" / "ocr.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("prompt", [None, "", "grounding"])
def test_lighton_prompt_and_generated_text(ocr_module, monkeypatch, prompt):
    class Inputs(dict):
        input_ids = [[10, 11]]

        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            content = messages[0]["content"]
            assert content[0]["type"] == "image"
            assert content[1:] == ([{"type": "text", "text": "grounding"}]
                                   if prompt == "grounding" else [])
            assert kwargs["enable_thinking"] is False
            return Inputs(input_ids=[[10, 11]])

        def batch_decode(self, tokens, **kwargs):
            assert tokens == [[42, 43]]  # prompt tokens must be removed
            return ["Recognized text"]

    class Model:
        device = "cpu"
        dtype = "bf16"

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 32
            return [[10, 11, 42, 43]]

    monkeypatch.setattr(ocr_module, "get_or_load", lambda key, loader: (Model(), Processor()))
    assert ocr_module.ocr(Image.new("RGB", (8, 8)), size="lighton", prompt=prompt,
                          max_new_tokens=32) == {
        "text": "Recognized text", "model": "lightonai/LightOnOCR-3-0.8B",
    }


def test_lighton_loader_uses_qwen35(ocr_module, monkeypatch):
    loaded = []
    model = SimpleNamespace(eval=lambda: None)

    class ModelLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            loaded.append(model_id)
            return model

    processor = object()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3_5ForConditionalGeneration=ModelLoader,
        AutoProcessor=SimpleNamespace(from_pretrained=lambda model_id: processor),
    ))
    assert ocr_module._load_lighton("lightonai/LightOnOCR-3-0.8B") == (model, processor)
    assert loaded == ["lightonai/LightOnOCR-3-0.8B"]
