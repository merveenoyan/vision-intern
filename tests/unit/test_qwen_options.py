from types import SimpleNamespace

import pytest
from PIL import Image


@pytest.mark.parametrize('model_id', ['Qwen/Qwen3.6-35B-A3B', 'Qwen/Qwen3.8-27B'])
def test_new_qwen_requests_disable_thinking(monkeypatch, model_id):
    from tools import vlm_client
    seen = {}
    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='[]'))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(vlm_client, '_get_client', lambda *args: client)
    assert vlm_client._run_openai(Image.new('RGB',(8,8)), 'detect', model_id,
                                 'http://localhost:8000/v1', 'unused', 1024) == '[]'
    assert seen['extra_body']['chat_template_kwargs'] == {'enable_thinking': False}


def test_detection_parser_ignores_reasoning_boxes():
    from tools.vlm_detect import parse_bboxes_from_response
    reasoning = '<think>{"label":"wrong","bbox_2d":[0,0,100,100]}</think>'
    answer = '[{"label":"stop","bbox_2d":[10,20,30,40]}]'
    assert [b['label'] for b in parse_bboxes_from_response(reasoning+answer)] == ['stop']
    assert parse_bboxes_from_response('<think>'+answer) == []


def test_qwen35_remains_an_option_with_unchanged_request_policy():
    from tools.model_options import QWEN_LABEL_MODELS, structured_output_template_kwargs
    assert QWEN_LABEL_MODELS == (
        "Qwen/Qwen3.5-9B", "Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.8-27B",
    )
    assert structured_output_template_kwargs("Qwen/Qwen3.5-9B") == {}
    assert structured_output_template_kwargs("Qwen/Qwen3.5-9B:provider") == {}
