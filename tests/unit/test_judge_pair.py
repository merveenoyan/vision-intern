import pytest

from tools.model_options import DEFAULT_JUDGE_PAIR, JUDGE_MODEL_OPTIONS
from workflows.vlm_judge import _normalize_judges, ensemble_row


def test_d1_replaces_second_judge():
    pair = [DEFAULT_JUDGE_PAIR[0], JUDGE_MODEL_OPTIONS[-1]]
    specs = _normalize_judges(pair, '', 'transformers', None, None)
    assert [s['model_id'] for s in specs] == pair
    verdicts = {model: [{'detection_idx': 0, 'verdict': 'correct',
                        'score': .9, 'reason': ''}] for model in pair}
    assert ensemble_row(verdicts, 1, 2)[0]['ensemble_keep']


def test_third_judge_rejected_before_inference():
    with pytest.raises(ValueError, match='third judge'):
        _normalize_judges(list(JUDGE_MODEL_OPTIONS), '', 'transformers', None, None)
    with pytest.raises(ValueError, match='third judge'):
        ensemble_row({model: [] for model in JUDGE_MODEL_OPTIONS}, 0, 2)


def test_merge_rejects_third_file_before_loading(monkeypatch):
    from pathlib import Path
    monkeypatch.setenv('REPO_DIR', str(Path(__file__).resolve().parents[2]))
    from jobs import merge_judges
    monkeypatch.setattr('sys.argv', ['merge', '--verdicts', 'a::a.parquet',
                                   '--verdicts', 'b::b.parquet',
                                   '--verdicts', 'd1::d1.parquet'])
    with pytest.raises(SystemExit) as exc:
        merge_judges.main()
    assert exc.value.code == 2
