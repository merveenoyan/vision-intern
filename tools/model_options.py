"""Explicit worker options; arbitrary model IDs remain supported."""

QWEN_LABEL_MODELS = (
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.8-27B",
)


def structured_output_template_kwargs(model_id: str) -> dict:
    """Request direct structured answers from the newer thinking-default Qwens.

    Provider suffixes such as ':provider' do not change the model policy.
    OpenAI-compatible servers must support chat_template_kwargs.
    """
    if model_id.split(":", 1)[0] in {"Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.8-27B"}:
        return {"enable_thinking": False}
    return {}


# Two slots in the pipeline; d1 replaces a judge rather than adding a third.
DEFAULT_JUDGE_PAIR = ("google/gemma-4-E4B-it", "LiquidAI/LFM3-VL-2B")
JUDGE_MODEL_OPTIONS = (*DEFAULT_JUDGE_PAIR, "LiquidAI/d1-Omni-2B")


def validate_judge_count(count: int) -> None:
    """Allow standalone scoring or a two-judge ensemble."""
    if not 1 <= count <= 2:
        raise ValueError("Use one standalone judge or two ensemble judges; "
                         "d1 is a replacement option, not a third judge.")
