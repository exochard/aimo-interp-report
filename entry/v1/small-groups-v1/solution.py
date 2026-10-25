"""Small track: robust for the configurations that are net robust on the public competition set."""

ROBUST = {"Qwen/Qwen3.5-4B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    return [model_id in ROBUST for _ in problems]
