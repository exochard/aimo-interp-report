"""Group rule plus the kit's uncertainty profile for DeepSeek, the one model with a trained artifact.

Configurations in ROBUST are net robust on the public competition set; DeepSeek is judged per
problem from its own greedy trace (kit baseline, bundled artifact); everything else is non-robust.
"""

from uncertainty_profile.inference import predict_robustness

ROBUST = {("Qwen/Qwen3.5-4B", "*")}
PROFILED = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    if model_id == PROFILED:
        return predict_robustness(model_id=model_id, problem_texts=problems)
    return [(model_id, reasoning_effort) in ROBUST or (model_id, "*") in ROBUST for _ in problems]
