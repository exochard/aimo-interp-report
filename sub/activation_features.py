"""Native decoder activations, pooled without labels or fitted projections."""
import math

FRACTIONS = (0.25, 0.5, 0.75)
TRACE_TOKENS = 128
MAX_PROMPT_TOKENS = 4096
POOL_NAMES = ("prompt_mean", "prompt_last", "trace_mean", "trace_last")


def layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h", "model.language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("no decoder layer list found")


def capture(model, tokenizer, problem, effort="default"):
    import torch
    # Use the same model-native chat convention as the existing probe.
    from trace_features import chat

    prompt = chat(tokenizer, problem, effort)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    n = enc["input_ids"].shape[1]
    if n > MAX_PROMPT_TOKENS:
        raise ValueError("prompt exceeds frozen context cap; do not truncate the problem")
    with torch.inference_mode():
        seq = model.generate(**enc, do_sample=False, max_new_tokens=TRACE_TOKENS,
                             pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                             else tokenizer.eos_token_id)
    if seq.shape[1] <= n:
        raise ValueError("empty reasoning prefix")
    decoder = layers(model)
    indices = [max(0, math.ceil(f * len(decoder)) - 1) for f in FRACTIONS]
    pooled = {}

    def hook(index):
        def collect(_module, _inputs, output):
            h = (output[0] if isinstance(output, tuple) else output)[0].float()
            pooled[index] = torch.stack((h[:n].mean(0), h[n - 1], h[n:].mean(0), h[-1])).cpu()
        return collect

    handles = [decoder[i].register_forward_hook(hook(i)) for i in indices]
    try:
        with torch.inference_mode():
            # The decoder avoids allocating a sequence-by-vocabulary logits tensor.
            core = getattr(model, "model", model)
            core(input_ids=seq, attention_mask=torch.ones_like(seq), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    values = torch.cat([pooled[i] for i in indices]).numpy()
    if not bool(torch.isfinite(torch.from_numpy(values)).all()):
        raise ValueError("nonfinite activation")
    return values, {"layer_indices": indices, "prompt_tokens": n,
                    "trace_tokens": int(seq.shape[1] - n), "pools": list(POOL_NAMES)}
