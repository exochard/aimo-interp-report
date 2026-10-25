import numpy as np

from fit_activation_probe import native_kernel, nested, predict


def test_native_spaces_do_not_mix_model_coordinates():
    values = [np.array([[1., 0.], [0., 2.]])] * 3
    k = native_kernel(values, ["a", "a", "b"])
    assert k[0, 1] == 1.25
    assert k[0, 2] == 0
    assert np.linalg.eigvalsh(k).min() >= -1e-12


def test_problem_holdout_includes_all_model_rows_and_inner_tuning_is_independent():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 5))
    k = x @ x.T
    groups = np.repeat(["a", "b", "c", "d", "e", "f"], 2)
    y = np.array([0, 1] * 6, bool)
    p, folds = nested(k, y, groups)
    for fold in folds:
        tr, te = fold["fit_rows"], fold["held_out_rows"]
        assert not set(groups[tr]) & set(groups[te])
        altered = y.copy()
        altered[te] = ~altered[te]
        changed, changed_folds = nested(k, altered, groups)
        other = next(f for f in changed_folds if f["problem_sha256"] == fold["problem_sha256"])
        assert other["inner_accuracy"] == fold["inner_accuracy"]
        assert other["penalty"] == fold["penalty"]
        np.testing.assert_array_equal(changed[te], p[te])


def test_unused_heldout_kernel_rows_cannot_change_prediction():
    k = np.eye(4)
    y = np.array([0, 1, 1, 0], bool)
    expected = predict(k, y, np.array([0, 1]), np.array([2]), 0.1)
    k[3] = 1000
    k[:, 3] = 1000
    np.testing.assert_array_equal(expected, predict(k, y, np.array([0, 1]), np.array([2]), 0.1))


def test_real_decoder_hooks_pool_prompt_and_trace_and_are_removed():
    import torch
    from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM
    from activation_features import capture

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 63

        def apply_chat_template(self, *args, **kwargs):
            return "problem"

        def __call__(self, *args, **kwargs):
            return BatchEncoding(dict(input_ids=torch.tensor([[1, 2, 3]]),
                                      attention_mask=torch.ones((1, 3), dtype=torch.long)))

    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        eos_token_id=63, pad_token_id=0)).eval()
    values, info = capture(model, Tokenizer(), "problem")
    assert values.shape == (12, 16)
    assert info['layer_indices'] == [0, 1, 2]
    assert info['prompt_tokens'] == 3
    assert np.isfinite(values).all()
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_failed_attempt_cannot_publish_a_stale_quality_result(tmp_path):
    import json
    from record_activation_result import record
    (tmp_path/'probe.json').write_text(json.dumps(dict(passed=True)))
    (tmp_path/'terminal.json').write_text(json.dumps(dict(extraction_exit=1, fit_exit=75,
        probe_sha256=None, reason='current extraction failed', finished='2026-10-14T14:00+02:00')))
    report = tmp_path/'report.md'
    report.write_text('<!-- AIMO_RESULT_START -->\npending\n<!-- AIMO_RESULT_END -->')
    record(tmp_path, report)
    assert 'INCOMPLETE: current extraction failed' in report.read_text()
    assert 'PASS:' not in report.read_text()


def test_terminal_must_match_probe_hash(tmp_path):
    import json
    from record_activation_result import record
    (tmp_path/'probe.json').write_text(json.dumps(dict(passed=True)))
    (tmp_path/'terminal.json').write_text(json.dumps(dict(extraction_exit=0, fit_exit=0,
        probe_sha256='old-run', reason='complete', finished='2026-10-14T14:00+02:00')))
    report = tmp_path/'report.md'
    report.write_text('<!-- AIMO_RESULT_START -->\npending\n<!-- AIMO_RESULT_END -->')
    record(tmp_path, report)
    assert 'INCOMPLETE: terminal/probe hash mismatch' in report.read_text()


def test_graphics_guard_stops_only_its_own_child(monkeypatch):
    import subprocess
    import sys
    import pytest
    import gpu_available
    children = []
    original = subprocess.Popen

    def launch(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    probes = [[], [{'pid': '99999999', 'name': 'game', 'type': 'G', 'mib': 2048}]]
    monkeypatch.setattr(gpu_available.subprocess, 'Popen', launch)
    monkeypatch.setattr(gpu_available, 'blockers', lambda allowed=(): probes.pop(0))
    with pytest.raises(RuntimeError, match='foreground GPU'):
        gpu_available.run([sys.executable, '-c', 'import time; time.sleep(60)'])
    assert len(children) == 1
    assert children[0].returncode is not None
