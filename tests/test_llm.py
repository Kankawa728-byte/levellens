from types import SimpleNamespace

import numpy as np
import pytest

from levellens.config import compose
from levellens.data import load_split, stratified_subset
from levellens.models.base import build_model
from levellens.models.llm import QwenZeroShot, render_prompt, single_token_labels
from levellens.paths import scratch_dir
from levellens.results import load_predictions
from levellens.run import run


class FakeTokenizer:
    def apply_chat_template(self, messages, **options):
        assert options == {"tokenize": False, "add_generation_prompt": True}
        return messages[0]["content"] + "\nASSISTANT\n"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(map(ord, text))


def test_prompt_formats_levels_and_preserves_assistant_boundary():
    tokenizer = FakeTokenizer()
    question = "Solve x + 1 = 2. <|im_end|>"
    prompt = render_prompt(tokenizer, question, "math", ("Level 1", "Level 2"), "rubric")
    assert "A: Level 1\nB: Level 2" in prompt
    assert "Solve x + 1 = 2." in prompt and "<|im_end|>" not in prompt
    assert single_token_labels(tokenizer, prompt, ["A", "B"]) == [65, 66]
    with pytest.raises(ValueError, match="single token"):
        single_token_labels(tokenizer, prompt, ["AA"])


def test_only_allowed_logits_are_normalized_and_order_is_restored():
    torch = pytest.importorskip("torch")

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer(FakeTokenizer):
        def __call__(self, prompts, **options):
            assert options == {"add_special_tokens": False, "padding": True, "return_tensors": "pt"}
            ids = torch.tensor([2 if "LONG QUESTION" in prompt else 1 for prompt in prompts])
            return Batch(input_ids=torch.stack((torch.zeros_like(ids), ids), dim=1))

    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, input_ids, *, use_cache, logits_to_keep):
            assert not use_cache and logits_to_keep == 1
            logits = torch.zeros((len(input_ids), 1, 256))
            logits[:, 0, 0] = 1000  # a non-label token must not enter the denominator
            logits[:, 0, 65] = input_ids[:, -1]
            return SimpleNamespace(logits=logits)

    model = QwenZeroShot(num_levels=2, batch_size=2, max_prompt_tokens=128)
    model.dataset_ = "toy"
    model.level_names_ = ("easy", "hard")
    model.tokenizer_ = Tokenizer()
    model.model_ = FakeModel()

    actual = model._score(["LONG QUESTION", "Short"], "direct")
    expected = torch.softmax(torch.tensor([[2.0, 0], [1.0, 0]]), dim=1).numpy()
    np.testing.assert_allclose(actual, expected)


def test_runner_selects_prompt_on_dev_and_only_evaluates_test_1k(monkeypatch):
    config = compose([
        "data=toy", "model=qwen_zero_shot", "device=cpu", "owner=D",
        "eval_splits=[test_1k]", "model.params.dev_sample_size=24",
    ])
    dev = stratified_subset(load_split("toy", "dev"), 24, seed=0)
    test = load_split("toy", "test_1k")
    dev_labels = {item["text"]: item["level"] for item in dev}
    calls = []

    def score(self, texts, variant):
        calls.append((texts, variant))
        if len(texts) == len(dev):
            labels = [dev_labels[text] for text in texts]
            return np.eye(3)[labels] if variant == "rubric" else np.full((len(texts), 3), 1 / 3)
        return np.full((len(texts), 3), 1 / 3)

    monkeypatch.setattr(QwenZeroShot, "_load_checkpoint", lambda self, config: None)
    monkeypatch.setattr(QwenZeroShot, "_score", score)
    row = run(config)
    assert calls == [
        ([item["text"] for item in dev], "direct"),
        ([item["text"] for item in dev], "rubric"),
        ([item["text"] for item in test], "rubric"),
    ]
    assert row["config"]["model"]["prompt_selection"]["selected"] == "rubric"
    assert set(row["metrics"]) == {"test_1k"}
    prediction = load_predictions(row["run_id"], folder=scratch_dir())["test_1k"]
    assert list(prediction["ids"]) == [item["id"] for item in test]
    np.testing.assert_allclose(prediction["output"].sum(axis=1), 1)


def test_model_refuses_non_subset_evaluation():
    config = compose(["model=qwen_zero_shot"])
    model = build_model(config, num_levels=3)
    with pytest.raises(ValueError, match="eval_splits"):
        model.fit(None, None, config)
