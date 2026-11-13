"""Share E: output heads, the BiLSTM, the transformer wrapper, the Predictor export and the grid file.

Everything here runs on the CPU without a download. The transformer tests build a tiny random
BERT in a temporary folder and use it as the backbone.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from levellens.config import compose  # noqa: E402
from levellens.data import load_split  # noqa: E402
from levellens.features.glove import tokenize  # noqa: E402
from levellens.heads import CornHead, CrossEntropyHead, build_head, class_weights  # noqa: E402
from levellens.models.neural import pad_batch  # noqa: E402
from levellens.paths import cache_dir, config_dir, scratch_dir  # noqa: E402
from levellens.predictor import Predictor  # noqa: E402
from levellens.results import load_predictions  # noqa: E402
from levellens.run import run  # noqa: E402

BILSTM = ["data=toy", "model=bilstm", "device=cpu", "model.params.embeddings=random", "model.params.hidden=64", "model.params.lr=3e-3", "model.params.patience=0",
          "model.params.epochs=12"]


# ---- heads ----------------------------------------------------------------------------------

@pytest.mark.parametrize("num_levels", [2, 3, 5, 7, 12])
def test_corn_probabilities_are_between_0_and_1_and_sum_to_1(num_levels):
    torch.manual_seed(0)
    head = CornHead(in_features=16, num_levels=num_levels)
    logits = head(torch.randn(200, 16) * 20)  # large inputs: saturated sigmoids must not break the sum
    proba = head.proba(logits)
    assert logits.shape == (200, num_levels - 1) and proba.shape == (200, num_levels)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert torch.allclose(proba.sum(dim=1), torch.ones(200), atol=1e-6)


def test_corn_known_values():
    head = CornHead(in_features=4, num_levels=3)
    # P(y>0) = 0.5 and P(y>1 | y>0) = 0.5  ->  P(0) = 0.5, P(1) = 0.25, P(2) = 0.25
    assert torch.allclose(head.proba(torch.zeros(1, 2)), torch.tensor([[0.5, 0.25, 0.25]]))
    # level 0 only trains task 0 (target "no"); level 2 trains both tasks (target "yes"): the mean of 3 x log 2
    loss = head.loss(torch.zeros(2, 2), torch.tensor([0, 2]))
    assert loss.item() == pytest.approx(np.log(2))
    # the second task of a level-0 question is outside its conditional subset: its logit gets no gradient
    logits = torch.zeros(1, 2, requires_grad=True)
    head.loss(logits, torch.tensor([0])).backward()
    assert logits.grad[0, 0] > 0 and logits.grad[0, 1] == 0


def test_corn_loss_falls_and_learns_the_order():
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 300).unsqueeze(1)
    y = torch.bucketize(x.squeeze(1), torch.tensor([-0.5, 0.0, 0.5]))  # four ordered levels along one axis
    head = CornHead(in_features=1, num_levels=4)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.1)
    first = head.loss(head(x), y).item()
    for _ in range(300):
        optimizer.zero_grad()
        head.loss(head(x), y).backward()
        optimizer.step()
    assert head.loss(head(x), y).item() < 0.5 * first
    assert (head.proba(head(x)).argmax(dim=1) == y).float().mean() > 0.9


def test_cross_entropy_head_and_class_weights():
    weights = class_weights([0, 0, 0, 1], num_levels=3)
    assert weights.tolist() == pytest.approx([4 / 9, 4 / 3, 0.0])  # n / (K * count); an empty level gets 0
    head = CrossEntropyHead(in_features=8, num_levels=3, class_weights=weights)
    proba = head.proba(head(torch.randn(5, 8)))
    assert proba.shape == (5, 3) and torch.allclose(proba.sum(dim=1), torch.ones(5))
    # with the weights, three level-0 questions together count as much as one level-1 question
    logits, y = torch.zeros(4, 3), torch.tensor([0, 0, 0, 1])
    assert head.loss(logits, y).item() == pytest.approx(np.log(3))
    for cls in (CrossEntropyHead, CornHead):
        weighted = cls(in_features=8, num_levels=3, class_weights=weights)
        assert torch.isfinite(weighted.loss(weighted(torch.randn(4, 8)), y))


def test_build_head_reads_the_config_and_rejects_head_none():
    assert isinstance(build_head(compose(["head=corn"]).head, 8, 5), CornHead)
    assert isinstance(build_head(compose(["head=ce"]).head, 8, 5), CrossEntropyHead)
    with pytest.raises(ValueError, match="head=ce or head=corn"):
        build_head(compose(["head=none"]).head, 8, 5)


# ---- pieces of the loop -----------------------------------------------------------------------

def test_tokenize_and_dynamic_padding():
    assert tokenize("What's $x^2$? A. 12 B. x_1") == ["what's", "$", "x", "^", "2", "$", "?", "a", ".", "12", "b", ".", "x", "_", "1"]
    input_ids, mask = pad_batch([[5, 6, 7], [8]], pad_id=0)
    assert input_ids.tolist() == [[5, 6, 7], [8, 0, 0]]  # padded to the longest row of this batch only
    assert mask.tolist() == [[True, True, True], [True, False, False]]


# ---- BiLSTM -----------------------------------------------------------------------------------

@pytest.mark.parametrize("head", ["ce", "corn"])
def test_bilstm_learns_the_toy_dataset_and_reloads(head):
    row = run(compose([*BILSTM, f"head={head}", "save_model=true"]))
    assert row["run_id"] == f"toy-bilstm-{head}-s0" and row["n_params"] > 0
    assert row["metrics"]["test"]["accuracy"] > 0.6  # the majority floor is 1/3

    stored = load_predictions(row["run_id"], scratch_dir())["test"]
    assert stored["output"].shape == (90, 3) and np.allclose(stored["output"].sum(axis=1), 1.0, atol=1e-5)
    # E8: the Predictor gives the predictions of the training run
    result = Predictor.load(row["run_id"]).predict(load_split("toy", "test"))
    assert np.allclose(result["proba"], stored["output"], atol=1e-5)
    assert result["level_names"] == ["easy", "medium", "hard"]


def test_bilstm_attention_ignores_padding():
    from levellens.models.recurrent import BiLSTMEncoder

    encoder = BiLSTMEncoder(vocab_size=10, hidden=8, layers=2, dropout=0.0, embedding_dropout=0.0).eval()
    short, _ = pad_batch([[3, 4]], pad_id=0)
    padded, mask = pad_batch([[3, 4], [5, 6, 7, 8, 9]], pad_id=0)
    with torch.no_grad():
        alone = encoder(short, torch.ones_like(short, dtype=torch.bool))
        together = encoder(padded, mask)
        weights = encoder.attention_weights(torch.randn(2, 5, 16), mask)
    assert torch.allclose(alone[0], together[0], atol=1e-6)  # padding does not change the question vector
    assert weights[0, 2:].sum() == 0 and torch.allclose(weights.sum(dim=1), torch.ones(2))


def test_unknown_setting_is_reported():
    with pytest.raises(ValueError, match="hiden"):
        run(compose([*BILSTM, "head=ce", "model.params.hiden=8"]))


# ---- transformer wrapper ----------------------------------------------------------------------

@pytest.fixture
def tiny_backbone(tmp_path):
    """A random 2-layer BERT with a word-level vocabulary, saved like a hub model."""
    transformers = pytest.importorskip("transformers")
    from levellens.data.toy import WORDS

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "which", "answer", "fits", ":", "?", ".", "a", "b", "c", "d",
             *(word for level in WORDS for word in level)]
    folder = tmp_path / "tiny-bert"
    folder.mkdir()
    (folder / "vocab.txt").write_text("\n".join(words), encoding="utf-8")
    transformers.BertTokenizerFast(str(folder / "vocab.txt")).save_pretrained(folder)
    config = transformers.BertConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                     intermediate_size=64, max_position_embeddings=64)
    torch.manual_seed(0)
    transformers.BertModel(config).save_pretrained(folder)
    return folder


def transformer_overrides(backbone, *more):
    return ["data=toy", "model=distilbert", "device=cpu", f"model.params.backbone={backbone}", "model.params.lr=1e-3",
            "model.params.max_length=32", "model.params.epochs=8", "model.params.patience=0", *more]


@pytest.mark.parametrize("head", ["ce", "corn"])
def test_transformer_wrapper_trains_any_backbone_with_any_head(tiny_backbone, head):
    row = run(compose(transformer_overrides(tiny_backbone, f"head={head}", "save_model=true")))
    assert row["metrics"]["test"]["accuracy"] > 0.5 and row["n_params"] > 0
    assert list((cache_dir() / "tokens").glob("*.pt"))  # the token ids of train, dev and test were cached

    stored = load_predictions(row["run_id"], scratch_dir())["test"]
    predictor = Predictor.load(row["run_id"])  # E8: loads from the saved folder alone
    result = predictor.predict(load_split("toy", "test"))
    assert np.allclose(result["proba"], stored["output"], atol=1e-5)
    single = predictor.predict(["Which answer fits: cat dog sun red? A. cup B. hat C. day D. run"])
    assert single["proba"].shape == (1, 3) and np.isclose(single["proba"].sum(), 1.0)


def test_early_stopping_keeps_the_best_weights_and_reports_to_the_callback(tiny_backbone):
    from levellens.models.base import build_model

    config = compose(transformer_overrides(tiny_backbone, "head=ce", "model.params.patience=1", "model.params.epochs=30",
                                           "model.params.evals_per_epoch=2"))
    model = build_model(config, num_levels=3)
    seen = []
    model.on_eval = lambda step, score: seen.append((step, score))
    model.fit(load_split("toy", "train"), load_split("toy", "dev"), config)
    scores = [entry["dev_qwk"] for entry in model.history]
    assert len(scores) < 60 and scores[-1] <= max(scores)  # stopped early, after an evaluation that was not better
    assert [step for step, _ in seen] == list(range(1, len(scores) + 1))
    assert model.best_dev_qwk == pytest.approx(max(scores), abs=1e-4)

    from levellens.evaluate import evaluate
    dev = load_split("toy", "dev")
    restored = evaluate(np.array([item["level"] for item in dev]), model.predict_proba(dev), 3)["qwk"]
    assert restored == pytest.approx(model.best_dev_qwk, abs=1e-4)  # the best weights are back, not the last ones


def test_fp16_is_refused(tiny_backbone):
    with pytest.raises(ValueError, match="fp16"):
        run(compose(transformer_overrides(tiny_backbone, "head=ce", "model.params.precision=fp16")))


# ---- grid for share C ---------------------------------------------------------------------------

def test_tier1_grid_has_20_transformer_runs_that_compose():
    from levellens.grid import read_grid

    lines = read_grid(config_dir() / "grid" / "tier1_transformers.txt")
    assert len(lines) == 20
    seen = set()
    for overrides in lines:
        config = compose(overrides)
        assert config.owner == "E" and config.model.target.endswith("Transformer") and "target" in config.head
        seen.add((config.data.name, config.model.name, config.head.name, config.seed))
    assert len(seen) == 20
    assert {key for key in seen if key[3] > 0} == {("arc", m, h, s) for m in ("deberta-v3", "distilbert")
                                                   for h in ("ce", "corn") for s in (1, 2)}
    assert len(read_grid(config_dir() / "grid" / "tier1_bilstm.txt")) == 6


def test_length_buckets_cover_every_row_once_with_similar_lengths():
    from levellens.models.neural import length_buckets

    rows = [[0] * (1 + index % 37) for index in range(500)]
    generator = torch.Generator().manual_seed(0)
    batches = length_buckets(rows, batch_size=16, generator=generator)
    assert sorted(row for batch in batches for row in batch) == list(range(500))
    spread = np.mean([max(len(rows[r]) for r in batch) - min(len(rows[r]) for r in batch) for batch in batches])
    assert spread < 5  # a random batch of 16 would span about 32 lengths
    assert batches != length_buckets(rows, batch_size=16, generator=generator)  # a new order every epoch


# ---- Optuna ---------------------------------------------------------------------------------------

def test_tune_searches_on_dev_and_retrains_the_best_setting(tiny_backbone):
    pytest.importorskip("optuna")
    import json

    from levellens.paths import results_dir
    from levellens.tune import split_arguments, tune

    overrides, n_trials, study = split_arguments([*transformer_overrides(tiny_backbone), "n_trials=3", "study=toy-tiny",
                                                  "model.params.epochs=3"])
    assert n_trials == 3 and study == "toy-tiny" and "n_trials=3" not in overrides
    row = tune(overrides, n_trials, study)
    assert row["run_id"] == f"toy-distilbert-{row['head']}-optuna-s0" and row["variant"] == "optuna"

    summary = json.loads((results_dir() / "optuna" / "toy-tiny.json").read_text(encoding="utf-8"))
    assert summary["trials"] == 3 and set(summary["best_params"]) == {"head", "lr", "batch_size", "weight_decay",
                                                                      "warmup_ratio", "dropout"}
    assert row["head"] == summary["best_params"]["head"]
    assert row["config"]["model"]["params"]["lr"] == pytest.approx(summary["best_params"]["lr"])
    assert (results_dir() / "optuna" / "toy-tiny.csv").exists()
    assert Predictor.load(row["run_id"]).predict(["Which answer fits: cat dog?"])["proba"].shape == (1, 3)

    tune(overrides, n_trials, study)  # a second call continues the stored study: no new trials
    assert json.loads((results_dir() / "optuna" / "toy-tiny.json").read_text(encoding="utf-8"))["trials"] == 3


def test_grid_command_lists_the_chosen_lines_with_extra_overrides(capsys):
    from levellens.grid import main

    main([str(config_dir() / "grid" / "tier1_bilstm.txt"), "--list", "--only", "1,3", "device=cpu"])
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 2 and printed[0].startswith(" 1  python -m levellens.run data=math model=bilstm head=ce")
    assert printed[1].endswith("device=cpu") and "data=arc" in printed[1]
