"""Training accounting, seeded subsets, checkpoint selection and a tiny run."""

import json
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import src.train as training
from src.dataset import main as prepare_data
from src.models.transformer import TransformerRecommender
from src.scoring import make_score_batch


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        (1, 10, 5, 10), (1, 20, 5, 20), (1, 30, 5, 30),
        (1, 60, 1, 35), (1, 40, 5, 40), (1, 50, 5, 50),
        (2, 40, 5, 10), (2, 50, 5, 20), (2, 60, 5, 30),
        (2, 10, 5, 40), (2, 20, 5, 50),
    ]
    (raw / "ratings.dat").write_text(
        "\n".join("::".join(map(str, row)) for row in rows), encoding="latin-1"
    )
    (raw / "movies.dat").write_text(
        "\n".join(f"{movie}::Movie {movie}::Drama" for movie in range(10, 61, 10)),
        encoding="latin-1",
    )
    processed = tmp_path / "processed"
    with monkeypatch.context() as context:
        context.setattr(sys, "argv", [
            "src.dataset", "--data-dir", str(raw), "--output-dir", str(processed),
            "--max-len", "2",
        ])
        prepare_data()
    return raw, processed / "preparation.json"


def tiny_config(**overrides):
    return training.TrainConfig(**{
        "epochs": 2, "batch_size": 3, "eval_batch_size": 1,
        "d_model": 8, "n_heads": 2, "n_layers": 1,
        "dropout": 0.0, "num_threads": 2, **overrides,
    })


class ConstantLogits(nn.Module):
    def __init__(self, values=(1.5, -0.5, 0.2)):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(values))

    def forward(self, histories, lengths):
        return self.logits.unsqueeze(0).expand(len(histories), -1)


def test_epoch_weights_loss_by_examples_and_keeps_remainder():
    targets = torch.tensor([1, 1, 2, 3, 3])
    dataset = [{
        "history_ids": torch.tensor([1]), "lengths": torch.tensor(1),
        "target_ids": target,
    } for target in targets]
    model = ConstantLogits()
    expected = nn.functional.cross_entropy(model.logits.expand(5, -1), targets - 1).item()
    # Zero learning rate isolates accounting from parameter updates.
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    result = training.train_epoch(model, DataLoader(dataset, batch_size=2), optimizer, "cpu")
    assert result["examples_seen"] == 5
    assert result["optimizer_steps"] == 3
    assert result["train_loss"] == pytest.approx(expected)


def test_epoch_rejects_nonfinite_loss():
    model = ConstantLogits((float("nan"), 0.0, 1.0))
    loader = DataLoader([{
        "history_ids": torch.tensor([1]), "lengths": torch.tensor(1),
        "target_ids": torch.tensor(3),
    }], batch_size=1)
    with pytest.raises(RuntimeError, match="Non-finite training loss"):
        training.train_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.01), "cpu")


def test_subset_is_unique_seeded_and_does_not_change_original():
    dataset = list(range(100))
    first, scope = training.select_training_examples(dataset, 11, 42)
    same, same_scope = training.select_training_examples(dataset, 11, 42)
    different, _ = training.select_training_examples(dataset, 11, 43)
    assert first.indices == same.indices
    assert first.indices != different.indices
    assert len(set(first.indices)) == 11
    assert scope == same_scope
    assert scope["scope"] == "trial_subset"
    assert scope["available_examples"] == len(dataset) == 100
    assert scope["examples_per_epoch"] == 11
    full, full_scope = training.select_training_examples(dataset, 200, 42)
    assert full is dataset
    assert full_scope["scope"] == "full_training"
    assert full_scope["subset_indices_sha256"] is None


def test_checkpoint_round_trip_and_fingerprint_guard(tmp_path):
    torch.manual_seed(9)
    settings = {"num_items": 6, "max_len": 3, "d_model": 8, "n_heads": 2, "n_layers": 1}
    model = TransformerRecommender(**settings)
    histories, lengths = np.array([[1, 2, 0]]), np.array([2])
    expected = make_score_batch(model)(histories, lengths)
    path = tmp_path / "best.pt"
    training.save_checkpoint(path, model, settings, "expected-fingerprint", 2, {"ndcg@10": 0.4})
    # Mutating the live model must not alter the saved best weights.
    with torch.no_grad():
        model.item_embedding.weight.add_(5)
    restored, payload = training.load_transformer_checkpoint(path, "expected-fingerprint")
    np.testing.assert_allclose(make_score_batch(restored)(histories, lengths), expected)
    assert payload["epoch"] == 2
    assert not restored.training
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        training.load_transformer_checkpoint(path, "wrong-fingerprint")


def test_end_to_end_training_saves_readable_validation_only(prepared, tmp_path, monkeypatch):
    raw, preparation = prepared
    output = tmp_path / "run"
    stages = []
    original = training.build_evaluation_examples

    def record_stage(ratings, splits, vocabulary, stage, max_len):
        stages.append(stage)
        return original(ratings, splits, vocabulary, stage, max_len)

    monkeypatch.setattr(training, "build_evaluation_examples", record_stage)
    report = training.run_training(raw, preparation, output, tiny_config(), show_progress=False)
    assert stages == ["validation"]
    assert report["split"] == "validation"
    assert report["training_scope"]["scope"] == "full_training"
    assert report["coverage"]["eligible_targets"] == report["metrics"]["users"] == 2
    assert report["training_examples_seen_total"] == 8  # Four prefixes each epoch.
    assert report["optimizer_steps_total"] == 4  # Includes the short final batch.
    assert report["epochs_completed"] == 2
    assert report["best_epoch"] in (1, 2)
    assert report["metrics"] == report["checkpoint_selection_metrics"]
    assert {p.name for p in output.iterdir()} == {
        "config.json", "history.jsonl", "best.pt",
        "validation_metrics.json", "validation_predictions.jsonl",
    }
    predictions = [json.loads(line) for line in
                   (output / "validation_predictions.jsonl").read_text().splitlines()]
    assert [p["target_movie_id"] for p in predictions] == [40, 10]
    assert all(10 <= movie <= 60 for p in predictions for movie in p["top_movie_ids"])
    saved = json.loads((output / "validation_metrics.json").read_text())
    assert saved == report
    assert not any("test_" in path.name for path in output.iterdir())


def test_selection_uses_ndcg_keeps_ties_and_reloads_best(prepared, tmp_path, monkeypatch):
    raw, preparation = prepared
    scores = iter([0.5, 0.5, 0.4, 0.5])  # Three epochs, then best-checkpoint reload.
    recall = iter([0.1, 0.9, 1.0, 0.1])  # Higher Recall must not override NDCG selection.
    seen_states = []
    original_evaluate = training.evaluate_validation

    def controlled_validation(model, *args):
        result, seconds = original_evaluate(model, *args)
        result["metrics"]["ndcg@10"] = next(scores)
        result["metrics"]["recall@10"] = next(recall)
        seen_states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return result, seconds

    monkeypatch.setattr(training, "evaluate_validation", controlled_validation)
    output = tmp_path / "selection"
    report = training.run_training(
        raw, preparation, output, tiny_config(epochs=10, patience=2), show_progress=False
    )
    assert report["best_epoch"] == 1
    assert report["epochs_completed"] == 3
    assert report["stopped_early"]
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert [row["saved_best"] for row in history] == [True, False, False]
    assert not torch.equal(seen_states[0]["item_embedding.weight"], seen_states[2]["item_embedding.weight"])
    for key in seen_states[0]:
        torch.testing.assert_close(seen_states[0][key], seen_states[-1][key], rtol=0, atol=0)


def test_same_seed_reproduces_cpu_run(prepared, tmp_path):
    raw, preparation = prepared
    first_dir, second_dir = tmp_path / "seed1", tmp_path / "seed2"
    first = training.run_training(raw, preparation, first_dir, tiny_config(), show_progress=False)
    second = training.run_training(raw, preparation, second_dir, tiny_config(), show_progress=False)
    assert first["metrics"] == second["metrics"]
    assert (first_dir / "validation_predictions.jsonl").read_bytes() == (second_dir / "validation_predictions.jsonl").read_bytes()
    first_model, _ = training.load_transformer_checkpoint(first_dir / "best.pt", first["preparation_fingerprint"])
    second_model, _ = training.load_transformer_checkpoint(second_dir / "best.pt", second["preparation_fingerprint"])
    for key, value in first_model.state_dict().items():
        torch.testing.assert_close(value, second_model.state_dict()[key], rtol=0, atol=0)


def test_trial_limits_training_but_keeps_full_validation(prepared, tmp_path):
    raw, preparation = prepared
    report = training.run_training(
        raw, preparation, tmp_path / "trial", tiny_config(epochs=1, max_train_examples=2),
        show_progress=False,
    )
    assert report["training_scope"]["scope"] == "trial_subset"
    assert report["training_examples_seen_total"] == 2
    assert report["catalogue_movies"] == 6
    assert report["metrics"]["users"] == 2


def test_cli_runs_with_validation_only_defaults(prepared, tmp_path, monkeypatch, capsys):
    raw, preparation = prepared
    output = tmp_path / "cli"
    monkeypatch.setattr(sys, "argv", [
        "src.train", "--data-dir", str(raw), "--preparation", str(preparation),
        "--output-dir", str(output), "--epochs", "1", "--d-model", "8",
        "--n-layers", "1", "--num-threads", "2", "--quiet",
    ])
    training.main()
    assert '"split": "validation"' in capsys.readouterr().out
    assert (output / "best.pt").exists()


def test_existing_run_is_preserved(tmp_path):
    existing = tmp_path / "run"
    existing.mkdir()
    sentinel = existing / "config.json"
    sentinel.write_text("preserve this")
    with pytest.raises(FileExistsError, match="choose a new run directory"):
        training.run_training("unused", "unused", existing, tiny_config())
    assert sentinel.read_text() == "preserve this"


@pytest.mark.parametrize("values", [
    {"epochs": 0}, {"max_train_examples": 0}, {"learning_rate": float("nan")},
    {"learning_rate": 0}, {"weight_decay": -1}, {"d_model": 7},
])
def test_invalid_settings_are_rejected(values):
    with pytest.raises(ValueError):
        tiny_config(**values).validate()
