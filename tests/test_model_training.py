"""Both architecture choices, format-1 compatibility, and MLP diagnostics."""

import json
import sys

import numpy as np
import pytest
import torch

from src.dataset import main as prepare_data
from src.diagnose import run_diagnostics
from src.models.mlp import PooledMLP
from src.models.transformer import TransformerRecommender
from src.scoring import make_score_batch
import src.train as training


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        (1, 10, 5, 10), (1, 20, 5, 10), (1, 30, 5, 20),
        (1, 60, 1, 25), (1, 40, 5, 20), (1, 50, 5, 40),
        (2, 40, 5, 10), (2, 50, 5, 10), (2, 60, 5, 20),
        (2, 10, 5, 30), (2, 20, 5, 40),
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
            "--max-len", "3",
        ])
        prepare_data()
    return raw, processed / "preparation.json"


def tiny_config(model="mlp", **changes):
    return training.TrainConfig(**{
        "model": model, "epochs": 2, "batch_size": 3, "eval_batch_size": 1,
        "num_threads": 2, "d_model": 8, "mlp_hidden": 12,
        "n_layers": 1, "dropout": 0, **changes,
    })


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_common_training_and_checkpoint_loader(prepared, tmp_path, kind):
    raw, preparation = prepared
    output = tmp_path / kind
    report = training.run_training(raw, preparation, output, tiny_config(kind), show_progress=False)
    assert report["model_type"] == kind
    assert report["model"] == ("pooled_mlp" if kind == "mlp" else "causal_transformer")
    assert report["metrics"]["users"] == 2
    assert report["training_examples_seen_total"] == 8
    assert report["metrics"] == report["checkpoint_selection_metrics"]
    model, checkpoint = training.load_checkpoint(output / "best.pt", report["preparation_fingerprint"])
    assert isinstance(model, PooledMLP if kind == "mlp" else TransformerRecommender)
    assert checkpoint["format_version"] == 2
    assert checkpoint["model_type"] == kind
    assert checkpoint["epoch"] == report["best_epoch"]
    assert not model.training
    if kind == "mlp":
        assert set(report["model_config"]) == {"num_items", "max_len", "d_model", "hidden", "dropout"}
        with pytest.raises(ValueError, match="Expected a transformer checkpoint"):
            training.load_transformer_checkpoint(output / "best.pt", report["preparation_fingerprint"])


def test_original_format_one_transformer_loads_without_changing_file(tmp_path):
    torch.manual_seed(42)
    settings = {"num_items": 6, "max_len": 3, "d_model": 8, "n_heads": 2, "n_layers": 1}
    original = TransformerRecommender(**settings).eval()
    path = tmp_path / "original_best.pt"
    torch.save({
        "format_version": 1, "model_state_dict": original.state_dict(),
        "model_config": settings, "preparation_fingerprint": "original-fingerprint",
        "epoch": 5, "validation_metrics": {"users": 2, "ndcg@10": 0.16},
    }, path)
    saved_bytes = path.read_bytes()
    histories, lengths = np.array([[1, 2, 0]]), np.array([2])
    expected = make_score_batch(original)(histories, lengths)
    for loader in (training.load_checkpoint, training.load_transformer_checkpoint):
        model, checkpoint = loader(path, "original-fingerprint")
        assert checkpoint["model_type"] == "transformer"
        np.testing.assert_array_equal(make_score_batch(model)(histories, lengths), expected)
    assert path.read_bytes() == saved_bytes


@pytest.mark.parametrize("kind,settings", [
    ("mlp", {"num_items": 6, "max_len": 3, "d_model": 8, "hidden": 12}),
    ("transformer", {"num_items": 6, "max_len": 3, "d_model": 8, "n_heads": 2, "n_layers": 1}),
])
def test_new_checkpoint_rejects_wrong_data_protocol(tmp_path, kind, settings):
    model = training.build_model(kind, settings)
    path = tmp_path / f"{kind}.pt"
    training.save_checkpoint(path, model, settings, "right", 1, {"ndcg@10": 0.1})
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        training.load_checkpoint(path, "wrong")
    other_kind = "transformer" if kind == "mlp" else "mlp"
    with pytest.raises(ValueError, match="does not match"):
        training.save_checkpoint(path, model, settings, "right", 1, {}, model_type=other_kind)


def test_unknown_architecture_is_rejected(tmp_path):
    settings = {"num_items": 6, "d_model": 8, "hidden": 12}
    model = PooledMLP(**settings)
    path = tmp_path / "bad.pt"
    training.save_checkpoint(path, model, settings, "right", 1, {})
    payload = torch.load(path, weights_only=True)
    payload["model_type"] = "unknown-model"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Unknown model type"):
        training.load_checkpoint(path, "right")


def test_mlp_does_not_require_head_divisibility():
    tiny_config(d_model=7).validate()
    with pytest.raises(ValueError, match="divisible by n_heads"):
        tiny_config("transformer", d_model=7).validate()


def test_same_seed_repeats_mlp_training(prepared, tmp_path):
    raw, preparation = prepared
    paths = [tmp_path / "first", tmp_path / "second"]
    reports = [training.run_training(raw, preparation, path, tiny_config(), show_progress=False)
               for path in paths]
    assert reports[0]["metrics"] == reports[1]["metrics"]
    assert (paths[0] / "validation_predictions.jsonl").read_bytes() == (paths[1] / "validation_predictions.jsonl").read_bytes()
    models = [training.load_checkpoint(path / "best.pt", reports[0]["preparation_fingerprint"])[0]
              for path in paths]
    for name, tensor in models[0].state_dict().items():
        torch.testing.assert_close(tensor, models[1].state_dict()[name], rtol=0, atol=0)


def test_mlp_order_diagnostics_reproduce_checkpoint(prepared, tmp_path):
    raw, preparation = prepared
    output = tmp_path / "mlp"
    training.run_training(raw, preparation, output, tiny_config(), show_progress=False)
    report = run_diagnostics(raw, preparation, output, tmp_path / "diagnostics", num_threads=2)
    assert report["model_type"] == "mlp"
    assert report["original_matches_checkpoint"]
    for mode in ("shuffle_ties", "shuffle_all"):
        assert report[mode]["mean_metrics"] == pytest.approx(report["original"]["metrics"])


def test_cli_selects_mlp(prepared, tmp_path, monkeypatch, capsys):
    raw, preparation = prepared
    output = tmp_path / "cli"
    monkeypatch.setattr(sys, "argv", [
        "src.train", "--model", "mlp", "--data-dir", str(raw),
        "--preparation", str(preparation), "--output-dir", str(output),
        "--epochs", "1", "--d-model", "8", "--mlp-hidden", "12",
        "--num-threads", "2", "--quiet",
    ])
    training.main()
    assert '"model_type": "mlp"' in capsys.readouterr().out
    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "mlp"
    assert config["model_config"]["hidden"] == 12
