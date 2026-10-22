"""Final testing preserves frozen runs and the existing holdout protocol."""

import csv
import json
import math
import sys

import pytest

import src.final_evaluate as final
from src.dataset import main as prepare_data
from src.evaluate import ranking_metrics
from src.train import TrainConfig, run_training


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        (1, 10, 5, 10), (1, 20, 5, 20), (1, 30, 5, 30),
        (1, 40, 5, 40), (1, 60, 1, 45), (1, 50, 5, 50),
        (2, 40, 5, 10), (2, 50, 5, 20), (2, 60, 5, 30),
        (2, 10, 5, 40), (2, 20, 5, 40),
        (3, 10, 5, 10), (3, 20, 5, 20), (3, 30, 5, 30),
        (3, 40, 5, 40), (3, 70, 5, 50),
    ]
    (raw / "ratings.dat").write_text("\n".join("::".join(map(str, row)) for row in rows), encoding="latin-1")
    (raw / "movies.dat").write_text("\n".join(f"{i}::Movie {i}::Drama" for i in range(10, 71, 10)), encoding="latin-1")
    processed = tmp_path / "processed"
    with monkeypatch.context() as context:
        context.setattr(sys, "argv", ["src.dataset", "--data-dir", str(raw),
                                    "--output-dir", str(processed), "--max-len", "3"])
        prepare_data()
    preparation = processed / "preparation.json"
    runs = tmp_path / "runs"
    for kind in ("mlp", "transformer"):
        for seed in (42, 43):
            run_training(raw, preparation, runs / f"{kind}_20ep_seed{seed}_l4", TrainConfig(
                model=kind, seed=seed, epochs=1, batch_size=4, eval_batch_size=2,
                d_model=8, mlp_hidden=12, n_heads=2, n_layers=1, num_threads=2, dropout=0,
            ), show_progress=False)
    return raw, preparation, runs


def run(experiment, output, **changes):
    return final.run_final_evaluation(*experiment, output, seeds=(42, 43),
                                     num_threads=2, show_progress=False, **changes)


def test_end_to_end_frozen_test_comparison(experiment, tmp_path, monkeypatch):
    raw, preparation, runs = experiment
    snapshots = {path: path.read_bytes() for path in runs.rglob("*") if path.is_file()}
    original = final.build_evaluation_examples
    observed = []

    def check_test_history(ratings, splits, vocabulary, stage, max_len):
        assert stage == "test"
        examples, coverage = original(ratings, splits, vocabulary, stage, max_len)
        user_one = next(example for example in examples if example.user_id == 1)
        assert [vocabulary.decode(int(i)) for i in user_one.history_ids] == [20, 30, 40]
        observed.append(stage)
        return examples, coverage

    monkeypatch.setattr(final, "build_evaluation_examples", check_test_history)
    output = tmp_path / "final"
    report = run(experiment, output, batch_size=1)
    assert observed == ["test"]
    assert report["coverage"] == {"total_targets": 3, "eligible_targets": 2,
                                  "excluded_unknown_targets": 1, "excluded_empty_histories": 0}
    assert len(report["per_run"]) == 5  # Four trained checkpoints and one popularity fit.
    for row in report["per_run"]:
        folder = "popularity" if row["seed"] is None else f"{row['model_type']}_20ep_seed{row['seed']}_l4"
        predictions = [json.loads(line) for line in (output / folder / "test_predictions.jsonl").read_text().splitlines()]
        assert row["metrics"] == ranking_metrics([r["rank"] for r in predictions])
        user_one = next(p for p in predictions if p["user_id"] == 1)
        assert user_one["top_movie_ids"] == [50]  # Includes validation and low ratings in exclusions.
        assert set(row["by_time_relation"]) == {"strictly_later", "tied_timestamp"}
        if row["seed"] is not None:
            assert row["checkpoint_epoch"] == 1
    assert all(path.read_bytes() == contents for path, contents in snapshots.items())
    assert json.loads((output / "comparison.json").read_text()) == report
    assert len(list(csv.DictReader((output / "comparison.csv").open()))) == 9
    assert "test results" in (output / "comparison.md").read_text()


def test_preflight_fails_before_test_data_or_outputs(experiment, tmp_path, monkeypatch):
    runs = experiment[2]
    (runs / "transformer_20ep_seed43_l4" / "best.pt").unlink()
    monkeypatch.setattr(final, "build_evaluation_examples", lambda *args: pytest.fail("Test data read before preflight completed"))
    output = tmp_path / "final"
    with pytest.raises(FileNotFoundError):
        run(experiment, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["best_epoch", "code", "trial"])
def test_rejects_inconsistent_or_trial_runs(experiment, tmp_path, change):
    path = experiment[2] / "mlp_20ep_seed42_l4"
    target = path / ("validation_metrics.json" if change == "best_epoch" else "config.json")
    data = json.loads(target.read_text())
    if change == "best_epoch":
        data["best_epoch"] = 2
    elif change == "code":
        data["code_sha256"]["evaluate.py"] = "changed"
    else:
        data["training_scope"]["scope"] = "trial_subset"
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        run(experiment, tmp_path / "final")
    assert not (tmp_path / "final").exists()


def test_existing_output_is_preserved(experiment, tmp_path):
    output = tmp_path / "final"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError, match="not empty"):
        run(experiment, output)
    assert marker.read_text() == "keep"


def test_summary_uses_sample_sd():
    reports = [{"model_type": "mlp", "metrics": {"users": 10, "recall@10": value,
                "ndcg@10": value / 2}, "by_time_relation": {}} for value in (0.2, 0.4)]
    row = final.summarize(reports)[0]
    assert row["recall@10_mean"] == pytest.approx(0.3)
    assert row["recall@10_sample_sd"] == pytest.approx(math.sqrt(0.02))


def test_cli(experiment, tmp_path, monkeypatch, capsys):
    raw, preparation, runs = experiment
    output = tmp_path / "cli"
    monkeypatch.setattr(sys, "argv", ["src.final_evaluate", "--data-dir", str(raw),
        "--preparation", str(preparation), "--runs-root", str(runs), "--output-dir", str(output),
        "--seeds", "42", "43", "--num-threads", "2"])
    final.main()
    assert (output / "comparison.md").is_file()
    assert "Saved comparison" in capsys.readouterr().out
