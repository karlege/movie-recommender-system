"""History perturbation invariants and saved-checkpoint diagnostic checks."""

import json
import sys

import numpy as np
import pytest
import torch

import src.diagnose as diagnose
from src.data import Interaction, UserSplit
from src.dataset import EvaluationExample, MovieVocabulary, main as prepare_data
from src.evaluate import evaluate_examples
from src.train import TrainConfig, run_training


def example_users():
    return [
        EvaluationExample(1, np.array([1, 2, 3, 4, 5, 6, 0, 0]), 6, 9,
                          np.array([1, 2, 3, 4, 5, 6, 11])),
        EvaluationExample(2, np.array([7, 0, 0, 0, 0, 0, 0, 0]), 1, 8,
                          np.array([7, 12])),
    ], {1: np.array([10, 10, 20, 20, 30, 30]), 2: np.array([50])}


@pytest.mark.parametrize("mode", ["shuffle_ties", "shuffle_all"])
def test_shuffle_preserves_movies_padding_targets_exclusions_and_inputs(mode):
    examples, timestamps = example_users()
    originals = [example.history_ids.copy() for example in examples]
    result, changes = diagnose.perturb_examples(examples, timestamps, mode, 42)
    for original, perturbed, saved in zip(examples, result, originals):
        np.testing.assert_array_equal(original.history_ids, saved)
        np.testing.assert_array_equal(
            np.sort(perturbed.history_ids[:original.length]),
            np.sort(original.history_ids[:original.length]),
        )
        np.testing.assert_array_equal(perturbed.history_ids[original.length:], 0)
        np.testing.assert_array_equal(perturbed.seen_ids, original.seen_ids)
        assert perturbed.target_id == original.target_id
        assert perturbed.user_id == original.user_id
        assert perturbed.length == original.length
        assert perturbed.target_id not in perturbed.history_ids
    assert changes["shuffleable_histories"] == 1
    np.testing.assert_array_equal(result[1].history_ids, originals[1])


def test_tie_shuffle_never_moves_a_movie_across_timestamp_groups():
    examples, timestamps = example_users()
    for seed in (1, 2, 42):
        result, _ = diagnose.perturb_examples(examples, timestamps, "shuffle_ties", seed)
        for start in (0, 2, 4):
            np.testing.assert_array_equal(
                np.sort(result[0].history_ids[start:start + 2]),
                examples[0].history_ids[start:start + 2],
            )


@pytest.mark.parametrize("mode", ["shuffle_ties", "shuffle_all"])
def test_shuffles_are_seeded_and_independent_of_user_iteration_order(mode):
    examples, timestamps = example_users()
    first, first_changes = diagnose.perturb_examples(examples, timestamps, mode, 42)
    second, second_changes = diagnose.perturb_examples(examples, timestamps, mode, 42)
    reversed_result, _ = diagnose.perturb_examples(examples[::-1], timestamps, mode, 42)
    for left, right, reversed_right in zip(first, second, reversed_result[::-1]):
        np.testing.assert_array_equal(left.history_ids, right.history_ids)
        np.testing.assert_array_equal(left.history_ids, reversed_right.history_ids)
    assert first_changes == second_changes
    alternatives = [diagnose.perturb_examples(examples, timestamps, mode, seed)[0][0].history_ids
                    for seed in range(10)]
    assert any(not np.array_equal(alternatives[0], value) for value in alternatives[1:])


def test_timestamps_follow_the_actual_retained_window():
    events = tuple(Interaction(movie, timestamp) for movie, timestamp in
                   zip(range(1, 7), (10, 10, 20, 20, 30, 30)))
    splits = {1: UserSplit(1, events, Interaction(7, 40), Interaction(8, 50))}
    vocabulary = MovieVocabulary(range(1, 9))
    example = EvaluationExample(1, np.array([4, 5, 6, 0]), 3, 7, np.arange(1, 7))
    times = diagnose.retained_timestamps([example], splits, vocabulary)
    np.testing.assert_array_equal(times[1], [20, 30, 30])
    result, _ = diagnose.perturb_examples([example], times, "shuffle_ties", 42)
    assert result[0].history_ids[0] == 4
    assert set(result[0].history_ids[1:3]) == {5, 6}
    assert result[0].history_ids[3] == 0
    with pytest.raises(ValueError, match="alignment mismatch"):
        wrong = EvaluationExample(1, np.array([1, 5, 6, 0]), 3, 7, np.arange(1, 7))
        diagnose.retained_timestamps([wrong], splits, vocabulary)


def test_shuffle_leaves_histories_without_ties_unchanged():
    examples, _ = example_users()
    times = {1: np.arange(6), 2: np.array([10])}
    result, changes = diagnose.perturb_examples(examples, times, "shuffle_ties", 42)
    assert changes == {"shuffleable_histories": 0, "changed_histories": 0, "changed_positions": 0}
    for original, perturbed in zip(examples, result):
        np.testing.assert_array_equal(original.history_ids, perturbed.history_ids)


def test_popularity_rankings_are_identical_for_every_history_condition():
    examples, timestamps = example_users()
    counts = np.arange(20)

    def scorer(histories, lengths):
        return np.broadcast_to(counts, (len(histories), 20))

    original = evaluate_examples(examples, scorer, 20)
    for mode in ("shuffle_ties", "shuffle_all"):
        changed, _ = diagnose.perturb_examples(examples, timestamps, mode, 42)
        assert evaluate_examples(changed, scorer, 20) == original


def test_averages_preserve_user_count_and_reference_guard():
    rows = [
        {"users": 4, "recall@10": 0.25, "ndcg@10": 0.1},
        {"users": 4, "recall@10": 0.75, "ndcg@10": 0.3},
    ]
    averaged = diagnose.mean_metrics(rows)
    assert averaged["users"] == 4
    assert averaged["recall@10"] == 0.5
    assert averaged["ndcg@10"] == pytest.approx(0.2)
    diagnose.check_reference_metrics(rows[0], rows[0].copy())
    with pytest.raises(ValueError, match="differ from the saved checkpoint"):
        diagnose.check_reference_metrics(rows[0], rows[1])
    with pytest.raises(ValueError, match="same number of users"):
        diagnose.mean_metrics([rows[0], {**rows[1], "users": 5}])


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        (1, 10, 5, 10), (1, 20, 5, 10), (1, 30, 5, 20),
        (1, 40, 5, 20), (1, 50, 5, 30),
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
    preparation = processed / "preparation.json"
    run = tmp_path / "run"
    run_training(raw, preparation, run, TrainConfig(
        epochs=1, batch_size=2, eval_batch_size=1, d_model=8, n_layers=1,
        dropout=0, num_threads=2,
    ), show_progress=False)
    return raw, preparation, run


def test_end_to_end_diagnostics_reproduce_reference_and_use_validation_only(saved_run, tmp_path, monkeypatch):
    raw, preparation, run = saved_run
    checkpoint_bytes = (run / "best.pt").read_bytes()
    stages = []
    original = diagnose.build_evaluation_examples

    def record_stage(ratings, splits, vocabulary, stage, max_len):
        stages.append(stage)
        return original(ratings, splits, vocabulary, stage, max_len)

    monkeypatch.setattr(diagnose, "build_evaluation_examples", record_stage)
    output = tmp_path / "diagnostic"
    result = diagnose.run_diagnostics(
        raw, preparation, run, output, seeds=(42, 43), batch_size=1, num_threads=2
    )
    assert stages == ["validation"]
    assert result["original_matches_checkpoint"]
    assert result["checkpoint_epoch"] == 1
    assert result["coverage"]["eligible_targets"] == 2
    for mode in ("shuffle_ties", "shuffle_all"):
        assert result[mode]["mean_metrics"]["users"] == 2
        assert set(result[mode]["mean_by_time_relation"]) == {"strictly_later", "tied_timestamp"}
        assert [repeat["seed"] for repeat in result[mode]["repeats"]] == [42, 43]
    assert (run / "best.pt").read_bytes() == checkpoint_bytes
    assert json.loads((output / "diagnostics.json").read_text()) == result
    assert [p.name for p in output.iterdir()] == ["diagnostics.json"]


def test_saved_reference_mismatch_blocks_diagnostics(saved_run, tmp_path):
    raw, preparation, run = saved_run
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    # All six candidates fit in top ten, so the actual Recall@10 is 1.0.
    checkpoint["validation_metrics"]["recall@10"] = 0.0
    torch.save(checkpoint, run / "best.pt")
    with pytest.raises(ValueError, match="differ from the saved checkpoint"):
        diagnose.run_diagnostics(raw, preparation, run, tmp_path / "blocked", num_threads=2)
    assert not (tmp_path / "blocked").exists()


def test_cli_runs_with_default_three_shuffle_seeds(saved_run, tmp_path, monkeypatch, capsys):
    raw, preparation, run = saved_run
    output = tmp_path / "cli"
    monkeypatch.setattr(sys, "argv", [
        "src.diagnose", "--data-dir", str(raw), "--preparation", str(preparation),
        "--run-dir", str(run), "--output-dir", str(output), "--num-threads", "2",
    ])
    diagnose.main()
    assert '"original_matches_checkpoint": true' in capsys.readouterr().out
    result = json.loads((output / "diagnostics.json").read_text())
    assert result["shuffle_seeds"] == [42, 43, 44]
