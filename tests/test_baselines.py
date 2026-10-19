"""Popularity counts and an end-to-end synthetic preparation/evaluation run."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.baselines import popularity_scores, restore_preparation, run_popularity, time_relation_labels
from src.data import Interaction, UserSplit
from src.dataset import MovieVocabulary


def make_split(user, movies, timestamps=(10, 20, 30, 40, 50)):
    events = tuple(Interaction(movie, time) for movie, time in zip(movies, timestamps))
    return UserSplit(user, events[:3], events[3], events[4])


def prepare_fixture(root):
    rows = [
        (1, 10, 5, 10), (1, 20, 5, 20), (1, 30, 5, 30),
        (1, 60, 1, 35), (1, 40, 5, 40), (1, 50, 5, 50),
        (2, 40, 5, 10), (2, 50, 5, 20), (2, 60, 5, 30),
        (2, 10, 5, 40), (2, 20, 5, 50),
    ]
    (root / "ratings.dat").write_text(
        "\n".join("::".join(map(str, row)) for row in rows), encoding="latin-1"
    )
    (root / "movies.dat").write_text(
        "\n".join(f"{movie}::Movie {movie} (2000)::Drama" for movie in range(10, 61, 10)),
        encoding="latin-1",
    )
    subprocess.run(
        [sys.executable, "-m", "src.dataset", "--data-dir", str(root),
         "--output-dir", str(root / "processed"), "--max-len", "2"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=True,
    )
    return root / "processed/preparation.json"


class BaselineTests(unittest.TestCase):
    def test_counts_training_observations_once(self):
        splits = {
            1: make_split(1, [10, 20, 30, 40, 99]),
            2: make_split(2, [10, 40, 50, 60, 98]),
        }
        vocabulary = MovieVocabulary.from_training_splits(splits)
        counts = popularity_scores(splits, vocabulary)
        self.assertEqual(int(counts.sum()), 6)
        self.assertEqual(counts[vocabulary.encode(10) - 1], 2)
        self.assertEqual(counts[vocabulary.encode(40) - 1], 1)  # Not +1 for user 1's val.
        np.testing.assert_array_equal(counts, [2, 1, 1, 1, 1])

    def test_time_labels_use_original_boundaries(self):
        splits = {
            1: make_split(1, [1, 2, 3, 4, 5], (10, 20, 30, 30, 50)),
            2: make_split(2, [1, 2, 3, 4, 5], (10, 20, 30, 40, 40)),
        }
        self.assertEqual(time_relation_labels(splits, "validation"), {
            1: "tied_timestamp", 2: "strictly_later"
        })
        self.assertEqual(time_relation_labels(splits, "test"), {
            1: "strictly_later", 2: "tied_timestamp"
        })
        with self.assertRaises(ValueError):
            time_relation_labels(splits, "other")

    def test_end_to_end_validation_and_readable_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preparation = prepare_fixture(root)
            report = run_popularity(root, preparation, root / "runs", k=1, batch_size=1)
            self.assertEqual(report["split"], "validation")
            self.assertEqual(report["catalogue_movies"], 6)
            self.assertEqual(report["training_positive_interactions_counted"], 6)
            self.assertEqual(report["metrics"], {"users": 2, "recall@1": 1.0, "ndcg@1": 1.0})
            predictions = [json.loads(line) for line in
                           (root / "runs/validation_predictions.jsonl").read_text().splitlines()]
            self.assertEqual(predictions[0]["top_movie_ids"], [40])
            self.assertEqual(predictions[0]["target_movie_id"], 40)
            self.assertEqual(predictions[1]["top_movie_ids"], [10])
            self.assertFalse((root / "runs/test_metrics.json").exists())

    def test_changed_raw_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preparation = prepare_fixture(root)
            with (root / "ratings.dat").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "Source checksum mismatch"):
                restore_preparation(root, preparation)

    def test_inconsistent_preparation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preparation = prepare_fixture(root)
            manifest = json.loads(preparation.read_text())
            manifest["protocol"]["max_history_length"] = 100
            preparation.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                restore_preparation(root, preparation)

    def test_cli_uses_validation_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preparation = prepare_fixture(root)
            result = subprocess.run(
                [sys.executable, "-m", "src.baselines", "--data-dir", str(root),
                 "--preparation", str(preparation), "--output-dir", str(root / "runs")],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, check=True,
            )
            self.assertIn('"split": "validation"', result.stdout)
            self.assertTrue((root / "runs/validation_metrics.json").exists())
            self.assertFalse((root / "runs/test_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
