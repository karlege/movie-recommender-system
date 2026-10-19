"""Small, synthetic correctness checks; no MovieLens download required.

Works with pytest or Python's standard unittest runner.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import (
    RATING_COLUMNS,
    load_movielens,
    split_positive_histories,
    summarize_data,
)


def make_ratings(rows):
    return pd.DataFrame(rows, columns=RATING_COLUMNS)


class DataTests(unittest.TestCase):
    def setUp(self):
        # Intentionally unsorted, with a low rating between positive events.
        self.ratings = make_ratings([
            (1, 50, 5, 500),
            (1, 20, 4, 200),
            (1, 99, 2, 250),
            (1, 10, 5, 100),
            (1, 40, 4, 400),
            (1, 30, 5, 300),
        ])

    def test_chronological_holdout_and_training_pairs(self):
        split = split_positive_histories(self.ratings)[1]
        train_ids = [e.movie_id for e in split.train]
        self.assertEqual(train_ids, [10, 20, 30])
        self.assertEqual(split.validation.movie_id, 40)
        self.assertEqual(split.test.movie_id, 50)
        # Training windows must be derived from the training portion only.
        pairs = [(train_ids[:i], train_ids[i]) for i in range(1, len(train_ids))]
        self.assertEqual(pairs, [([10], 20), ([10, 20], 30)])
        self.assertLess(split.train[-1].timestamp, split.validation.timestamp)
        self.assertLess(split.validation.timestamp, split.test.timestamp)

    def test_low_ratings_and_original_table_are_preserved(self):
        original = self.ratings.copy(deep=True)
        split = split_positive_histories(self.ratings)[1]
        pd.testing.assert_frame_equal(self.ratings, original)
        self.assertIn(99, self.ratings["movieId"].tolist())
        self.assertNotIn(99, [e.movie_id for e in split.train])

    def test_short_positive_histories_are_excluded(self):
        short_user = make_ratings([(2, i, 4, i) for i in (1, 2, 3, 4)])
        combined = pd.concat([self.ratings, short_user], ignore_index=True)
        self.assertEqual(set(split_positive_histories(combined)), {1})

    def test_users_are_split_independently(self):
        other_user = make_ratings([(2, i, 5, i) for i in (1, 2, 3, 4, 5)])
        combined = pd.concat([self.ratings, other_user], ignore_index=True)
        splits = split_positive_histories(combined)
        self.assertEqual(splits[1].test.movie_id, 50)
        self.assertEqual(splits[2].test.movie_id, 5)
        self.assertEqual([e.movie_id for e in splits[2].train], [1, 2, 3])

    def test_timestamp_ties_use_movie_ids_not_input_row_order(self):
        tied = make_ratings([(1, i, 5, 100) for i in (50, 20, 40, 10, 30)])
        split = split_positive_histories(tied)[1]
        reversed_split = split_positive_histories(tied.iloc[::-1])[1]
        self.assertEqual(split, reversed_split)
        self.assertEqual([e.movie_id for e in split.train], [10, 20, 30])
        self.assertEqual(split.validation.movie_id, 40)
        self.assertEqual(split.test.movie_id, 50)

    def test_duplicate_user_movie_is_rejected(self):
        duplicated = pd.concat([self.ratings, self.ratings.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Duplicate user-movie"):
            split_positive_histories(duplicated)

    def test_invalid_settings_and_rating_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive_threshold"):
            split_positive_histories(self.ratings, positive_threshold=6)
        with self.assertRaisesRegex(ValueError, "min_positive"):
            split_positive_histories(self.ratings, min_positive=3)
        invalid = self.ratings.copy()
        invalid.loc[0, "rating"] = 6
        with self.assertRaisesRegex(ValueError, "Ratings must"):
            split_positive_histories(invalid)

    def test_no_eligible_users_is_handled(self):
        self.assertEqual(split_positive_histories(self.ratings, min_positive=10), {})

    def test_summary_counts_observations_not_repeated_prefixes(self):
        movies = pd.DataFrame({"movieId": [10, 20, 30, 40, 50, 99]})
        splits = split_positive_histories(self.ratings)
        summary = summarize_data(self.ratings, movies, splits, 4)
        self.assertEqual(summary["ratings"], 6)
        self.assertEqual(summary["positive_interactions"], 5)
        self.assertEqual(summary["training_positive_interactions"], 3)
        self.assertEqual(summary["training_prefix_examples"], 2)
        self.assertEqual(summary["test_targets_before_catalogue_filter"], 1)

    def test_loader_and_inspection_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = "\n".join(
                "::".join(map(str, row))
                for row in self.ratings.itertuples(index=False, name=None)
            )
            (root / "ratings.dat").write_text(rows + "\n", encoding="latin-1")
            metadata = "\n".join(
                f'{movie_id}::Amélie "example" (2001)::Comedy|Drama'
                for movie_id in (10, 20, 30, 40, 50, 99)
            )
            (root / "movies.dat").write_text(metadata + "\n", encoding="latin-1")
            ratings, movies = load_movielens(root)
            self.assertEqual(len(ratings), 6)
            self.assertEqual(movies.iloc[0]["title"], 'Amélie "example" (2001)')
            result = subprocess.run(
                [sys.executable, "-m", "src.data", "--data-dir", str(root)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=True,
            )
            first_json = result.stdout.split("\n\nSplit settings:")[0]
            self.assertEqual(json.loads(first_json)["eligible_users"], 1)
            self.assertIn("Validation movie: 40", result.stdout)

    def test_missing_input_files_have_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "ratings.dat"):
                load_movielens(directory)


if __name__ == "__main__":
    unittest.main()
