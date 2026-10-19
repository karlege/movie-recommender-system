"""Vocabulary, prefix construction, padding, masks, and tie-audit checks."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import Interaction, RATING_COLUMNS, UserSplit, split_positive_histories
from src.dataset import (
    MovieVocabulary,
    PrefixDataset,
    build_evaluation_examples,
    pad_history,
    timestamp_tie_audit,
)


def make_split(user, train, validation, test):
    events = [Interaction(movie, (i + 1) * 10) for i, movie in enumerate(
        [*train, validation, test]
    )]
    return UserSplit(user, tuple(events[:-2]), events[-2], events[-1])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        # The low-rated movie 6 is known globally because another user liked
        # it during training, making its exclusion test meaningful.
        rows = [
            (1, 1, 5, 10), (1, 2, 4, 20), (1, 6, 2, 25),
            (1, 3, 5, 30), (1, 4, 5, 40), (1, 5, 5, 50),
            (1, 7, 1, 60),  # Future dislike must not enter test exclusions.
            (2, 4, 5, 10), (2, 5, 5, 20), (2, 6, 5, 30),
            (2, 7, 5, 40), (2, 8, 5, 50),
            (3, 7, 5, 10), (3, 8, 5, 20), (3, 9, 5, 30),
            (3, 100, 5, 40), (3, 101, 5, 50),  # Held-out-only items.
        ]
        self.ratings = pd.DataFrame(rows, columns=RATING_COLUMNS)
        self.splits = split_positive_histories(self.ratings)
        self.vocab = MovieVocabulary.from_training_splits(self.splits)

    def test_vocabulary_uses_training_only(self):
        self.assertEqual(self.vocab.movie_ids, tuple(range(1, 10)))
        self.assertNotIn(100, self.vocab.movie_to_index)
        self.assertNotIn(101, self.vocab.movie_to_index)
        self.assertEqual(self.vocab.embedding_size, 10)

    def test_vocabulary_is_deterministic_and_round_trips(self):
        vocab = MovieVocabulary([40, 10, 40, 900])
        self.assertEqual(vocab.movie_ids, (10, 40, 900))
        self.assertEqual(vocab.encode(40), 2)
        self.assertEqual(vocab.decode(2), 40)
        with self.assertRaises(KeyError):
            vocab.encode(999)
        with self.assertRaises(ValueError):
            vocab.decode(0)
        with self.assertRaises(ValueError):
            MovieVocabulary([])

    def test_padding_and_recent_history(self):
        padded, length = pad_history([1, 2], 4)
        np.testing.assert_array_equal(padded, [1, 2, 0, 0])
        self.assertEqual(length, 2)
        padded, length = pad_history([1, 2, 3, 4], 2)
        np.testing.assert_array_equal(padded, [3, 4])
        self.assertEqual(length, 2)
        self.assertEqual(padded.dtype, np.int64)

    def test_empty_invalid_or_padding_input_is_rejected(self):
        for indices, max_len in (([], 3), ([1, 0], 3), ([1], 0)):
            with self.assertRaises(ValueError):
                pad_history(indices, max_len)

    def test_prefix_targets_and_padding(self):
        dataset = PrefixDataset(self.splits, self.vocab, max_len=4)
        self.assertEqual(len(dataset), 6)
        first, second = dataset[0], dataset[1]
        np.testing.assert_array_equal(first["history_ids"], [1, 0, 0, 0])
        self.assertEqual(first["target_ids"], 2)
        np.testing.assert_array_equal(second["history_ids"], [1, 2, 0, 0])
        self.assertEqual(second["target_ids"], 3)
        self.assertEqual(second["lengths"], 2)
        self.assertEqual(dataset[2]["user_id"], 2)
        self.assertEqual(dataset[-1]["target_ids"], 9)
        with self.assertRaises(IndexError):
            dataset[len(dataset)]

    def test_prefix_truncation_does_not_change_target(self):
        split = make_split(1, [10, 20, 30, 40], 50, 60)
        vocab = MovieVocabulary.from_training_splits({1: split})
        sample = PrefixDataset({1: split}, vocab, max_len=2)[2]
        np.testing.assert_array_equal(sample["history_ids"], [2, 3])
        self.assertEqual(sample["target_ids"], 4)

    def test_training_does_not_use_this_users_holdouts(self):
        dataset = PrefixDataset(self.splits, self.vocab)
        for i in (0, 1):
            sample = dataset[i]
            self.assertNotIn(self.vocab.encode(4), sample["history_ids"])
            self.assertNotIn(self.vocab.encode(5), sample["history_ids"])
            self.assertNotIn(sample["target_ids"], (4, 5))

    def test_validation_excludes_past_low_ratings_but_not_future(self):
        examples, counts = build_evaluation_examples(
            self.ratings, self.splits, self.vocab, "validation", max_len=2
        )
        user = next(e for e in examples if e.user_id == 1)
        np.testing.assert_array_equal(user.history_ids, [2, 3])
        self.assertEqual(set(user.seen_ids), {1, 2, 3, 6})
        self.assertNotIn(user.target_id, user.seen_ids)
        self.assertNotIn(5, user.seen_ids)
        self.assertNotIn(7, user.seen_ids)
        self.assertEqual(counts["eligible_targets"], 2)
        self.assertEqual(counts["excluded_unknown_targets"], 1)

    def test_test_history_adds_validation_and_masks_it(self):
        examples, _ = build_evaluation_examples(
            self.ratings, self.splits, self.vocab, "test", max_len=2
        )
        user = next(e for e in examples if e.user_id == 1)
        np.testing.assert_array_equal(user.history_ids, [3, 4])
        self.assertEqual(user.target_id, 5)
        self.assertEqual(set(user.seen_ids), {1, 2, 3, 4, 6})
        self.assertNotIn(5, user.seen_ids)
        self.assertNotIn(7, user.seen_ids)

    def test_unknown_validation_does_not_remove_known_test(self):
        splits = {
            1: make_split(1, [1, 2, 3], 100, 4),
            2: make_split(2, [4, 5, 6], 200, 201),
        }
        rows = [
            (uid, e.movie_id, 5, e.timestamp)
            for uid, split in splits.items()
            for e in (*split.train, split.validation, split.test)
        ]
        ratings = pd.DataFrame(rows, columns=RATING_COLUMNS)
        vocab = MovieVocabulary.from_training_splits(splits)
        validation, _ = build_evaluation_examples(ratings, splits, vocab, "validation")
        test, counts = build_evaluation_examples(ratings, splits, vocab, "test")
        self.assertEqual(validation, [])
        self.assertEqual([e.user_id for e in test], [1])
        self.assertEqual(test[0].length, 3)
        np.testing.assert_array_equal(test[0].history_ids[:3], [1, 2, 3])
        self.assertEqual(counts["excluded_unknown_targets"], 1)

    def test_timestamp_tie_masks_follow_same_order(self):
        tied = self.ratings.copy()
        tied.loc[(tied.userId == 1) & tied.movieId.isin([3, 4, 6]), "timestamp"] = 40
        splits = split_positive_histories(tied)
        examples, _ = build_evaluation_examples(tied, splits, self.vocab, "validation")
        user = next(e for e in examples if e.user_id == 1)
        self.assertIn(3, user.seen_ids)     # Same timestamp, lower movie ID.
        self.assertNotIn(6, user.seen_ids)  # Same timestamp, higher movie ID.
        audit = timestamp_tie_audit(splits)
        self.assertEqual(audit["validation_target_tied_with_previous_positive"], 1)

    def test_audit_counts_validation_and_test_ties_separately(self):
        events = tuple(Interaction(movie, 100) for movie in range(1, 6))
        split = UserSplit(1, events[:3], events[3], events[4])
        audit = timestamp_tie_audit({1: split})
        self.assertEqual(audit["validation_tied_fraction"], 1.0)
        self.assertEqual(audit["test_tied_fraction"], 1.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_pytorch_dataloader_batch_contract(self):
        import torch
        from torch.utils.data import DataLoader

        dataset = PrefixDataset(self.splits, self.vocab, max_len=4)
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
        self.assertEqual(batch["history_ids"].shape, (2, 4))
        self.assertEqual(batch["target_ids"].shape, (2,))
        self.assertTrue(all(t.dtype == torch.int64 for t in batch.values()))

    def test_cli_saves_a_reproducible_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lines = ["::".join(map(str, row)) for row in
                     self.ratings.itertuples(index=False, name=None)]
            (root / "ratings.dat").write_text("\n".join(lines), encoding="latin-1")
            ids = sorted(self.ratings.movieId.unique())
            (root / "movies.dat").write_text(
                "\n".join(f"{movie}::Example (2000)::Drama" for movie in ids),
                encoding="latin-1",
            )
            command = [sys.executable, "-m", "src.dataset", "--data-dir", str(root),
                       "--output-dir", str(root / "processed"), "--max-len", "2"]
            cwd = Path(__file__).resolve().parents[1]
            subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)
            manifest_path = root / "processed/preparation.json"
            first = manifest_path.read_bytes()
            manifest = json.loads(first)
            self.assertEqual(manifest["movie_ids_in_index_order"], list(range(1, 10)))
            self.assertEqual(manifest["summary"]["training_prefix_examples"], 6)
            self.assertEqual(len(manifest["fingerprint"]), 64)
            subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)
            self.assertEqual(first, manifest_path.read_bytes())
            changed = command[:-1] + ["3"]
            subprocess.run(changed, cwd=cwd, capture_output=True, text=True, check=True)
            self.assertNotEqual(
                manifest["fingerprint"],
                json.loads(manifest_path.read_text())["fingerprint"],
            )


if __name__ == "__main__":
    unittest.main()
