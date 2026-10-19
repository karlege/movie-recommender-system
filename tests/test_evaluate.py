"""Reference tests for exact ranks, candidate masks, metrics, and batches."""

import unittest

import numpy as np

from src.dataset import EvaluationExample
from src.evaluate import evaluate_examples, rank_candidates, ranking_metrics


def example(user, target, seen=()):
    return EvaluationExample(
        user_id=user,
        history_ids=np.array([4, 0, 0], dtype=np.int64),
        length=1,
        target_id=target,
        seen_ids=np.asarray(seen, dtype=np.int64),
    )


class EvaluationTests(unittest.TestCase):
    def test_rank_one_ten_and_eleven(self):
        scores = np.arange(12, 0, -1)
        for target in (1, 10, 11):
            row = rank_candidates(scores, target, [], k=10)
            self.assertEqual(row["rank"], target)
            self.assertEqual(row["hit"], target <= 10)
        self.assertEqual(rank_candidates(scores, 1, [])["ndcg"], 1.0)
        self.assertAlmostEqual(
            rank_candidates(scores, 10, [])["ndcg"], 1 / np.log2(11)
        )
        self.assertEqual(rank_candidates(scores, 11, [])["ndcg"], 0.0)

    def test_masked_high_scores_never_enter_recommendations(self):
        ranked = rank_candidates([100, 99, 5, 4], 3, [1, 2], k=10)
        self.assertEqual(ranked["rank"], 1)
        self.assertEqual(ranked["top_ids"], [3, 4])
        self.assertEqual(ranked["candidate_count"], 2)

    def test_scores_tied_by_ascending_real_item_id(self):
        ranked = rank_candidates([1, 1, 1, 1, 1], 3, [1])
        self.assertEqual(ranked["top_ids"], [2, 3, 4, 5])
        self.assertEqual(ranked["rank"], 2)

    def test_indices_are_one_based_and_padding_is_excluded(self):
        self.assertEqual(rank_candidates([1, 9, 2], 2, [1])["rank"], 1)
        with self.assertRaisesRegex(ValueError, "real embedding"):
            rank_candidates([1, 2], 0, [])
        with self.assertRaisesRegex(ValueError, "real embedding"):
            rank_candidates([1, 2], 1, [0])

    def test_invalid_scores_targets_and_settings_are_rejected(self):
        for scores in ([1, np.nan], [1, np.inf], [1, -np.inf]):
            with self.assertRaisesRegex(ValueError, "finite"):
                rank_candidates(scores, 1, [])
        with self.assertRaisesRegex(ValueError, "already seen"):
            rank_candidates([1, 2], 1, [1])
        with self.assertRaisesRegex(ValueError, "positive"):
            rank_candidates([1, 2], 1, [], k=0)
        with self.assertRaises(ValueError):
            rank_candidates([[1, 2]], 1, [])

    def test_scoring_array_is_not_modified(self):
        scores = np.array([100.0, 5.0, 3.0])
        before = scores.copy()
        rank_candidates(scores, 2, [1])
        np.testing.assert_array_equal(scores, before)

    def test_metrics_average_users_not_batches_or_groups(self):
        metrics = ranking_metrics([1, 10, 11], k=10)
        self.assertEqual(metrics["users"], 3)
        self.assertAlmostEqual(metrics["recall@10"], 2 / 3)
        self.assertAlmostEqual(metrics["ndcg@10"], (1 + 1 / np.log2(11)) / 3)

    def test_empty_metrics_are_null_not_zero_or_nan(self):
        self.assertEqual(
            ranking_metrics([], k=10),
            {"users": 0, "recall@10": None, "ndcg@10": None},
        )
        with self.assertRaises(ValueError):
            ranking_metrics([0])
        with self.assertRaises(ValueError):
            ranking_metrics([1.5])

    def test_batched_results_match_single_user_evaluation(self):
        examples = [example(1, 1), example(2, 2), example(3, 3)]
        calls = []

        def scorer(histories, lengths):
            self.assertEqual(histories.shape[1], 3)
            self.assertTrue((lengths == 1).all())
            calls.append(len(histories))
            return np.broadcast_to([4, 3, 2, 1], (len(histories), 4))

        cohorts = {1: "tied_timestamp", 2: "strictly_later", 3: "strictly_later"}
        result = evaluate_examples(examples, scorer, 4, k=1, batch_size=2, cohorts=cohorts)
        self.assertEqual(calls, [2, 1])
        separate = evaluate_examples(examples, scorer, 4, k=1, batch_size=1, cohorts=cohorts)
        self.assertEqual(result, separate)
        self.assertAlmostEqual(result["metrics"]["recall@1"], 1 / 3)
        self.assertEqual(result["by_time_relation"]["tied_timestamp"]["recall@1"], 1.0)
        self.assertEqual(result["by_time_relation"]["strictly_later"]["recall@1"], 0.0)

    def test_scorer_shape_and_duplicate_users_are_checked(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            evaluate_examples([example(1, 1)], lambda h, l: np.ones((1, 5)), 4)
        with self.assertRaisesRegex(ValueError, "one held-out"):
            evaluate_examples([example(1, 1), example(1, 2)], lambda h, l: None, 4)
        with self.assertRaisesRegex(ValueError, "cohort label"):
            evaluate_examples([example(1, 1)], lambda h, l: None, 4, cohorts={})

    def test_empty_evaluation_does_not_call_the_scorer(self):
        def scorer(histories, lengths):
            self.fail("The scorer should not be called for no examples.")

        result = evaluate_examples([], scorer, 4)
        self.assertIsNone(result["metrics"]["recall@10"])
        self.assertEqual(result["predictions"], [])


if __name__ == "__main__":
    unittest.main()
