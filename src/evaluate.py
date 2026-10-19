"""Shared exact ranking evaluation for one held-out positive per user.

The scoring callback receives histories and lengths only. It returns a
finite [batch, catalogue_size] NumPy array, with columns corresponding to
embedding IDs 1..M. This module owns candidate exclusions and all metrics.
"""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from src.dataset import EvaluationExample


def ranking_metrics(ranks: Sequence[int], k: int = 10) -> dict:
    """Mean Recall@k and NDCG@k for one relevant item per user."""
    if k < 1:
        raise ValueError("k must be positive.")
    ranks = np.asarray(ranks)
    if ranks.size == 0:
        return {"users": 0, f"recall@{k}": None, f"ndcg@{k}": None}
    if ranks.ndim != 1 or not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError("Ranks must be a one-dimensional integer sequence.")
    if (ranks < 1).any():
        raise ValueError("Ranks are 1-based and must be positive.")
    hits = ranks <= k
    discounts = np.where(hits, 1.0 / np.log2(ranks + 1), 0.0)
    return {
        "users": len(ranks),
        f"recall@{k}": float(hits.mean()),
        f"ndcg@{k}": float(discounts.mean()),
    }


def rank_candidates(scores, target_id: int, seen_ids, k: int = 10) -> dict:
    """Exclude seen items and rank by score descending, then item ID ascending.

    scores[0] belongs to embedding ID 1, not padding ID 0. Both seen_ids and
    target_id use embedding IDs. Padding is never a score column/candidate.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Scores must be a nonempty one-dimensional array.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite before candidate exclusions.")
    if k < 1:
        raise ValueError("k must be positive.")
    if not 1 <= target_id <= len(scores):
        raise ValueError("Target must be a real embedding ID in 1..M.")
    seen_ids = np.asarray(seen_ids)
    if seen_ids.ndim != 1:
        raise ValueError("Seen IDs must be one-dimensional.")
    if seen_ids.size and not np.issubdtype(seen_ids.dtype, np.integer):
        raise ValueError("Seen IDs must be integers.")
    seen_ids = seen_ids.astype(np.int64, copy=False)
    if ((seen_ids < 1) | (seen_ids > len(scores))).any():
        raise ValueError("Seen IDs must be real embedding IDs in 1..M.")
    if target_id in seen_ids:
        raise ValueError("Target is already seen; inspect data construction.")

    available = np.ones(len(scores), dtype=bool)
    available[seen_ids - 1] = False
    columns = np.flatnonzero(available)
    # Last lexsort key is primary; lower item ID breaks equal scores.
    order = np.lexsort((columns, -scores[columns]))
    ranked_ids = columns[order] + 1
    rank = int(np.flatnonzero(ranked_ids == target_id)[0]) + 1
    return {
        "rank": rank,
        "top_ids": ranked_ids[:k].tolist(),
        "candidate_count": len(ranked_ids),
        "hit": bool(rank <= k),
        "ndcg": float(1.0 / np.log2(rank + 1)) if rank <= k else 0.0,
    }


def evaluate_examples(
    examples: Sequence[EvaluationExample],
    score_batch: Callable[[np.ndarray, np.ndarray], np.ndarray],
    num_items: int,
    k: int = 10,
    batch_size: int = 128,
    cohorts: Mapping[int, str] | None = None,
) -> dict:
    """Evaluate every supplied user equally, independent of batch size.

    The scoring callback is deliberately not given targets, seen-item sets,
    or user IDs. Future models should encode only the supplied histories.
    Cohorts provide supplementary diagnostics without changing the main set.
    """
    if min(num_items, k, batch_size) < 1:
        raise ValueError("num_items, k, and batch_size must be positive.")
    user_ids = [example.user_id for example in examples]
    if len(set(user_ids)) != len(user_ids):
        raise ValueError("Expected exactly one held-out target per user.")
    if cohorts is not None and any(user not in cohorts for user in user_ids):
        raise ValueError("Every evaluated user needs a cohort label.")

    records = []
    cohort_ranks = defaultdict(list)
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        histories = np.stack([example.history_ids for example in batch])
        lengths = np.asarray([example.length for example in batch], dtype=np.int64)
        scores = np.asarray(score_batch(histories, lengths))
        if scores.shape != (len(batch), num_items):
            raise ValueError(
                f"Scorer returned shape {scores.shape}; expected "
                f"{(len(batch), num_items)}. Do not include padding column 0."
            )
        for example, row_scores in zip(batch, scores):
            ranked = rank_candidates(row_scores, example.target_id, example.seen_ids, k)
            record = {
                "user_id": int(example.user_id),
                "target_id": int(example.target_id),
                **ranked,
            }
            if cohorts is not None:
                label = cohorts[example.user_id]
                record["time_relation"] = label
                cohort_ranks[label].append(ranked["rank"])
            records.append(record)

    return {
        "metrics": ranking_metrics([record["rank"] for record in records], k),
        "by_time_relation": {
            label: ranking_metrics(ranks, k)
            for label, ranks in sorted(cohort_ranks.items())
        },
        "predictions": records,
    }
