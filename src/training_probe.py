"""Fixed in-sample ranking probe: one trained prefix per validation user."""

from dataclasses import replace
import hashlib
import json

from src.dataset import build_evaluation_examples


def build_training_probe(ratings, splits, vocabulary, validation_examples, max_len):
    """Use each eligible user's final TRAINING target, never a held-out target.

    Reuse validation record construction only to apply identical history
    truncation and prior-raw-rating exclusions. These temporary split views
    do not change the splits used for optimization or validation.
    """
    views = {}
    for example in validation_examples:
        split = splits[example.user_id]
        if len(split.train) < 2:
            raise ValueError("A training probe needs a target and earlier history.")
        views[example.user_id] = replace(
            split, train=split.train[:-1], validation=split.train[-1]
        )
    examples, coverage = build_evaluation_examples(
        ratings, views, vocabulary, "validation", max_len
    )
    if len(examples) != len(validation_examples):
        raise ValueError("Every validation user must have an eligible training probe.")
    identity = [
        [e.user_id, e.target_id, e.history_ids.tolist(), e.seen_ids.tolist()]
        for e in examples
    ]
    return examples, {
        "scope": "last_training_prefix_per_validation_user",
        "description": "In-sample diagnostic subset; not all training prefixes",
        "users": len(examples),
        "coverage": coverage,
        "examples_sha256": hashlib.sha256(
            json.dumps(identity, separators=(",", ":")).encode()
        ).hexdigest(),
    }
