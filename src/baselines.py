"""Fit training-popularity counts and evaluate with the shared ranking code.

Validation evaluation:
    python -m src.baselines --split validation

The preprocessing fingerprint is checked before fitting or evaluation.
Test evaluation uses tthe same preparation.json andion and is
reserved for final reporting after validation-based model selection.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from src.data import load_movielens, split_positive_histories
from src.dataset import MovieVocabulary, build_evaluation_examples
from src.evaluate import evaluate_examples


def popularity_scores(splits, vocabulary: MovieVocabulary) -> np.ndarray:
    """One count per original positive TRAINING observation, not per prefix."""
    counts = np.zeros(len(vocabulary), dtype=np.int64)
    for split in splits.values():
        columns = np.asarray(
            [vocabulary.encode(event.movie_id) - 1 for event in split.train],
            dtype=np.int64,
        )
        np.add.at(counts, columns, 1)
    return counts


def time_relation_labels(splits, stage: str) -> dict[int, str]:
    """Labels use the original positive boundary, before history truncation."""
    if stage not in ("validation", "test"):
        raise ValueError("stage must be 'validation' or 'test'.")
    labels = {}
    for user_id, split in splits.items():
        previous = split.train[-1] if stage == "validation" else split.validation
        target = split.validation if stage == "validation" else split.test
        if target.timestamp < previous.timestamp:
            raise ValueError("A held-out target precedes its history.")
        labels[user_id] = (
            "tied_timestamp" if target.timestamp == previous.timestamp
            else "strictly_later"
        )
    return labels


def restore_preparation(data_dir: Path, preparation_path: Path):
    """Rebuild the saved preprocessing and reject stale/mismatched inputs."""
    data_dir, preparation_path = Path(data_dir), Path(preparation_path)
    manifest = json.loads(preparation_path.read_text(encoding="utf-8"))
    fields = ("protocol", "source_sha256", "movie_ids_in_index_order", "code_sha256")
    inputs = {field: manifest[field] for field in fields}
    expected = hashlib.sha256(
        json.dumps(inputs, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if manifest["fingerprint"] != expected:
        raise ValueError("Preparation fingerprint is inconsistent; rerun src.dataset.")
    protocol = manifest["protocol"]
    if protocol["version"] != "positive-history-v1":
        raise ValueError("Unsupported preprocessing protocol version.")

    for name, digest in manifest["source_sha256"].items():
        with (data_dir / name).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Source checksum mismatch: {name}.")
    for name, digest in manifest["code_sha256"].items():
        actual = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(
                f"Preprocessing code changed: {name}. Rerun src.dataset and review "
                "the resulting protocol/fingerprint before comparing groups."
            )

    ratings, _ = load_movielens(data_dir)
    splits = split_positive_histories(
        ratings, protocol["positive_threshold"], protocol["min_positive"]
    )
    vocabulary = MovieVocabulary.from_training_splits(splits)
    if list(vocabulary.movie_ids) != manifest["movie_ids_in_index_order"]:
        raise ValueError("Rebuilt vocabulary differs from the saved preparation.")
    return manifest, ratings, splits, vocabulary


def run_popularity(
    data_dir: Path,
    preparation_path: Path,
    output_dir: Path,
    stage: str = "validation",
    k: int = 10,
    batch_size: int = 128,
) -> dict:
    manifest, ratings, splits, vocabulary = restore_preparation(
        data_dir, preparation_path
    )
    examples, coverage = build_evaluation_examples(
        ratings, splits, vocabulary, stage,
        manifest["protocol"]["max_history_length"],
    )
    labels = time_relation_labels(splits, stage)
    started = perf_counter()
    counts = popularity_scores(splits, vocabulary)
    fit_seconds = perf_counter() - started

    def score_batch(history_ids, lengths):
        # Same item scores for everyone; per-user exclusions are handled only
        # by the evaluator. Histories/lengths are intentionally unused here.
        return np.broadcast_to(counts, (len(history_ids), len(vocabulary)))

    started = perf_counter()
    evaluation = evaluate_examples(
        examples, score_batch, len(vocabulary), k, batch_size, labels
    )
    evaluation_seconds = perf_counter() - started
    report = {
        "model": "training_popularity",
        "split": stage,
        "preparation_fingerprint": manifest["fingerprint"],
        "catalogue_movies": len(vocabulary),
        "training_positive_interactions_counted": int(counts.sum()),
        "coverage": coverage,
        "metrics": evaluation["metrics"],
        "by_time_relation": evaluation["by_time_relation"],
        "k": k,
        "batch_size": batch_size,
        "score_tie_break": "ascending embedding ID (also ascending original movie ID)",
        "trainable_parameters": 0,
        "stored_item_counts": len(counts),
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "evaluation_code_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("baselines.py", "evaluate.py")
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{stage}_metrics.json"
    predictions_path = output_dir / f"{stage}_predictions.jsonl"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in evaluation["predictions"]:
            readable = {
                **prediction,
                "target_movie_id": vocabulary.decode(prediction["target_id"]),
                "top_movie_ids": [vocabulary.decode(i) for i in prediction["top_ids"]],
            }
            handle.write(json.dumps(readable) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ml-1m"))
    parser.add_argument(
        "--preparation", type=Path, default=Path("data/processed/preparation.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/popularity"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    report = run_popularity(
        args.data_dir, args.preparation, args.output_dir,
        args.split, args.k, args.batch_size,
    )
    # Detailed environment/code hashes remain in the saved result file.
    shown = {key: report[key] for key in (
        "model", "split", "preparation_fingerprint", "catalogue_movies",
        "training_positive_interactions_counted", "coverage", "metrics",
        "by_time_relation", "fit_seconds", "evaluation_seconds",
    )}
    print(json.dumps(shown, indent=2))
    print("\nSaved:", args.output_dir / f"{args.split}_metrics.json")
    print("Saved:", args.output_dir / f"{args.split}_predictions.jsonl")


if __name__ == "__main__":
    main()
