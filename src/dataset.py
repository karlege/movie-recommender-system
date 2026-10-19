"""Build a training-only vocabulary, padded examples, and evaluation records.

Run from the repository root:
    python -m src.dataset --data-dir data/raw/ml-1m --check-batch

Data construction uses NumPy;
PyTorch's default DataLoader collator turns samples into integer tensors.
"""

import argparse
import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import Interaction, UserSplit, load_movielens, split_positive_histories


class MovieVocabulary:
    """Real movie indices are 1..M; 0 is reserved exclusively for padding."""

    def __init__(self, movie_ids):
        self.movie_ids = tuple(sorted(set(int(i) for i in movie_ids)))
        if not self.movie_ids or self.movie_ids[0] <= 0:
            raise ValueError("A vocabulary needs at least one positive movie ID.")
        self.movie_to_index = {
            movie_id: index for index, movie_id in enumerate(self.movie_ids, start=1)
        }

    @classmethod
    def from_training_splits(cls, splits: dict[int, UserSplit]):
        # Neither held-out targets nor the complete movies table fit this map.
        return cls(event.movie_id for s in splits.values() for event in s.train)

    def __len__(self):
        return len(self.movie_ids)

    @property
    def embedding_size(self):
        return len(self) + 1

    def encode(self, movie_id: int) -> int:
        return self.movie_to_index[movie_id]  # Unknown IDs raise KeyError.

    def decode(self, index: int) -> int:
        if not 1 <= index <= len(self):
            raise ValueError("Only real movie indices 1..M can be decoded.")
        return self.movie_ids[index - 1]

    def encode_known(self, events: tuple[Interaction, ...]) -> list[int]:
        # Unknown observed movies are dropped, not converted to padding tokens.
        return [
            self.encode(e.movie_id) for e in events if e.movie_id in self.movie_to_index
        ]


def pad_history(indices, max_len: int) -> tuple[np.ndarray, int]:
    """Keep the most recent max_len known items, then pad on the right."""
    if max_len < 1:
        raise ValueError("max_len must be positive.")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or (indices <= 0).any():
        raise ValueError("History must be a nonempty sequence of real movie indices.")
    recent = indices[-max_len:]
    padded = np.zeros(max_len, dtype=np.int64)
    padded[:len(recent)] = recent
    return padded, len(recent)


class PrefixDataset:
    """Map-style dataset: derive each prefix on demand instead of storing it.

    __len__ and __getitem__ are the interface PyTorch DataLoader needs.
    Only original training sequences and cumulative example counts are stored.
    """

    def __init__(self, splits, vocabulary: MovieVocabulary, max_len: int = 50):
        if max_len < 1:
            raise ValueError("max_len must be positive.")
        self.max_len = max_len
        self._users = []
        self._sequences = []
        self._ends = []
        total = 0
        for user_id in sorted(splits):
            # All these movies must be known; do not silently drop training IDs.
            sequence = np.asarray(
                [vocabulary.encode(e.movie_id) for e in splits[user_id].train],
                dtype=np.int64,
            )
            if len(sequence) < 2:
                raise ValueError("Every training user needs at least two movies.")
            self._users.append(user_id)
            self._sequences.append(sequence)
            total += len(sequence) - 1
            self._ends.append(total)

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        user_position = bisect_right(self._ends, index)
        previous_end = self._ends[user_position - 1] if user_position else 0
        target_position = index - previous_end + 1
        sequence = self._sequences[user_position]
        start = max(0, target_position - self.max_len)
        history_ids, length = pad_history(
            sequence[start:target_position], self.max_len
        )
        return {
            "user_id": np.int64(self._users[user_position]),
            "history_ids": history_ids,
            "lengths": np.int64(length),
            "target_ids": sequence[target_position],
        }


@dataclass
class EvaluationExample:
    user_id: int
    history_ids: np.ndarray
    length: int
    target_id: int
    seen_ids: np.ndarray  # Real embedding indices, NOT zero-based score columns.


def build_evaluation_examples(
    ratings: pd.DataFrame,
    splits: dict[int, UserSplit],
    vocabulary: MovieVocabulary,
    stage: str,
    max_len: int = 50,
) -> tuple[list[EvaluationExample], dict]:
    """Prepare a common warm-start evaluation set and prior-rated exclusions."""
    if stage not in ("validation", "test"):
        raise ValueError("stage must be 'validation' or 'test'.")
    if max_len < 1:
        raise ValueError("max_len must be positive.")

    # Group once: never rescan the full ratings table separately for each user.
    raw_by_user = {
        int(user): group[["movieId", "timestamp"]].to_numpy(dtype=np.int64)
        for user, group in ratings.groupby("userId", sort=False)
    }
    counts = {
        "total_targets": len(splits),
        "eligible_targets": 0,
        "excluded_unknown_targets": 0,
        "excluded_empty_histories": 0,
    }
    examples = []
    for user_id in sorted(splits):
        split = splits[user_id]
        target = split.validation if stage == "validation" else split.test
        if target.movie_id not in vocabulary.movie_to_index:
            counts["excluded_unknown_targets"] += 1
            continue
        history = split.train
        if stage == "test":
            history = (*history, split.validation)
        known = vocabulary.encode_known(history)
        if not known:
            counts["excluded_empty_histories"] += 1
            continue
        padded, length = pad_history(known, max_len)

        # Candidate exclusions use ALL earlier raw ratings, even low ratings
        # and items outside the retained positive-history window.
        raw = raw_by_user[user_id]
        movie_ids, timestamps = raw[:, 0], raw[:, 1]
        earlier = (timestamps < target.timestamp) | (
            (timestamps == target.timestamp) & (movie_ids < target.movie_id)
        )
        seen = sorted({
            vocabulary.encode(int(movie)) for movie in movie_ids[earlier]
            if int(movie) in vocabulary.movie_to_index
        })
        target_id = vocabulary.encode(target.movie_id)
        if target_id in seen:
            raise ValueError("A target is already seen; inspect duplicate ratings.")
        examples.append(EvaluationExample(
            user_id=user_id,
            history_ids=padded,
            length=length,
            target_id=target_id,
            seen_ids=np.asarray(seen, dtype=np.int64),
        ))
    counts["eligible_targets"] = len(examples)
    return examples, counts


def timestamp_tie_audit(splits: dict[int, UserSplit]) -> dict:
    """Audit ties at holdout boundaries separately from overall raw-data ties."""
    total = len(splits)
    validation_ties = sum(
        s.train[-1].timestamp == s.validation.timestamp for s in splits.values()
    )
    test_ties = sum(s.validation.timestamp == s.test.timestamp for s in splits.values())
    return {
        "users_before_catalogue_filter": total,
        "validation_target_tied_with_previous_positive": validation_ties,
        "test_target_tied_with_previous_positive": test_ties,
        "validation_tied_fraction": validation_ties / total if total else 0.0,
        "test_tied_fraction": test_ties / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ml-1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--positive-threshold", type=int, choices=range(1, 6), default=4)
    parser.add_argument("--min-positive", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--check-batch", action="store_true")
    args = parser.parse_args()

    ratings, _ = load_movielens(args.data_dir)
    splits = split_positive_histories(
        ratings, args.positive_threshold, args.min_positive
    )
    vocabulary = MovieVocabulary.from_training_splits(splits)
    training = PrefixDataset(splits, vocabulary, args.max_len)
    _, validation_counts = build_evaluation_examples(
        ratings, splits, vocabulary, "validation", args.max_len
    )
    _, test_counts = build_evaluation_examples(
        ratings, splits, vocabulary, "test", args.max_len
    )
    summary = {
        "catalogue_movies": len(vocabulary),
        "embedding_rows_including_padding": vocabulary.embedding_size,
        "training_prefix_examples": len(training),
        "max_history_length": args.max_len,
        "validation": validation_counts,
        "test": test_counts,
        "holdout_timestamp_ties": timestamp_tie_audit(splits),
    }

    protocol = {
        "version": "positive-history-v1",
        "positive_threshold": args.positive_threshold,
        "min_positive": args.min_positive,
        "max_history_length": args.max_len,
        "ordering": ["timestamp", "movieId"],
        "catalogue": "positive items in eligible users' training portions",
        "padding_id": 0,
        "padding_side": "right",
        "unknown_history_items": "drop before truncation",
        "unknown_targets": "exclude and report; never move the split",
        "seen_items": "all earlier raw ratings under the same ordering",
        "test_history": "training positives plus observed validation positive",
    }
    source_hashes = {}
    for name in ("ratings.dat", "movies.dat"):
        with (args.data_dir / name).open("rb") as handle:
            source_hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    fingerprint_inputs = {
        "protocol": protocol,
        "source_sha256": source_hashes,
        "movie_ids_in_index_order": vocabulary.movie_ids,
        "code_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("data.py", "dataset.py")
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_inputs, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {**fingerprint_inputs, "fingerprint": fingerprint, "summary": summary}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "preparation.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nFirst training example:", {
        key: value.tolist() for key, value in training[0].items()
    })
    print("\nPreparation fingerprint:", fingerprint)
    print("Saved:", destination)

    if args.check_batch:
        from torch.utils.data import DataLoader

        # num_workers=0 is the simplest initial setting, including on macOS.
        batch = next(iter(DataLoader(training, batch_size=4, shuffle=False, num_workers=0)))
        print("\nPyTorch batch:")
        for name, tensor in batch.items():
            print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


if __name__ == "__main__":
    main()
