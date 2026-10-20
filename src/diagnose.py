"""Validate a saved checkpoint and measure sensitivity to history order.

Run: python -m src.diagnose --run-dir runs/transformer_run1 \
    --output-dir runs/transformer_order_diagnostics

All conditions use the same validation users, observed movies, targets,
padding, and candidate exclusions. Only retained history order changes.
This is an inference-time sensitivity check, not a retrained ablation.
"""

import argparse
import hashlib
import json
import math
import platform
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.baselines import popularity_scores, restore_preparation, time_relation_labels
from src.dataset import build_evaluation_examples
from src.evaluate import evaluate_examples
from src.scoring import make_score_batch
from src.train import load_transformer_checkpoint, resolve_device


def retained_timestamps(examples, splits, vocabulary):
    """Align timestamps with each example's already retained training history.

    Filtering/truncation precedes perturbation: shuffling never brings an
    omitted movie, validation target, or test target into a history.
    """
    result = {}
    for example in examples:
        known = [event for event in splits[example.user_id].train
                 if event.movie_id in vocabulary.movie_to_index]
        retained = known[-example.length:]
        encoded = [vocabulary.encode(event.movie_id) for event in retained]
        if not np.array_equal(encoded, example.history_ids[:example.length]):
            raise ValueError("Timestamp/history alignment mismatch.")
        timestamps = np.asarray([event.timestamp for event in retained], dtype=np.int64)
        if (np.diff(timestamps) < 0).any():
            raise ValueError("Original timestamps must be nondecreasing.")
        result[example.user_id] = timestamps
    return result


def perturb_examples(examples, timestamps, mode, seed):
    """Shuffle real tokens only, with a separate seeded RNG for each user."""
    if mode not in ("shuffle_ties", "shuffle_all"):
        raise ValueError("mode must be shuffle_ties or shuffle_all.")
    if seed < 0:
        raise ValueError("Shuffle seed must be nonnegative.")
    output = []
    possible, changed, changed_positions = 0, 0, 0
    for example in examples:
        length = example.length
        history = example.history_ids.copy()
        rng = np.random.default_rng(np.random.SeedSequence([seed, example.user_id]))
        if mode == "shuffle_all":
            groups = [np.arange(length)]
        else:
            times = np.asarray(timestamps[example.user_id])
            if times.shape != (length,) or (np.diff(times) < 0).any():
                raise ValueError("Timestamps must match the real history and be sorted.")
            boundaries = np.flatnonzero(times[1:] != times[:-1]) + 1
            groups = np.split(np.arange(length), boundaries)
        possible += int(any(len(group) > 1 for group in groups))
        for group in groups:
            if len(group) > 1:
                history[group] = history[group][rng.permutation(len(group))]
        differences = int(np.count_nonzero(history != example.history_ids))
        changed += int(differences > 0)
        changed_positions += differences
        output.append(replace(example, history_ids=history))
    return output, {
        "shuffleable_histories": possible,
        "changed_histories": changed,
        "changed_positions": changed_positions,
    }


def mean_metrics(reports):
    """Mean across shuffle seeds, not a confidence interval across users."""
    if not reports or any(row["users"] != reports[0]["users"] for row in reports):
        raise ValueError("Every shuffle repeat must evaluate the same number of users.")
    return {
        "users": reports[0]["users"],
        **{key: float(np.mean([row[key] for row in reports]))
           for key in ("recall@10", "ndcg@10")},
    }


def check_reference_metrics(actual, expected):
    if actual["users"] != expected["users"] or any(
        not math.isclose(actual[key], expected[key], rel_tol=1e-9, abs_tol=1e-12)
        for key in ("recall@10", "ndcg@10")
    ):
        raise ValueError(
            "Original validation metrics differ from the saved checkpoint. "
            "Review the model/evaluator/environment before interpreting shuffles."
        )


def run_diagnostics(
    data_dir, preparation_path, run_dir, output_dir,
    seeds=(42, 43, 44), batch_size=128, device="cpu", num_threads=4,
):
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Provide distinct, nonnegative shuffle seeds.")
    if min(batch_size, num_threads) < 1:
        raise ValueError("Batch size and thread count must be positive.")
    device = resolve_device(device)
    output_dir, run_dir = Path(output_dir), Path(run_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Output directory is not empty; choose a new diagnostic directory.")
    started = perf_counter()
    torch.set_num_threads(num_threads)
    manifest, ratings, splits, vocabulary = restore_preparation(data_dir, preparation_path)
    model, checkpoint = load_transformer_checkpoint(
        run_dir / "best.pt", manifest["fingerprint"], device
    )
    source_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if (source_config["preparation_fingerprint"] != manifest["fingerprint"]
            or source_config["model_config"] != checkpoint["model_config"]):
        raise ValueError("Run configuration does not match checkpoint/preparation.")
    examples, coverage = build_evaluation_examples(
        ratings, splits, vocabulary, "validation",
        manifest["protocol"]["max_history_length"],
    )
    if not examples:
        raise ValueError("No eligible validation targets.")
    timestamps = retained_timestamps(examples, splits, vocabulary)
    labels = time_relation_labels(splits, "validation")
    scorer = make_score_batch(model)

    def evaluate(records, score_batch):
        before = perf_counter()
        result = evaluate_examples(
            records, score_batch, len(vocabulary), k=10,
            batch_size=batch_size, cohorts=labels,
        )
        return {
            "metrics": result["metrics"],
            "by_time_relation": result["by_time_relation"],
            "evaluation_seconds": perf_counter() - before,
        }

    original = evaluate(examples, scorer)
    check_reference_metrics(original["metrics"], checkpoint["validation_metrics"])
    counts = popularity_scores(splits, vocabulary)
    popularity = evaluate(
        examples, lambda histories, lengths: np.broadcast_to(counts, (len(histories), len(vocabulary)))
    )
    conditions = {}
    for mode in ("shuffle_ties", "shuffle_all"):
        repeats = []
        for seed in seeds:
            perturbed, changes = perturb_examples(examples, timestamps, mode, seed)
            repeats.append({"seed": seed, **changes, **evaluate(perturbed, scorer)})
        average = mean_metrics([repeat["metrics"] for repeat in repeats])
        conditions[mode] = {
            "mean_metrics": average,
            "mean_by_time_relation": {
                label: mean_metrics([repeat["by_time_relation"][label] for repeat in repeats])
                for label in original["by_time_relation"]
            },
            "delta_from_original": {
                key: average[key] - original["metrics"][key]
                for key in ("recall@10", "ndcg@10")
            },
            "repeats": repeats,
        }
    report = {
        "split": "validation", "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest(),
        "preparation_fingerprint": manifest["fingerprint"],
        "source_training_scope": source_config["training_scope"],
        "coverage": coverage, "original_matches_checkpoint": True,
        "shuffle_seeds": list(seeds), "batch_size": batch_size,
        "original": original, "popularity": popularity, **conditions,
        "interpretation": {
            "scope": "inference-time order sensitivity of this fitted checkpoint",
            "shuffle_ties": "permute retained observed movies only within identical timestamps",
            "shuffle_all": "permute all retained observed movies, preserving their set",
            "limits": [
                "Inference-time shuffling changes the input distribution; a drop alone does not establish useful chronology.",
                "Tie shuffling does not change the held-out target or remove movie-ID effects in target selection and training.",
                "Compare against a trained order-independent MLP before attributing gains to sequence modeling.",
                "Shuffle-seed averages describe permutation variability, not statistical confidence intervals.",
            ],
        },
        "environment": {
            "torch": str(torch.__version__), "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "device": str(device), "torch_threads": torch.get_num_threads(),
        },
        "source_run_code_sha256": source_config["code_sha256"],
        "diagnostic_code_sha256": {
            name: hashlib.sha256(Path(__file__).parent.joinpath(name).read_bytes()).hexdigest()
            for name in ("diagnose.py", "models/transformer.py", "scoring.py", "evaluate.py", "train.py")
        },
        "total_seconds": perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ml-1m"))
    parser.add_argument("--preparation", type=Path, default=Path("data/processed/preparation.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shuffle-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    report = run_diagnostics(
        args.data_dir, args.preparation, args.run_dir, args.output_dir,
        args.shuffle_seeds, args.batch_size, args.device, args.num_threads,
    )
    print(json.dumps({
        "split": report["split"], "checkpoint_epoch": report["checkpoint_epoch"],
        "original_matches_checkpoint": report["original_matches_checkpoint"],
        "validation_users": report["coverage"]["eligible_targets"],
        "original": {k: report["original"][k] for k in ("metrics", "by_time_relation")},
        "popularity": {k: report["popularity"][k] for k in ("metrics", "by_time_relation")},
        **{mode: {k: report[mode][k] for k in (
            "mean_metrics", "mean_by_time_relation", "delta_from_original"
        )} for mode in ("shuffle_ties", "shuffle_all")},
        "total_seconds": report["total_seconds"],
    }, indent=2))
    print("\nSaved:", args.output_dir / "diagnostics.json")


if __name__ == "__main__":
    main()
