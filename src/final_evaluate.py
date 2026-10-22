"""Test frozen validation-selected MLP/transformer checkpoints and popularity.

Run from the project root: python -m src.final_evaluate --runs-root runs
No fitting or checkpoint selection is performed. Original run files are read-only.
"""

import argparse
import csv
import hashlib
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.baselines import popularity_scores, restore_preparation, time_relation_labels
from src.dataset import build_evaluation_examples
from src.evaluate import evaluate_examples
from src.scoring import make_score_batch
from src.train import load_checkpoint, resolve_device, synchronize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_run(run_dir, fingerprint, model_type, seed, num_items, max_len):
    """Check all inputs before reading any test targets or writing outputs."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    saved = json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    model, checkpoint = load_checkpoint(run_dir / "best.pt", fingerprint, "cpu")
    if (config["preparation_fingerprint"] != fingerprint
            or saved["preparation_fingerprint"] != fingerprint
            or config.get("model_type", "transformer") != model_type
            or checkpoint["model_type"] != model_type
            or config["training_config"]["seed"] != seed
            or config["model_config"] != checkpoint["model_config"]
            or config["model_config"]["num_items"] != num_items
            or config["model_config"]["max_len"] != max_len):
        raise ValueError(f"Run configuration/checkpoint mismatch: {run_dir}")
    scope = config["training_scope"]
    if (scope["scope"] != "full_training"
            or scope["examples_per_epoch"] != scope["available_examples"]):
        raise ValueError(f"Final comparison requires full-training runs: {run_dir}")
    if config["training_config"].get("training_tie_seed") is not None:
        raise ValueError(f"Use original-order runs for this comparison: {run_dir}")
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines() if line.strip()]
    best = max(history, key=lambda row: row["metrics"]["ndcg@10"])
    if (checkpoint["epoch"] != saved["best_epoch"]
            or checkpoint["epoch"] != best["epoch"]
            or checkpoint["validation_metrics"] != saved["metrics"]
            or checkpoint["validation_metrics"] != best["metrics"]):
        raise ValueError(f"Checkpoint is not the recorded validation-selected best: {run_dir}")
    for name in (f"models/{model_type}.py", "scoring.py", "evaluate.py"):
        if digest(Path(__file__).parent / name) != config["code_sha256"][name]:
            raise ValueError(f"Training/evaluation code changed: {name}; review before testing.")
    return {
        "model": model, "model_type": model_type, "seed": seed,
        "run_dir": run_dir, "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": digest(run_dir / "best.pt"),
        "validation_selection_metrics": checkpoint["validation_metrics"],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def summarize(reports):
    """Average independently trained seeds, with sample SD, also by time cohort."""
    groups = {}
    for report in reports:
        for cohort, metrics in {"overall": report["metrics"], **report["by_time_relation"]}.items():
            groups.setdefault((report["model_type"], cohort), []).append(metrics)
    rows = []
    for (model, cohort), values in groups.items():
        if len({value["users"] for value in values}) != 1:
            raise ValueError("Runs must use identical user coverage.")
        row = {"model": model, "cohort": cohort, "users": values[0]["users"], "runs": len(values)}
        for metric in ("recall@10", "ndcg@10"):
            scores = [value[metric] for value in values]
            row[metric + "_mean"] = statistics.mean(scores) if row["users"] else None
            row[metric + "_sample_sd"] = (
                statistics.stdev(scores) if len(scores) > 1 and row["users"] else None
            )
        rows.append(row)
    return rows


def comparison_markdown(summary):
    text = ["# Frozen-checkpoint test results", "",
            "Neural models: mean +/- sample SD across training seeds; each checkpoint was selected using validation NDCG@10. "
            "Popularity is deterministic. SD is not a confidence interval.", ""]
    for cohort in dict.fromkeys(row["cohort"] for row in summary):
        text += [f"## {cohort.replace('_', ' ').title()}", "",
                 "| Model | Users | Runs | Recall@10 | NDCG@10 |",
                 "|---|---:|---:|---:|---:|"]
        for row in summary:
            if row["cohort"] != cohort:
                continue
            cells = []
            for metric in ("recall@10", "ndcg@10"):
                mean, sd = row[metric + "_mean"], row[metric + "_sample_sd"]
                cells.append("-" if mean is None else f"{mean:.5f}" + (f" +/- {sd:.5f}" if sd is not None else ""))
            text.append(f"| {row['model']} | {row['users']} | {row['runs']} | {' | '.join(cells)} |")
        text.append("")
    text += ["Test histories include the validation interaction; vocabulary and popularity counts remain training-only. "
             "Timestamp ties and architectural differences limit claims about the causal benefit of chronology.", ""]
    return "\n".join(text)


def run_final_evaluation(data_dir, preparation_path, runs_root, output_dir,
                         seeds=(42, 43, 44), device="cpu", batch_size=128,
                         num_threads=4, show_progress=True):
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Provide at least two distinct nonnegative training seeds.")
    if min(batch_size, num_threads) < 1:
        raise ValueError("Batch size and thread count must be positive.")
    output_dir, runs_root = Path(output_dir), Path(runs_root)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Output directory is not empty; choose a new final-evaluation directory.")
    device = resolve_device(device)
    torch.set_num_threads(num_threads)
    manifest, ratings, splits, vocabulary = restore_preparation(data_dir, preparation_path)
    max_len = manifest["protocol"]["max_history_length"]
    sources = [
        check_run(runs_root / f"{kind}_20ep_seed{seed}_l4", manifest["fingerprint"],
                  kind, seed, len(vocabulary), max_len)
        for kind in ("mlp", "transformer") for seed in seeds
    ]
    # All requested checkpoints must pass preflight before test data are built.
    examples, coverage = build_evaluation_examples(ratings, splits, vocabulary, "test", max_len)
    if not examples:
        raise ValueError("No eligible test targets.")
    labels = time_relation_labels(splits, "test")
    counts = popularity_scores(splits, vocabulary)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "device": str(device), "num_threads": num_threads,
    }
    code_hashes = {name: digest(Path(__file__).parent / name) for name in (
        "final_evaluate.py", "dataset.py", "evaluate.py", "scoring.py", "baselines.py",
        "models/mlp.py", "models/transformer.py",
    )}
    reports = []
    # Popularity is fit once on original training positives, never validation/test.
    tasks = [{"model_type": "popularity", "seed": None, "trainable_parameters": 0}, *sources]
    for source in tasks:
        kind = source["model_type"]
        name = source["run_dir"].name if kind != "popularity" else "popularity"
        if show_progress:
            print(f"Testing {name}: {len(examples):,} users; device={device}", flush=True)
        if kind == "popularity":
            scorer = lambda histories, lengths: np.broadcast_to(counts, (len(histories), len(vocabulary)))
        else:
            source["model"].to(device)
            scorer = make_score_batch(source["model"])
        synchronize(device)
        before = perf_counter()
        result = evaluate_examples(examples, scorer, len(vocabulary), k=10,
                                   batch_size=batch_size, cohorts=labels)
        synchronize(device)
        seconds = perf_counter() - before
        report = {
            "model_type": kind, "seed": source["seed"], "split": "test",
            "preparation_fingerprint": manifest["fingerprint"], "coverage": coverage,
            "catalogue_movies": len(vocabulary), "metrics": result["metrics"],
            "by_time_relation": result["by_time_relation"],
            "trainable_parameters": source["trainable_parameters"],
            "batch_size": batch_size, "evaluation_seconds": seconds,
            "environment": environment, "evaluation_code_sha256": code_hashes,
        }
        if kind != "popularity":
            report.update({key: source[key] for key in (
                "checkpoint_epoch", "checkpoint_sha256", "validation_selection_metrics",
            )})
            report["source_run"] = str(source["run_dir"])
            source["model"].to("cpu")
        destination = output_dir / name
        destination.mkdir()
        write_json(destination / "test_metrics.json", report)
        with (destination / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for prediction in result["predictions"]:
                handle.write(json.dumps({
                    **prediction, "target_movie_id": vocabulary.decode(prediction["target_id"]),
                    "top_movie_ids": [vocabulary.decode(i) for i in prediction["top_ids"]],
                }, allow_nan=False) + "\n")
        reports.append(report)
    summary = summarize(reports)
    comparison = {
        "split": "test", "training_seeds": list(seeds), "coverage": coverage,
        "preparation_fingerprint": manifest["fingerprint"],
        "uncertainty": "sample SD across training seeds; not a confidence interval",
        "checkpoint_selection": "frozen best validation NDCG@10, no test-based selection or fitting",
        "summary": summary, "per_run": reports,
    }
    write_json(output_dir / "comparison.json", comparison)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output_dir / "comparison.md").write_text(comparison_markdown(summary), encoding="utf-8")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ml-1m"))
    parser.add_argument("--preparation", dest="preparation_path", type=Path,
                        default=Path("data/processed/preparation.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/final_test"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    result = run_final_evaluation(**vars(args))
    print("\n" + comparison_markdown(result["summary"]))
    print(f"Saved comparison and per-run metrics/predictions: {args.output_dir}")


if __name__ == "__main__":
    main()
