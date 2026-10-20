"""Train a pooled MLP or transformer; select by validation NDCG@10.

Run: python -m src.train --model mlp --output-dir runs/mlp_run1
The default model remains transformer for existing commands.
For a short trial, add: --max-train-examples 8192 --epochs 2
The test split is never evaluated by this command.
"""

import argparse
import hashlib
import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.baselines import restore_preparation, time_relation_labels
from src.dataset import PrefixDataset, build_evaluation_examples
from src.evaluate import evaluate_examples
from src.models.mlp import PooledMLP
from src.models.transformer import TransformerRecommender
from src.scoring import make_score_batch


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 256
    eval_batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    patience: int = 3
    max_train_examples: int | None = None
    seed: int = 42
    device: str = "cpu"
    num_threads: int = 4
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    ff_multiplier: int = 4
    dropout: float = 0.1
    model: str = "transformer"
    mlp_hidden: int = 128

    def validate(self):
        if self.model not in ("mlp", "transformer"):
            raise ValueError("model must be mlp or transformer.")
        positive_sizes = (
            self.epochs, self.batch_size, self.eval_batch_size, self.patience,
            self.num_threads, self.d_model,
        )
        if self.model == "transformer":
            positive_sizes += (self.n_heads, self.n_layers, self.ff_multiplier)
        else:
            positive_sizes += (self.mlp_hidden,)
        if min(positive_sizes) < 1:
            raise ValueError("Epochs, batch sizes, patience, threads and model sizes must be positive.")
        if self.max_train_examples is not None and self.max_train_examples < 1:
            raise ValueError("max_train_examples must be positive when supplied.")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative.")
        if not all(math.isfinite(x) for x in (
            self.learning_rate, self.weight_decay, self.grad_clip, self.dropout
        )):
            raise ValueError("Training hyperparameters must be finite.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Learning rate/gradient clip must be positive; weight decay nonnegative.")
        if not 0 <= self.dropout < 1:
            raise ValueError("Require dropout in [0, 1).")
        if self.model == "transformer" and self.d_model % self.n_heads:
            raise ValueError("Transformer d_model must be divisible by n_heads.")


def build_model(model_type, model_config):
    """Construct only supported architectures from their recorded settings."""
    classes = {"transformer": TransformerRecommender, "mlp": PooledMLP}
    if model_type not in classes:
        raise ValueError(f"Unknown model type: {model_type!r}.")
    return classes[model_type](**model_config)


def resolve_device(name):
    if name not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda or mps.")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose cpu or another available device.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; choose cpu or another available device.")
    return torch.device(name)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def select_training_examples(dataset, limit, seed):
    """Subset only training prefixes, leaving vocabulary and validation intact."""
    if len(dataset) == 0:
        raise ValueError("No training examples.")
    if limit is not None and limit < 1:
        raise ValueError("Training subset size must be positive.")
    used = min(limit, len(dataset)) if limit is not None else len(dataset)
    indices = None
    if used < len(dataset):
        indices = np.sort(np.random.default_rng(seed).choice(len(dataset), used, replace=False))
        selected = Subset(dataset, indices.tolist())
    else:
        selected = dataset
    return selected, {
        "scope": "full_training" if used == len(dataset) else "trial_subset",
        "available_examples": len(dataset),
        "examples_per_epoch": used,
        "subset_indices_sha256": (
            hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
            if indices is not None else None
        ),
    }


def train_epoch(model, loader, optimizer, device, grad_clip=1.0, show_progress=False):
    """One pass over training prefixes, including the final incomplete batch."""
    device = torch.device(device)
    model.train()
    loss_sum, count, steps = 0.0, 0, 0
    synchronize(device)
    started = perf_counter()
    progress = tqdm(loader, desc="Training", unit="batch", disable=not show_progress)
    for batch in progress:
        histories = batch["history_ids"].to(device)
        lengths = batch["lengths"].to(device)
        # Scores have columns 0..M-1, while item IDs are 1..M.
        target_columns = batch["target_ids"].to(device) - 1
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(histories, lengths), target_columns)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss.")
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        size = len(histories)
        loss_sum += loss.item() * size
        count += size
        steps += 1
        if show_progress:
            progress.set_postfix(loss=f"{loss_sum / count:.4f}", refresh=False)
    synchronize(device)
    if not count:
        raise ValueError("Training loader is empty.")
    return {
        "train_loss": loss_sum / count,
        "examples_seen": count,
        "optimizer_steps": steps,
        "training_seconds": perf_counter() - started,
    }


def evaluate_validation(model, examples, num_items, labels, batch_size):
    """Same full-catalogue ranking path and exclusions as the popularity model."""
    device = next(model.parameters()).device
    synchronize(device)
    started = perf_counter()
    evaluation = evaluate_examples(
        examples, make_score_batch(model), num_items, k=10,
        batch_size=batch_size, cohorts=labels,
    )
    synchronize(device)
    return evaluation, perf_counter() - started


def save_checkpoint(path, model, model_config, fingerprint, epoch, metrics, model_type=None):
    """Save tensors and primitive metadata; replace best.pt atomically."""
    if isinstance(model, PooledMLP):
        actual_type = "mlp"
    elif isinstance(model, TransformerRecommender):
        actual_type = "transformer"
    else:
        raise ValueError("Checkpoint model must be a supported recommender.")
    if model_type is not None and model_type != actual_type:
        raise ValueError("Checkpoint model type does not match the supplied model.")
    payload = {
        "format_version": 2,
        "model_type": actual_type,
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "model_config": model_config,
        "preparation_fingerprint": fingerprint,
        "epoch": epoch,
        "validation_metrics": metrics,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, expected_fingerprint, device="cpu"):
    """Load either architecture, including original format-1 transformers."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] == 1:
        model_type = "transformer"  # Original files predate model selection.
    elif checkpoint["format_version"] == 2:
        model_type = checkpoint.get("model_type")
    else:
        raise ValueError("Unsupported checkpoint format.")
    if checkpoint["preparation_fingerprint"] != expected_fingerprint:
        raise ValueError("Checkpoint preparation fingerprint mismatch.")
    model = build_model(model_type, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device(device)).eval()
    return model, {**checkpoint, "model_type": model_type}


def load_transformer_checkpoint(path, expected_fingerprint, device="cpu"):
    """Backward-compatible entry point for transformer-only callers."""
    model, checkpoint = load_checkpoint(path, expected_fingerprint, device)
    if checkpoint["model_type"] != "transformer":
        raise ValueError("Expected a transformer checkpoint; use load_checkpoint for MLPs.")
    return model, checkpoint


def run_training(data_dir, preparation_path, output_dir, config=None, show_progress=True):
    config = config or TrainConfig()
    config.validate()
    device = resolve_device(config.device)
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Output directory is not empty; choose a new run directory.")
    started = perf_counter()
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.num_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    manifest, ratings, splits, vocabulary = restore_preparation(data_dir, preparation_path)
    max_len = manifest["protocol"]["max_history_length"]
    full_training = PrefixDataset(splits, vocabulary, max_len)
    training, training_scope = select_training_examples(
        full_training, config.max_train_examples, config.seed
    )
    examples, coverage = build_evaluation_examples(
        ratings, splits, vocabulary, "validation", max_len
    )
    if not examples:
        raise ValueError("No eligible validation targets.")
    labels = time_relation_labels(splits, "validation")
    loader = DataLoader(
        training, batch_size=config.batch_size, shuffle=True, num_workers=0,
        drop_last=False, generator=torch.Generator().manual_seed(config.seed),
    )
    model_config = {
        "num_items": len(vocabulary), "max_len": max_len,
        "d_model": config.d_model, "dropout": config.dropout,
    }
    if config.model == "transformer":
        model_config.update(
            n_heads=config.n_heads, n_layers=config.n_layers,
            ff_multiplier=config.ff_multiplier,
        )
    else:
        model_config["hidden"] = config.mlp_hidden
    model = build_model(config.model, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    metadata = {
        "model": "causal_transformer" if config.model == "transformer" else "pooled_mlp",
        "model_type": config.model, "split": "validation",
        "preparation_fingerprint": manifest["fingerprint"],
        "catalogue_movies": len(vocabulary), "coverage": coverage,
        "training_scope": training_scope, "training_config": asdict(config),
        "model_config": model_config,
        "embedding_convention": "shared table; normal(std=0.02); history scaled by sqrt(d_model)",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "selection_metric": "validation ndcg@10; strictly greater, ties keep earlier epoch",
        "loss": "full-catalogue cross-entropy, targets shifted from item IDs to score columns",
        "environment": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "device": str(device),
            "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_threads": torch.get_num_threads(),
        },
        "code_sha256": {
            name: hashlib.sha256(Path(__file__).parent.joinpath(name).read_bytes()).hexdigest()
            for name in ("train.py", "models/mlp.py", "models/transformer.py", "scoring.py", "evaluate.py", "baselines.py")
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if show_progress:
        print(
            f"Training: {training_scope['examples_per_epoch']:,}/{len(full_training):,} prefixes; "
            f"validation: {len(examples):,} users; device: {device}.", flush=True
        )
    preparation_seconds = perf_counter() - started
    history, best_score, best_epoch, bad_epochs = [], -math.inf, 0, 0
    loop_started = perf_counter()
    with (output_dir / "history.jsonl").open("w", encoding="utf-8") as handle:
        for epoch in range(1, config.epochs + 1):
            epoch_stats = train_epoch(
                model, loader, optimizer, device, config.grad_clip, show_progress
            )
            evaluation, seconds = evaluate_validation(
                model, examples, len(vocabulary), labels, config.eval_batch_size
            )
            metrics = evaluation["metrics"]
            score = metrics["ndcg@10"]
            if score is None or not math.isfinite(score):
                raise RuntimeError("Validation NDCG@10 is missing or non-finite.")
            improved = score > best_score
            if improved:
                best_score, best_epoch, bad_epochs = score, epoch, 0
                save_checkpoint(
                    output_dir / "best.pt", model, model_config,
                    manifest["fingerprint"], epoch, metrics, model_type=config.model,
                )
            else:
                bad_epochs += 1
            record = {
                "epoch": epoch, **epoch_stats, "validation_seconds": seconds,
                "metrics": metrics, "by_time_relation": evaluation["by_time_relation"],
                "saved_best": improved,
            }
            history.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if show_progress:
                print(json.dumps(record), flush=True)
            if bad_epochs >= config.patience:
                break
    selection_loop_seconds = perf_counter() - loop_started

    # Reload from disk so the final predictions correspond to the selected
    # epoch, even when the last epoch had worse validation performance.
    best_model, checkpoint = load_checkpoint(
        output_dir / "best.pt", manifest["fingerprint"], device
    )
    evaluation, final_seconds = evaluate_validation(
        best_model, examples, len(vocabulary), labels, config.eval_batch_size
    )
    report = {
        **metadata, "best_epoch": best_epoch, "epochs_completed": len(history),
        "stopped_early": len(history) < config.epochs,
        "metrics": evaluation["metrics"], "by_time_relation": evaluation["by_time_relation"],
        "checkpoint_selection_metrics": checkpoint["validation_metrics"],
        "training_seconds": sum(row["training_seconds"] for row in history),
        "validation_seconds_during_training": sum(row["validation_seconds"] for row in history),
        "data_and_model_preparation_seconds": preparation_seconds,
        "selection_loop_seconds": selection_loop_seconds,
        "best_checkpoint_evaluation_seconds": final_seconds,
        "total_run_seconds": perf_counter() - started,
        "optimizer_steps_total": sum(row["optimizer_steps"] for row in history),
        "training_examples_seen_total": sum(row["examples_seen"] for row in history),
    }
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "validation_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in evaluation["predictions"]:
            handle.write(json.dumps({
                **prediction, "target_movie_id": vocabulary.decode(prediction["target_id"]),
                "top_movie_ids": [vocabulary.decode(i) for i in prediction["top_ids"]],
            }) + "\n")
    return report


def main():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ml-1m"))
    parser.add_argument("--preparation", type=Path, default=Path("data/processed/preparation.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    for name in (
        "epochs", "batch_size", "eval_batch_size", "patience", "seed", "num_threads",
        "d_model", "n_heads", "n_layers", "ff_multiplier", "mlp_hidden",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "weight_decay", "grad_clip", "dropout"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--model", choices=("mlp", "transformer"), default="transformer")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    settings = {key: getattr(args, key) for key in asdict(defaults)}
    report = run_training(
        args.data_dir, args.preparation, args.output_dir,
        TrainConfig(**settings), show_progress=not args.quiet,
    )
    print(json.dumps({key: report[key] for key in (
        "model", "model_type", "split", "preparation_fingerprint", "training_scope", "coverage",
        "best_epoch", "epochs_completed", "stopped_early", "metrics", "by_time_relation",
        "training_seconds", "validation_seconds_during_training", "total_run_seconds",
    )}, indent=2))
    print("\nSaved run:", args.output_dir)


if __name__ == "__main__":
    main()
