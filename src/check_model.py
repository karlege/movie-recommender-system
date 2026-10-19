"""Check the transformer on one fixed TRAINING batch, without evaluating holdouts.

Run from the repo root: python -m src.check_model --steps 20
This is a learning smoke check, not a full training experiment. No checkpoint
or recommendation metrics are produced. All computations use CPU initially.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.baselines import restore_preparation
from src.dataset import PrefixDataset
from src.models.transformer import TransformerRecommender
from src.scoring import make_score_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ml-1m"))
    parser.add_argument(
        "--preparation", type=Path, default=Path("data/processed/preparation.json")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.steps) < 1:
        parser.error("batch-size and steps must be positive")

    torch.manual_seed(args.seed)
    # Limit thread overhead for this short CPU check; not a training benchmark.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    manifest, _, splits, vocabulary = restore_preparation(
        args.data_dir, args.preparation
    )
    max_len = manifest["protocol"]["max_history_length"]
    dataset = PrefixDataset(splits, vocabulary, max_len)
    generator = torch.Generator().manual_seed(args.seed)
    batch = next(iter(DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, generator=generator,
    )))
    histories, lengths = batch["history_ids"], batch["lengths"]
    # Item IDs are 1..M, but cross-entropy expects score COLUMN indices 0..M-1.
    target_columns = batch["target_ids"] - 1
    config = {
        "num_items": len(vocabulary), "max_len": max_len,
        "d_model": 64, "n_heads": 2, "n_layers": 2,
        "ff_multiplier": 4, "dropout": 0.1,
    }
    model = TransformerRecommender(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    def deterministic_loss():
        model.eval()
        with torch.no_grad():
            return nn.functional.cross_entropy(
                model(histories, lengths), target_columns
            ).item()

    initial_loss = deterministic_loss()
    for _ in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = model(histories, lengths)
        loss = nn.functional.cross_entropy(scores, target_columns)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss.")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise RuntimeError(f"Missing or non-finite gradient: {name}")
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    final_loss = deterministic_loss()
    if not final_loss < initial_loss:
        raise RuntimeError(
            f"Fixed-batch loss did not decrease: {initial_loss} -> {final_loss}."
        )
    score_array = make_score_batch(model)(histories.numpy(), lengths.numpy())
    if score_array.shape != (len(histories), len(vocabulary)):
        raise RuntimeError("Scoring output does not match evaluator interface.")
    report = {
        "check": "fixed_training_batch_only",
        "preparation_fingerprint": manifest["fingerprint"],
        "torch_version": torch.__version__,
        "device": "cpu", "seed": args.seed, "model_config": config,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "history_shape": list(histories.shape),
        "score_shape": list(score_array.shape),
        "optimizer_steps": args.steps,
        "initial_loss_without_dropout": initial_loss,
        "final_loss_without_dropout": final_loss,
        "finite_gradients": True,
    }
    print(json.dumps(report, indent=2))
    print("\nModel, gradients, fixed-batch learning, and scoring adapter checks passed.")


if __name__ == "__main__":
    main()
