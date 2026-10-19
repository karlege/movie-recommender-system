"""Adapt a PyTorch recommender to the existing NumPy ranking evaluator."""

import numpy as np
import torch


def make_score_batch(model):
    """Return callback(histories, lengths) -> finite [batch, catalogue] scores.

    The model must already be on its intended device. Evaluation uses
    model.eval() and disables autograd; the previous training mode is restored
    even when scoring raises. Targets and seen-item sets are never inputs.
    """
    def score_batch(history_ids, lengths):
        device = next(model.parameters()).device
        histories = torch.as_tensor(history_ids, dtype=torch.long, device=device)
        sizes = torch.as_tensor(lengths, dtype=torch.long, device=device)
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                scores = model(histories, sizes)
                if scores.ndim != 2 or not torch.isfinite(scores).all().item():
                    raise ValueError("Model must return finite [batch, catalogue] scores.")
                return scores.detach().cpu().numpy().astype(np.float64, copy=False)
        finally:
            model.train(was_training)

    return score_batch
