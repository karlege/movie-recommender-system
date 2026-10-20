"""Order-independent pooled MLP adapted from the group's notebook PooledMLP.

The masked mean -> Linear -> ReLU -> Dropout -> Linear encoder follows
movie-recommender-systems(1).ipynb. For the shared comparison, this version
uses one movie table for history and candidates, normal(std=0.02) movie
initialization, sqrt(d_model) history scaling, and default dropout 0.1,
matching the repository transformer's embedding convention.
"""

import math

import torch
from torch import nn


class PooledMLP(nn.Module):
    """Average the real history embeddings and map them to a user vector.

    Input IDs are 1..num_items with right padding 0. Returned score column
    j belongs to item ID j+1. Padding is excluded from pooling and scoring.
    There are no position embeddings: evaluation is order-independent up
    to floating-point summation differences.
    """

    def __init__(self, num_items, max_len=50, d_model=64, hidden=128, dropout=0.1):
        super().__init__()
        if min(num_items, max_len, d_model, hidden) < 1:
            raise ValueError("Model sizes must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.num_items = num_items
        self.max_len = max_len
        self.d_model = d_model
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def pool_history(self, history_ids, lengths):
        """Compute the masked mean, dividing by real length, not padded width."""
        if history_ids.dtype != torch.long or lengths.dtype != torch.long:
            raise ValueError("Histories and lengths must be torch.int64 tensors.")
        if history_ids.ndim != 2 or lengths.shape != (history_ids.shape[0],):
            raise ValueError("Expected histories [batch, width] and lengths [batch].")
        batch_size, width = history_ids.shape
        if batch_size < 1 or not 1 <= width <= self.max_len:
            raise ValueError("Batch must be nonempty and width within 1..max_len.")
        if history_ids.device != lengths.device:
            raise ValueError("Histories and lengths must use the same device.")
        if ((lengths < 1) | (lengths > width)).any().item():
            raise ValueError("Lengths must be within 1..history width.")
        if ((history_ids < 0) | (history_ids > self.num_items)).any().item():
            raise ValueError("History IDs must be within 0..num_items.")
        positions = torch.arange(width, device=history_ids.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        if not torch.equal(history_ids.ne(0), valid):
            raise ValueError("Histories must contain real items then right padding.")
        embedded = self.item_embedding(history_ids) * math.sqrt(self.d_model)
        embedded = embedded.masked_fill(~valid.unsqueeze(-1), 0)
        return embedded.sum(dim=1) / lengths.to(embedded.dtype).unsqueeze(1)

    def encode_history(self, history_ids, lengths):
        return self.mlp(self.pool_history(history_ids, lengths))

    def forward(self, history_ids, lengths):
        user_vectors = self.encode_history(history_ids, lengths)
        return user_vectors @ self.item_embedding.weight[1:].T
