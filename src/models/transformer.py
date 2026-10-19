"""Compact SASRec-inspired history encoder, using standard PyTorch layers.

This is a project implementation, not an exact reproduction of SASRec.
Real item IDs are 1..num_items; 0 is input padding only. Score column j
belongs to item ID j+1, matching src.evaluate. No softmax or seen-item
filtering is applied here.
"""

import math

import torch
from torch import nn


class TransformerRecommender(nn.Module):
    """Encode positive history and dot-product it with candidate embeddings.

    History and candidate towers share the learned movie embedding table.
    The final non-padding position summarizes the history. Learned position
    embeddings indicate order within the retained, right-padded window.
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if min(num_items, max_len, d_model, n_heads, n_layers, ff_multiplier) < 1:
            raise ValueError("Model sizes must be positive.")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.num_items = num_items
        self.max_len = max_len
        self.d_model = d_model

        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.input_dropout = nn.Dropout(dropout)
        # Construct each block separately so their initial parameters differ.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_multiplier * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def _validate_inputs(self, history_ids, lengths):
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
        return positions, valid

    def encode_sequence(self, history_ids, lengths):
        """Return [batch, width, d_model], zero at padding positions.

        A boolean True blocks an attention connection. The causal mask
        blocks future keys; the padding mask blocks padding keys. Since
        histories are nonempty and right-padded, every query has at least
        one earlier real key, avoiding fully masked attention rows.
        """
        positions, valid = self._validate_inputs(history_ids, lengths)
        width = history_ids.shape[1]
        padding_mask = ~valid
        causal_mask = torch.triu(
            torch.ones(width, width, dtype=torch.bool, device=history_ids.device),
            diagonal=1,
        )
        hidden = (
            self.item_embedding(history_ids) * math.sqrt(self.d_model)
            + self.position_embedding(positions).unsqueeze(0)
        )
        hidden = self.input_dropout(hidden).masked_fill(padding_mask.unsqueeze(-1), 0)
        for block in self.blocks:
            hidden = block(
                hidden, src_mask=causal_mask, src_key_padding_mask=padding_mask
            )
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0)
        return self.final_norm(hidden).masked_fill(padding_mask.unsqueeze(-1), 0)

    def encode_history(self, history_ids, lengths):
        """Gather the final REAL position, rather than the final padded slot."""
        hidden = self.encode_sequence(history_ids, lengths)
        rows = torch.arange(history_ids.shape[0], device=history_ids.device)
        return hidden[rows, lengths - 1]

    def forward(self, history_ids, lengths):
        """Return raw [batch, num_items] scores; padding is never a candidate."""
        user_vectors = self.encode_history(history_ids, lengths)
        return user_vectors @ self.item_embedding.weight[1:].T
