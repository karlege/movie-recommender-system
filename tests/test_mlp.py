"""Pooled-history behavior and the shared scoring interface."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from src.dataset import EvaluationExample
from src.evaluate import evaluate_examples
from src.models.mlp import PooledMLP
from src.scoring import make_score_batch


@pytest.fixture
def model():
    torch.manual_seed(42)
    return PooledMLP(num_items=12, max_len=6, d_model=16, hidden=24, dropout=0)


def test_evaluation_is_invariant_to_reordering(model):
    model.eval()
    histories = torch.tensor([[1, 2, 3, 4, 0], [4, 2, 1, 3, 0]])
    scores = model(histories, torch.tensor([4, 4]))
    torch.testing.assert_close(scores[0], scores[1], atol=1e-6, rtol=1e-5)


def test_padding_width_and_mixed_batches_do_not_change_prediction(model):
    model.eval()
    with torch.no_grad():
        short = model(torch.tensor([[1, 2]]), torch.tensor([2]))
        padded = model(torch.tensor([[1, 2, 0, 0, 0, 0]]), torch.tensor([2]))
        mixed = model(torch.tensor([[1, 2, 0, 0, 0, 0], [3, 4, 5, 6, 7, 8]]),
                      torch.tensor([2, 6]))
    torch.testing.assert_close(short, padded)
    torch.testing.assert_close(short[0], mixed[0], atol=1e-6, rtol=1e-5)


def test_pooling_excludes_padding_and_uses_actual_length(model):
    with torch.no_grad():
        model.item_embedding.weight[1].fill_(1)
        model.item_embedding.weight[2].fill_(3)
        # Masking should ignore padding even if its row were nonzero.
        model.item_embedding.weight[0].fill_(99)
    pooled = model.pool_history(torch.tensor([[1, 2, 0, 0]]), torch.tensor([2]))
    torch.testing.assert_close(pooled, torch.full_like(pooled, 2 * math.sqrt(16)))


def test_history_and_candidates_share_one_embedding_table(model):
    histories, lengths = torch.tensor([[1, 2, 0]]), torch.tensor([2])
    vector = model.encode_history(histories, lengths)
    scores = model(histories, lengths)
    assert scores.shape == (1, 12)
    for item_id in (1, 6, 12):
        expected = (vector[0] * model.item_embedding.weight[item_id]).sum()
        torch.testing.assert_close(scores[0, item_id - 1], expected)
    embedding_modules = [m for m in model.modules() if isinstance(m, nn.Embedding)]
    assert len(embedding_modules) == 1


def test_full_catalogue_loss_and_padding_gradients(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scores = model(torch.tensor([[1, 2, 0], [3, 0, 0]]), torch.tensor([2, 1]))
    loss = nn.functional.cross_entropy(scores, torch.tensor([12, 6]) - 1)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all().item()
    assert torch.count_nonzero(model.item_embedding.weight.grad[0]).item() == 0
    before = model.item_embedding.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before[1:], model.item_embedding.weight[1:])
    assert torch.count_nonzero(model.item_embedding.weight[0]).item() == 0


def test_mlp_can_learn_a_tiny_batch():
    torch.manual_seed(42)
    model = PooledMLP(8, max_len=3, d_model=16, hidden=24, dropout=0)
    histories = torch.tensor([[1, 0, 0], [2, 3, 0], [3, 4, 5]])
    lengths, targets = torch.tensor([1, 2, 3]), torch.tensor([6, 7, 8]) - 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = nn.functional.cross_entropy(model(histories, lengths), targets).item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(histories, lengths), targets)
        loss.backward()
        optimizer.step()
    scores = model(histories, lengths)
    assert nn.functional.cross_entropy(scores, targets).item() < initial * 0.1
    assert torch.equal(scores.argmax(dim=1), targets)


def test_mlp_uses_existing_evaluator_and_exclusions(model):
    model.eval()
    scorer = make_score_batch(model)
    histories, lengths = np.array([[1, 2, 0]]), np.array([2])
    scores = scorer(histories, lengths)[0]
    seen = np.array([1, 2, 11])
    allowed = scores.copy()
    allowed[seen - 1] = -np.inf
    target = int(allowed.argmax()) + 1
    record = EvaluationExample(1, histories[0], 2, target, seen)
    result = evaluate_examples([record], scorer, 12, k=1)
    assert result["metrics"] == {"users": 1, "recall@1": 1.0, "ndcg@1": 1.0}
    assert result["predictions"][0]["top_ids"] == [target]


@pytest.mark.parametrize("history,lengths", [
    ([[0, 0]], [0]), ([[1, 0]], [2]), ([[1, 0, 2]], [2]),
    ([[0, 1]], [1]), ([[13, 0]], [1]),
])
def test_mlp_rejects_invalid_history(model, history, lengths):
    with pytest.raises(ValueError):
        model(torch.tensor(history), torch.tensor(lengths))
