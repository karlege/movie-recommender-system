"""Behavior checks for causal encoding, padding, learning, and shared scoring."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from src.dataset import EvaluationExample
from src.evaluate import evaluate_examples
from src.models.transformer import TransformerRecommender
from src.scoring import make_score_batch


@pytest.fixture
def model():
    torch.manual_seed(7)
    return TransformerRecommender(
        num_items=12, max_len=6, d_model=16,
        n_heads=2, n_layers=2, dropout=0.0,
    )


@pytest.mark.parametrize("training", [True, False])
def test_future_tokens_cannot_change_earlier_states(model, training):
    model.train(training)
    histories = torch.tensor([[1, 2, 3, 4], [1, 2, 8, 9]])
    with torch.no_grad():
        hidden = model.encode_sequence(histories, torch.tensor([4, 4]))
    torch.testing.assert_close(hidden[0, :2], hidden[1, :2], atol=1e-6, rtol=1e-5)
    # The test would otherwise pass for a model ignoring the changed inputs.
    assert not torch.allclose(hidden[0, 2:], hidden[1, 2:])


@pytest.mark.parametrize("training", [True, False])
def test_appending_right_padding_does_not_change_scores(model, training):
    model.train(training)
    with torch.no_grad():
        short = model(torch.tensor([[1, 2, 3]]), torch.tensor([3]))
        padded = model(torch.tensor([[1, 2, 3, 0, 0, 0]]), torch.tensor([3]))
    torch.testing.assert_close(short, padded, atol=1e-6, rtol=1e-5)


def test_mixed_length_batch_matches_individual_histories(model):
    model.eval()
    histories = torch.tensor([[2, 0, 0, 0], [1, 3, 5, 7]])
    with torch.no_grad():
        batched = model(histories, torch.tensor([1, 4]))
        first = model(torch.tensor([[2]]), torch.tensor([1]))
        second = model(histories[1:], torch.tensor([4]))
    torch.testing.assert_close(batched[:1], first, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(batched[1:], second, atol=1e-6, rtol=1e-5)


def test_history_vector_uses_last_real_token_and_has_no_padding_outputs(model):
    histories = torch.tensor([[1, 2, 0, 0], [2, 3, 4, 5]])
    lengths = torch.tensor([2, 4])
    sequence = model.encode_sequence(histories, lengths)
    vector = model.encode_history(histories, lengths)
    torch.testing.assert_close(vector, torch.stack([sequence[0, 1], sequence[1, 3]]))
    assert torch.count_nonzero(sequence[0, 2:]).item() == 0


def test_score_columns_are_real_items_and_exclude_padding(model):
    history, lengths = torch.tensor([[1, 2]]), torch.tensor([2])
    vector = model.encode_history(history, lengths)
    scores = model(history, lengths)
    assert scores.shape == (1, 12)
    for item_id in (1, 6, 12):
        expected = (vector[0] * model.item_embedding.weight[item_id]).sum()
        torch.testing.assert_close(scores[0, item_id - 1], expected)


def test_reordering_history_can_change_prediction(model):
    histories = torch.tensor([[1, 2, 3], [3, 2, 1]])
    scores = model(histories, torch.tensor([3, 3]))
    assert not torch.allclose(scores[0], scores[1])


def test_cross_entropy_backward_and_padding_row(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scores = model(torch.tensor([[1, 2, 0], [3, 4, 5]]), torch.tensor([2, 3]))
    # Include the largest item ID to catch target/column off-by-one errors.
    targets = torch.tensor([12, 6])
    loss = nn.functional.cross_entropy(scores, targets - 1)
    loss.backward()
    assert torch.isfinite(loss).item()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all().item()
    assert torch.count_nonzero(model.item_embedding.weight.grad[0]).item() == 0
    before = model.item_embedding.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before[1:], model.item_embedding.weight[1:])
    assert torch.count_nonzero(model.item_embedding.weight[0]).item() == 0


def test_can_overfit_a_tiny_fixed_batch():
    torch.manual_seed(42)
    tiny = TransformerRecommender(
        num_items=8, max_len=4, d_model=16,
        n_heads=2, n_layers=1, dropout=0,
    )
    histories = torch.tensor([[1, 0, 0, 0], [2, 3, 0, 0], [3, 4, 5, 0]])
    lengths, targets = torch.tensor([1, 2, 3]), torch.tensor([5, 6, 7])
    optimizer = torch.optim.AdamW(tiny.parameters(), lr=0.01)
    initial = nn.functional.cross_entropy(tiny(histories, lengths), targets).item()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(tiny(histories, lengths), targets)
        loss.backward()
        optimizer.step()
    final_scores = tiny(histories, lengths)
    final = nn.functional.cross_entropy(final_scores, targets).item()
    assert final < 0.1 * initial
    assert torch.equal(final_scores.argmax(dim=1), targets)


def test_scorer_disables_dropout_and_restores_training_mode():
    torch.manual_seed(7)
    model = TransformerRecommender(12, d_model=16, dropout=0.5)
    model.train()
    scorer = make_score_batch(model)
    histories, lengths = np.array([[1, 2, 0]]), np.array([2])
    first, second = scorer(histories, lengths), scorer(histories, lengths)
    np.testing.assert_allclose(first, second, atol=1e-6, rtol=1e-5)
    assert first.shape == (1, 12)
    assert np.isfinite(first).all()
    assert model.training
    assert all(p.grad is None for p in model.parameters())
    model.eval()
    scorer(histories, lengths)
    assert not model.training


def test_scorer_restores_mode_on_invalid_input(model):
    model.train()
    with pytest.raises(ValueError, match="right padding"):
        make_score_batch(model)(np.array([[0, 1]]), np.array([1]))
    assert model.training


def test_neural_scorer_works_with_existing_evaluator(model):
    histories = np.array([[1, 2, 0, 0], [3, 0, 0, 0]], dtype=np.int64)
    lengths = np.array([2, 1], dtype=np.int64)
    scorer = make_score_batch(model)
    scores = scorer(histories, lengths)
    examples = []
    for row in range(2):
        seen = histories[row, :lengths[row]]
        allowed_scores = scores[row].copy()
        allowed_scores[seen - 1] = -np.inf
        target = int(allowed_scores.argmax()) + 1
        examples.append(EvaluationExample(row + 1, histories[row], int(lengths[row]), target, seen))
    single = evaluate_examples(examples, scorer, 12, k=1, batch_size=1)
    batched = evaluate_examples(examples, scorer, 12, k=1, batch_size=2)
    assert single == batched
    assert batched["metrics"] == {"users": 2, "recall@1": 1.0, "ndcg@1": 1.0}
    # Excluding the previous top result must alter the next recommendation.
    top = examples[0].target_id
    remaining = scores[0].copy()
    remaining[np.append(examples[0].seen_ids, top) - 1] = -np.inf
    alternate = int(remaining.argmax()) + 1
    changed = replace(examples[0], target_id=alternate, seen_ids=np.append(examples[0].seen_ids, top))
    result = evaluate_examples([changed], scorer, 12, k=1)
    assert result["predictions"][0]["top_ids"] == [alternate]


@pytest.mark.parametrize("history,lengths", [
    ([[0, 0]], [0]),       # Empty history.
    ([[1, 0]], [2]),       # Claimed length includes padding.
    ([[1, 0, 2]], [2]),    # Interior padding.
    ([[0, 1]], [1]),       # Left padding.
    ([[13, 0]], [1]),      # Unknown item ID.
    ([[1, 2]], [3]),       # Length exceeds width.
])
def test_rejects_invalid_history(model, history, lengths):
    with pytest.raises(ValueError):
        model(torch.tensor(history), torch.tensor(lengths))
