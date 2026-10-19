# Movie Recommendation Project

SSY340 – Deep Machine Learning, Chalmers University of Technology.

Compare a pooled-history MLP with a causal transformer inspired by SASRec to predict a user's next highly rated movie. The goal is to assess recommendation quality and computational cost, including whether modelling interaction order helps. A popularity baseline provides a reference.

## Data and evaluation

- **Dataset:** MovieLens 1M; ratings ≥ 4 are positive interactions, ordered by rating timestamp.
- **Split:** Last positive interaction per user for testing, second-last for validation; earlier interactions for training. The validation interaction enters the test history.
- **History:** Up to 50 positive interactions.
- **Planned evaluation:** Full-catalogue ranking with previously rated movies excluded; Recall@10, NDCG@10, parameter counts and training time.

The target is the next recorded positive rating, not a predicted star rating or necessarily the next movie watched.

## Prepare data and check tests

From the project root, with the project's dependencies installed and MovieLens 1M extracted into `data/raw/ml-1m/`:

```bash
uv run python -m src.dataset \
  --data-dir data/raw/ml-1m \
  --output-dir data/processed \
  --max-len 50 \
  --check-batch

uv run python -m pytest tests/test_data.py tests/test_dataset.py -q
```

Latest recorded progress: 25 data/dataset tests passed; the dataset build produced 3,525 catalogue movies and 557,170 training-prefix examples. Model training and final evaluation results are not yet verified in this README.

## References

- Harper & Konstan (2015), *The MovieLens Datasets: History and Context*. https://doi.org/10.1145/2827872
- Kang & McAuley (2018), *Self-Attentive Sequential Recommendation*. https://arxiv.org/abs/1808.09781
- SASRec reference implementation: https://github.com/kang205/SASRec
