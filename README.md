# Movie Recommender Systems

SSY340 - Deep ML course project, Chalmers. Group 100: Karl Egeland and Erik Gälldin.

Comparison of a pooled-history MLP and a SASRec-inspired causal transformer for predicting the next recorded positive interaction on MovieLens 1M. Each model shares its history/candidate movie embeddings and uses full-catalogue cross-entropy.

## Shared experiment

- Ratings **>= 4** are positive; require **>= 5 positives/user** (6,034 users).
- Order by timestamp, then movie ID. Hold out the second-last positive for validation and last for test; validation enters the test history.
- Maximum history **50**; train-only catalogue **3,525 movies**; **557,170 training prefixes**.
- Full-catalogue ranking excludes earlier raw ratings, including low ratings. Validation and test each cover **6,030 users**; four unknown targets per split are excluded.

Both models use 64-dimensional embeddings and dropout 0.1. MLP: mean pooling and 64 -> 128 -> 64 layers. Transformer: two blocks, two heads, feed-forward width 256. AdamW (learning rate 0.001), batch size 256, maximum 20 epochs, patience 5; select by validation NDCG@10.

## Results

**Mean +/- sample SD across training seeds 42, 43, and 44**, using each run's best validation-NDCG checkpoint. Popularity is deterministic. SD describes variation across seeds, not a confidence interval.

### Final test results

All six frozen checkpoints were evaluated on the same test split on CPU; no test-based fitting or checkpoint selection was performed.

| Model | Recall@10 | NDCG@10 | Parameters | Training s/epoch |
|---|---:|---:|---:|---:|
| Popularity | 0.04461 | 0.02223 | 0 | - |
| Pooled MLP | 0.13566 +/- 0.00241 | 0.06985 +/- 0.00115 | 242,240 | 13.77 |
| Causal transformer | **0.29585 +/- 0.00158** | **0.16712 +/- 0.00180** | 328,960 | 25.57 |

Times are average **L4 training seconds/epoch**, excluding validation/startup; they are not CPU test timings.

Test results by the timestamp relation between target and previous positive interaction:

| Cohort | Users | Model | Recall@10 | NDCG@10 |
|---|---:|---|---:|---:|
| Strictly later | 4,128 | Popularity | 0.04651 | 0.02340 |
| Strictly later | 4,128 | Pooled MLP | 0.10812 +/- 0.00414 | 0.05563 +/- 0.00190 |
| Strictly later | 4,128 | Causal transformer | 0.21334 +/- 0.00505 | 0.11419 +/- 0.00302 |
| Tied timestamp | 1,902 | Popularity | 0.04048 | 0.01971 |
| Tied timestamp | 1,902 | Pooled MLP | 0.19541 +/- 0.00428 | 0.10069 +/- 0.00185 |
| Tied timestamp | 1,902 | Causal transformer | 0.47494 +/- 0.00598 | 0.28199 +/- 0.00085 |

### Validation results

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| Popularity | 0.05257 | 0.02553 |
| Pooled MLP | 0.16130 +/- 0.00168 | 0.08539 +/- 0.00040 |
| Causal transformer | 0.30934 +/- 0.00307 | 0.17593 +/- 0.00160 |

Selected epochs (seeds 42/43/44): MLP **14/19/20**, transformer **19/17/10**.

For the fixed seed-42 transformer checkpoint, **validation** Recall@10 was **0.31244** originally, **0.30746** after shuffling timestamp ties, and **0.10392** after shuffling all retained history (means over three shuffle seeds).

Rating timestamps may differ from viewing order, and ties are resolved by movie ID. The shuffle diagnostics show order sensitivity; timestamp ties and architectural differences limit causal claims about chronology. A retrained tie-order ablation is outside the reported experiments.

## Cloud training and reproduction

All six runs used **Google Cloud Compute Engine with an NVIDIA L4**, PyTorch 2.3.1+cu118, and conda. Code, data, and the original preparation manifest were transferred to the VM. CUDA training launched with `nohup` continued after SSH disconnects.

From the project root on the configured VM, reproduce all six runs in a fresh directory:

```bash
conda activate dml
for task_model in mlp transformer; do
  for task_seed in 42 43 44; do
    python -m src.train --model "$task_model" \
      --output-dir "runs/reproduction/${task_model}_20ep_seed${task_seed}_l4" \
      --epochs 20 --patience 5 --seed "$task_seed" \
      --batch-size 256 --device cuda --num-threads 4 || break 2
  done
done
```

Run checks on the Mac with `uv run python -m pytest tests/ -q`.

For the downloaded L4 runs in `runs/`, evaluate the frozen validation-selected checkpoints and training-popularity baseline on test data:

```bash
uv run python -m src.final_evaluate \
  --runs-root runs --output-dir runs/final_test \
  --seeds 42 43 44 --device cpu --num-threads 4
```

On the VM, use `python` and `--device cuda`. For reproduced runs, set `--runs-root runs/reproduction`. Keep the original `data/processed/preparation.json`. Testing uses validation interactions in histories while retaining training-only weights, vocabulary, and popularity counts. Reserve test results for final reporting.

## Output files

Each training directory contains `config.json`, `history.jsonl`, `validation_metrics.json`, `validation_predictions.jsonl`, and `best.pt`. Final evaluation writes per-run test metrics/predictions and `comparison.md`, `comparison.csv`, and `comparison.json`, including results for strictly later and tied-timestamp targets. Output directories must be empty.

## References

- Harper & Konstan (2015), [The MovieLens Datasets: History and Context](https://doi.org/10.1145/2827872).
- Kang & McAuley (2018), [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781).
