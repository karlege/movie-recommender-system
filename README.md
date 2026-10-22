# Movie Recommender Systems

SSY340 / DIT969, Chalmers. Group 100: Karl Egeland and Erik Gälldin.

Comparison of a pooled-history MLP and a SASRec-inspired causal transformer for predicting the next recorded positive interaction on MovieLens 1M. Each model shares its history/candidate movie embeddings and uses full-catalogue cross-entropy.

## Shared experiment

- Ratings **>= 4** are positive; require **>= 5 positives/user** (6,034 users).
- Order by timestamp, then movie ID. Hold out the second-last positive for validation and last for test; validation enters the test history.
- Maximum history **50**; train-only catalogue **3,525 movies**; **557,170 training prefixes**.
- Full-catalogue ranking excludes earlier raw ratings, including low ratings. Validation covers **6,030 users**; four unknown targets are excluded.

Both models use 64-dimensional embeddings and dropout 0.1. MLP: mean pooling and 64 -> 128 -> 64 layers. Transformer: two blocks, two heads, feed-forward width 256. AdamW (learning rate 0.001), batch size 256, maximum 20 epochs, patience 5; select by validation NDCG@10.

## Validation results

**Mean +/- sample SD across training seeds 42, 43, and 44**, using each run's best validation-NDCG checkpoint and the same data preparation. SD describes variation across seeds.

| Model | Recall@10 | NDCG@10 | Parameters | Training s/epoch |
|---|---:|---:|---:|---:|
| Popularity (deterministic) | 0.05257 | 0.02553 | 0 | - |
| Pooled MLP | 0.16130 +/- 0.00168 | 0.08539 +/- 0.00040 | 242,240 | 13.77 |
| Causal transformer | **0.30934 +/- 0.00307** | **0.17593 +/- 0.00160** | 328,960 | 25.57 |

Times average per-run training seconds/epoch, excluding validation/startup. Best epochs (seeds 42/43/44): MLP **14/19/20**, transformer **19/17/10**. Strictly-later-target mean Recall@10: MLP **0.13727**, transformer **0.23506**.

For the fixed seed-42 transformer checkpoint, Recall@10 was **0.31244** originally, **0.30746** after shuffling timestamp ties, and **0.10392** after shuffling all retained history (means over three shuffle seeds).

Test results are pending. Rating timestamps may differ from viewing order, and ties are resolved by movie ID. The shuffle diagnostics show order sensitivity; timestamp ties and architectural differences limit causal claims about chronology. A retrained tie-order ablation is outside the reported experiments.

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
