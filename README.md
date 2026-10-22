# Movie Recommender Systems

SSY340 / DIT969, Chalmers. Group 100: Karl Egeland and Erik Gälldin.

Compare a pooled-history MLP with a SASRec-inspired causal transformer on MovieLens 1M. Both use shared input/output movie embeddings, dot-product scoring, and full-catalogue cross-entropy. The target is the next recorded positive rating; timestamps record ratings rather than viewing order.

## Shared experiment

- Ratings **>= 4** are positive; require **>= 5 positives/user** (6,034 users).
- Order by timestamp, then movie ID. Hold out the second-last positive for validation and last for test; validation enters the test history.
- Maximum history **50**; train-only catalogue **3,525 movies**; **557,170 training prefixes**.
- Full-catalogue ranking excludes earlier raw ratings, including low ratings. Validation covers **6,030 users**; four unknown targets are excluded.

Both models use 64-dimensional embeddings and dropout 0.1. MLP: mean pooling and 64 -> 128 -> 64 layers. Transformer: two blocks, two heads, feed-forward width 256. AdamW (learning rate 0.001), batch size 256, maximum 20 epochs, patience 5; select by validation NDCG@10.

## Verified results

**Validation, training seed 42.** These are single-run results; neural-model timings use the same NVIDIA L4.

| Model | Recall@10 | NDCG@10 | Parameters | Best epoch | Training s/epoch |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.05257 | 0.02553 | 0 | - | - |
| Pooled MLP | 0.15937 | 0.08568 | 242,240 | 14 | 13.74 |
| Causal transformer | **0.31244** | **0.17777** | 328,960 | 19 | 25.47 |

Times exclude validation/startup. MLP completed 19 epochs; transformer completed 20. Strictly-later-target Recall@10: MLP **0.13227**, transformer **0.24007**.

Transformer Recall@10: original **0.31244**, shuffled timestamp ties **0.30746**, all retained history shuffled **0.10392** (shuffle means over seeds 42-44). This shows order sensitivity; timestamp ties and architectural differences limit causal claims about chronology.

Training seeds 43/44 have also completed; aggregation awaits review of their reports. Test evaluation is pending; the training tie-order ablation is postponed.

## Cloud training and reproduction

GPU runs used **Google Cloud Compute Engine, NVIDIA L4, conda `dml`, PyTorch 2.3.1+cu118**. Code, data, and `preparation.json` were copied from the Mac. Training used CUDA and `nohup` to continue after SSH disconnects.

From the project root on the configured VM, use a fresh output directory for each run:

```bash
conda activate dml
for task_model in mlp transformer; do
  python -m src.train --model "$task_model" \
    --output-dir "runs/${task_model}_20ep_seed42_l4" \
    --epochs 20 --patience 5 --seed 42 \
    --batch-size 256 --device cuda --num-threads 4
done
```

Use seeds 43 and 44 for repeats. On the Mac, run checks with `uv run python -m pytest tests/ -q`.

**Run artifacts:** `runs/<model>_20ep_seed<seed>_l4/` contains configuration, `history.jsonl` (learning curves), `validation_metrics.json`, `validation_predictions.jsonl`, and `best.pt`. Share the downloaded runs archive for the complete reports/checkpoints.

## References

- Harper & Konstan (2015), [The MovieLens Datasets: History and Context](https://doi.org/10.1145/2827872).
- Kang & McAuley (2018), [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781).
