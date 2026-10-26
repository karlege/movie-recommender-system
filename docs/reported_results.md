# Reported experiment results

These rounded values are retained from the project README. Primary machine-readable test outputs are not included in the source archive. Report tables should be generated from the original comparison.json.

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

