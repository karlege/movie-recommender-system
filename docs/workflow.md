# Experiment workflow

## Protocol

Ratings >=4 are positive; retain users with at least five positives. Sort by timestamp then movie ID. Hold out the penultimate positive for validation and the last for test. Test history includes the observed validation interaction. Training-only catalogue: 3,525 movies; 557,170 training prefixes. Maximum history: 50. Validation and test each contain 6,030 eligible targets; four unknown targets per split are excluded. Ranking excludes all earlier raw ratings, including low ratings. No negative candidates are sampled.

Both models use 64-dimensional item embeddings. MLP: masked mean, 64->128->64. Transformer: two causal blocks, two heads, feed-forward width 256. AdamW learning rate .001, weight decay .01, batch size 256, gradient clipping 1, maximum 20 epochs, patience 5. Select the first maximum validation NDCG@10 epoch. Seeds: 42,43,44.

## Code map

| Stage | Entry point / implementation |
|---|---|
| Read ratings and split histories | src/data.py |
| Build prefixes, vocabulary, candidate exclusions and preparation manifest | src/dataset.py |
| Model definitions | src/models/mlp.py, src/models/transformer.py |
| Train and select checkpoint | src/train.py |
| Shared ranking and metrics | src/scoring.py, src/evaluate.py |
| Popularity and restore preparation | src/baselines.py |
| Frozen-checkpoint test comparison | src/final_evaluate.py |
| Evaluation-time order permutations | src/diagnose.py |
| In-sample ranking subset | src/training_probe.py |
| Recall plots, also usable independently | src/plot_train_recall.py |
| Report figures and tables | scripts/make_report_results.py |

## Prepare a new experiment

Place MovieLens files in data/raw/ml-1m, then run:

```bash
python -m src.dataset --data-dir data/raw/ml-1m --positive-threshold 4 --min-positive 5 --max-len 50
```

For existing experiments retain their manifest. Source hashes include docstrings; if restoration fails, review the source differences and verify protocol, raw-file hashes, movie-index order and counts before regenerating. Do not manually bypass hashes.

## Train diagnostic runs on the configured GPU VM

```bash
conda activate dml
bash run_recall_curves.sh
```

This records train and validation recall for seeds 42/43/44, and produces recall plots. Training recall uses one fixed trained prefix per eligible user, not all training prefixes. These runs are supplementary diagnostics, separate from original test results.

For an individual original-protocol run without training-recall logging:

```bash
python -m src.train --model mlp --output-dir runs/mlp_20ep_seed42_l4 --epochs 20 --patience 5 --seed 42 --batch-size 256 --device cuda --num-threads 4
```

Repeat for both architectures and all three seeds in fresh directories as required. Never overwrite original runs.

## Evaluate frozen original checkpoints

```bash
python -m src.final_evaluate --runs-root runs --output-dir runs/final_test --seeds 42 43 44 --device cpu --num-threads 4
python -m src.diagnose --help
```

Final evaluation requires original raw data, compatible source hashes, the original preparation manifest and frozen checkpoints. It does not fit models. The diagnostic CLI documents its input/output options. Use the original seed-42 transformer for the reported permutation experiment.
