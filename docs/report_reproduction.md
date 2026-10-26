# Generate report results

Run commands from the repository root. No GPU is required for plotting.

## 1. Import downloaded diagnostic runs

```bash
uv run python scripts/make_report_results.py --archive /absolute/path/to/train_recall_results.tar.gz --curves-only --validate-only
```

This copies config.json, history.jsonl and validation_metrics.json for six runs into results/training_curves/. It does not extract checkpoints, predictions or existing figures. Existing identical inputs are accepted; differing inputs are rejected.

## 2. Generate the two figures locally

```bash
uv run python scripts/make_report_results.py --curves-only
```

Use all_train_validation_recall.pdf and training_loss.pdf from report/generated/. The recall plot also writes individual seed figures. Training loss averages over all training prefixes; training recall uses one fixed trained prefix per eligible user. The vertical marker is selected by validation NDCG, not recall. Curves are diagnostic reruns; original test metrics remain separate.

## 3. Supply original results for tables

Copy these existing experiment outputs, without recomputing or editing values:

- Original runs/final_test/comparison.json -> results/original_test/comparison.json
- Original seed-42 transformer's diagnostics.json -> results/order_diagnostics/diagnostics.json
- Original six run folders under runs/, named mlp_20ep_seed42_l4 etc. Only config.json, history.jsonl and validation_metrics.json are needed for this reporting command.

The source ZIP contains the rounded subgroup values in docs/reported_results.md, but no primary final-test files. The reporting command requires primary files instead of treating rounded README values as independently verified data.

## 4. Generate everything

If report/generated/ already contains figures, select a fresh destination:

```bash
uv run python scripts/make_report_results.py --output-dir report/generated_complete
```

Use --validate-only first if desired. Missing or incompatible inputs cause an error before plotting. No training is performed. Timing is computed as mean seconds/epoch within each original run, then averaged equally across seeds (13.77 and 25.57 for the reported original logs). This differs from pooling all epochs when early stopping gives different run lengths.

| Report element | Generated file | Source |
|---|---|---|
| Training/validation recall | all_train_validation_recall.pdf | Diagnostic history, config and validation selection |
| Training cross-entropy | training_loss.pdf | Same diagnostic histories |
| Final test performance and cost | test_performance.tex | Original comparison.json and original training logs |
| Timestamp groups | timestamp_groups.tex | Original comparison.json |
| Order sensitivity | order_sensitivity.tex | Original diagnostics.json |

input_hashes.json records the source file hashes. Commit the reporting scripts and small inputs under results/ if suitable; keep generated PDFs, raw data and checkpoints out of Git. Original runs/ is ignored, so either keep those logs locally or copy their three small files per run to results/original_training/ and pass --original-runs results/original_training. Do not commit per-user prediction files merely to generate plots.

## 5. Overleaf

Add graphicx and booktabs to the preamble alongside amsmath. Upload the two PDFs into figures/ and the three generated .tex tables into the project root.

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/all_train_validation_recall.pdf}
\caption{Training-subset and validation Recall@10 in diagnostic reruns. Training recall evaluates one fixed trained prefix per eligible user. Vertical lines mark the best validation NDCG@10 epoch.}
\label{fig:recall-curves}
\end{figure*}

\begin{figure*}[!t]
\centering
\includegraphics[width=0.9\textwidth]{figures/training_loss.pdf}
\caption{Training cross-entropy over all training prefixes for the same diagnostic reruns, with three training seeds per model.}
\label{fig:loss-curves}
\end{figure*}

\input{test_performance}
\input{timestamp_groups}
\input{order_sensitivity}
```

Place figure blocks in Training behavior, the first table in Final test performance and complexity, and the other tables in Timestamp groups and order sensitivity. Replace corresponding previous table/figure blocks rather than adding duplicates. Wide floats may appear at the top of the next page.
