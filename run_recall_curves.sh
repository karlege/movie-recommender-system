#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
for model in mlp transformer; do
  for seed in 42 43 44; do
    python -m src.train \
      --model "$model" \
      --output-dir "runs/train_recall/${model}_20ep_seed${seed}_l4" \
      --epochs 20 --patience 5 --seed "$seed" \
      --batch-size 256 --device cuda --num-threads 4 \
      --log-train-recall --quiet
  done
done
python -m src.plot_train_recall \
  --runs-root runs/train_recall \
  --output-dir figures/train_recall
