"""Plot each rerun separately; best epoch is selected by validation NDCG@10."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Seeds must be distinct.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    inputs = []
    for model in ('mlp', 'transformer'):
        for seed in args.seeds:
            run = args.runs_root / f'{model}_20ep_seed{seed}_l4'
            config = json.loads((run / 'config.json').read_text())
            rows = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
            report = json.loads((run / 'validation_metrics.json').read_text())
            if not rows or any('train_metrics' not in r for r in rows):
                raise ValueError(f'{run}: missing training metrics; rerun with --log-train-recall.')
            if config['training_scope']['scope'] != 'full_training':
                raise ValueError(f'{run}: trial run cannot be used here.')
            if config['model_type'] != model or config['training_config']['seed'] != seed:
                raise ValueError(f'{run}: model/seed metadata mismatch.')
            best = max(rows, key=lambda r: r['metrics']['ndcg@10'])['epoch']
            if best != report['best_epoch']:
                raise ValueError(f'{run}: best epoch does not match the saved report.')
            if [r['epoch'] for r in rows] != list(range(1, len(rows)+1)):
                raise ValueError(f'{run}: missing or duplicated epochs.')
            inputs.append((model, seed, config, rows, best))
    if len({x[2]['training_probe']['examples_sha256'] for x in inputs}) != 1:
        raise ValueError('Runs use different training probes.')
    if len({x[2]['preparation_fingerprint'] for x in inputs}) != 1:
        raise ValueError('Runs use different preparation manifests.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Choose an empty figure output directory.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, len(args.seeds), figsize=(5*len(args.seeds), 7.5),
                             squeeze=False, sharex=True, sharey=True, layout='constrained')
    def draw(ax, model, seed, config, rows, best):
        epochs = [r['epoch'] for r in rows]
        ax.plot(epochs, [r['train_metrics']['recall@10'] for r in rows],
                label='Train (one prefix/user)', color='#176B9C')
        ax.plot(epochs, [r['metrics']['recall@10'] for r in rows],
                label='Validation', color='#C45D18')
        ax.axvline(best, color='0.45', ls=':', label=f'Best val. NDCG (epoch {best})')
        ax.set(title=f'{"Pooled MLP" if model == "mlp" else "Transformer"} | seed {seed}',
               xlabel='Epoch', ylabel='Recall@10', xlim=(1, config['training_config']['epochs']),
               ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xticks([1,5,10,15,20]);ax.grid(alpha=.2)
        ax.legend(fontsize=8, frameon=False, loc='best')
    for i, item in enumerate(inputs):
        draw(axes[i//len(args.seeds), i%len(args.seeds)], *item)
        single, ax = plt.subplots(figsize=(6,4), layout='constrained')
        draw(ax, *item)
        stem = f'{item[0]}_seed{item[1]}_train_validation_recall'
        single.savefig(args.output_dir / f'{stem}.pdf')
        single.savefig(args.output_dir / f'{stem}.png', dpi=200)
        plt.close(single)
    fig.savefig(args.output_dir / 'all_train_validation_recall.pdf')
    fig.savefig(args.output_dir / 'all_train_validation_recall.png', dpi=200)
    plt.close(fig)
    print(f'Saved figures to {args.output_dir}')


if __name__ == '__main__':
    main()
