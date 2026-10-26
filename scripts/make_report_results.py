"""Generate report figures and tables from saved experiment outputs; never trains models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
MODELS = ('mlp', 'transformer')
NAMES = {'mlp': 'Pooled MLP', 'transformer': 'Causal transformer', 'popularity': 'Popularity'}


def read_json(path):
    return json.loads(path.read_text())


def load_curves(root):
    runs = []
    for model in MODELS:
        for seed in SEEDS:
            folder = root / f'{model}_20ep_seed{seed}_l4'
            config = read_json(folder / 'config.json')
            report = read_json(folder / 'validation_metrics.json')
            rows = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines() if line.strip()]
            if config['model_type'] != model or config['training_config']['seed'] != seed:
                raise ValueError(f'{folder}: model/seed mismatch')
            if config['training_scope']['scope'] != 'full_training' or report['split'] != 'validation':
                raise ValueError(f'{folder}: expected full-training validation run')
            if not rows or [r['epoch'] for r in rows] != list(range(1, len(rows)+1)):
                raise ValueError(f'{folder}: missing/duplicated epochs')
            for r in rows:
                if not math.isfinite(r['train_loss']) or r['train_loss'] < 0:
                    raise ValueError(f'{folder}: invalid loss')
                for source in ('metrics', 'train_metrics'):
                    for metric in ('recall@10', 'ndcg@10'):
                        value = r[source][metric]
                        if not math.isfinite(value) or not 0 <= value <= 1:
                            raise ValueError(f'{folder}: invalid {source}/{metric}')
            best = max(rows, key=lambda r: r['metrics']['ndcg@10'])
            if best['epoch'] != report['best_epoch'] or best['metrics'] != report['metrics']:
                raise ValueError(f'{folder}: selected checkpoint mismatch')
            if len(rows) != report['epochs_completed']:
                raise ValueError(f'{folder}: incomplete history')
            if config['preparation_fingerprint'] != report['preparation_fingerprint']:
                raise ValueError(f'{folder}: preparation mismatch')
            runs.append((model, seed, config, rows, best['epoch']))
    for key in ('preparation_fingerprint',):
        if len({r[2][key] for r in runs}) != 1:
            raise ValueError('Runs use different preparations')
    if len({r[2]['training_probe']['examples_sha256'] for r in runs}) != 1:
        raise ValueError('Runs use different training subsets')
    return runs


def import_archive(archive, destination):
    """Copy only the 18 small JSON logs required for plotting; never extract paths."""
    names = [f'{model}_20ep_seed{seed}_l4/{file}' for model in MODELS for seed in SEEDS
             for file in ('config.json', 'history.jsonl', 'validation_metrics.json')]
    with tarfile.open(archive) as bundle:
        payloads = {}
        for name in names:
            member = bundle.getmember('runs/train_recall/' + name)
            if not member.isfile() or member.size > 10_000_000:
                raise ValueError(f'Unexpected archive member: {member.name}')
            payloads[name] = bundle.extractfile(member).read()
    for name, data in payloads.items():
        target = destination / name
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f'Different existing input: {target}')
    for name, data in payloads.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def latex_table(path, caption, label, header, rows, columns):
    text = ['\\begin{table*}[!t]', '\\centering', '\\caption{' + caption + '}',
            '\\label{' + label + '}', '\\begin{tabular}{' + columns + '}', '\\toprule',
            ' & '.join(header) + r' \\', '\\midrule']
    text += [' & '.join(row) + r' \\' for row in rows]
    text += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    path.write_text('\n'.join(text))


def score(row, key):
    mean = float(row[key + '_mean'])
    sd = row[key + '_sample_sd']
    if not math.isfinite(mean) or not 0 <= mean <= 1:
        raise ValueError('Invalid summary mean')
    if sd is None:
        return f'{mean:.5f}'
    sd = float(sd)
    if not math.isfinite(sd) or sd < 0:
        raise ValueError('Invalid summary standard deviation')
    return f'${mean:.5f}\\pm{sd:.5f}$'


def load_tables(test_path, diagnostics_path, original_root):
    comparison = read_json(test_path)
    if comparison['split'] != 'test' or sorted(comparison['training_seeds']) != list(SEEDS):
        raise ValueError('Expected original test comparison for seeds 42, 43, 44')
    summary = {(r['model'], r['cohort']): r for r in comparison['summary']}
    for model in (*MODELS, 'popularity'):
        for cohort, users in [('overall',6030),('strictly_later',4128),('tied_timestamp',1902)]:
            row = summary[(model,cohort)]
            if row['users'] != users or row['runs'] != (1 if model == 'popularity' else 3):
                raise ValueError('Unexpected test coverage or seed count')
            for metric in ('recall@10', 'ndcg@10'):
                score(row, metric)
    costs = {}
    for model in MODELS:
        reports = [r for r in comparison['per_run'] if r['model_type'] == model]
        if sorted(r['seed'] for r in reports) != list(SEEDS):
            raise ValueError('Missing or duplicate test seeds')
        parameters = {r['trainable_parameters'] for r in reports}
        if len(parameters) != 1:
            raise ValueError('Parameter counts differ across seeds')
        times = []
        for report in reports:
            folder = original_root / f"{model}_20ep_seed{report['seed']}_l4"
            cfg = read_json(folder / 'config.json')
            if cfg['preparation_fingerprint'] != comparison['preparation_fingerprint']:
                raise ValueError('Original training and test preparations differ')
            if cfg.get('training_config', {}).get('log_train_recall', False):
                raise ValueError('Timing inputs must be original runs, not diagnostic reruns')
            # Verify identity when checkpoints are retained; otherwise keep the
            # source selection metrics and epoch as a lightweight cross-check.
            selected = read_json(folder / 'validation_metrics.json')
            if selected['best_epoch'] != report['checkpoint_epoch'] or selected['metrics'] != report['validation_selection_metrics']:
                raise ValueError('Original validation selection does not match test report')
            history = [json.loads(x) for x in (folder / 'history.jsonl').read_text().splitlines() if x.strip()]
            times.append(statistics.mean(r['training_seconds'] for r in history))
        if not times or any(not math.isfinite(t) or t < 0 for t in times):
            raise ValueError('Invalid training timings')
        costs[model] = (parameters.pop(), statistics.mean(times))
    diag = read_json(diagnostics_path)
    if diag['split'] != 'validation' or diag['model_type'] != 'transformer' or not diag['original_matches_checkpoint']:
        raise ValueError('Expected fixed-transformer validation diagnostics')
    seed42 = next(r for r in comparison['per_run'] if r['model_type']=='transformer' and r['seed']==42)
    if diag['checkpoint_sha256'] != seed42['checkpoint_sha256']:
        raise ValueError('Diagnostics are not from the original seed-42 checkpoint')
    return summary, costs, diag


def write_tables(out, data):
    summary, costs, diag = data
    rows = []
    for model in ('popularity', *MODELS):
        r = summary[(model, 'overall')]
        params, timing = ('---', '---') if model == 'popularity' else (f'{costs[model][0]:,}', f'{costs[model][1]:.2f}')
        rows.append([NAMES[model],score(r,'recall@10'),score(r,'ndcg@10'),params,timing])
    latex_table(out/'test_performance.tex', 'Original test results: mean $\\pm$ sample SD across three training seeds. Popularity is deterministic. Time is L4 training seconds per epoch averaged within each run, then across seeds, excluding evaluation.', 'tab:main-results', ['Model','Recall@10','NDCG@10','Parameters','Training s/epoch'], rows, 'lcccc')
    rows = []
    for cohort,title in [('strictly_later','Strictly later'),('tied_timestamp','Same timestamp')]:
        a,b = [summary[(m,cohort)] for m in MODELS]
        rows.append([title,str(a['users']),score(a,'recall@10'),score(b,'recall@10'),score(a,'ndcg@10'),score(b,'ndcg@10')])
    latex_table(out/'timestamp_groups.tex','Original test results grouped by target timestamp relative to the preceding positive interaction; mean $\\pm$ sample SD across training seeds.', 'tab:timestamp-results', ['Target group','Users','MLP Recall','Transformer Recall','MLP NDCG','Transformer NDCG'],rows,'lrcccc')
    rows=[]
    for mode,title in [('original','Original'),('shuffle_ties','Shuffle timestamp ties'),('shuffle_all','Shuffle all history')]:
        metrics=diag[mode]['metrics' if mode=='original' else 'mean_metrics']
        rows.append([title,f"{metrics['recall@10']:.5f}",f"{metrics['ndcg@10']:.5f}"])
    latex_table(out/'order_sensitivity.tex','Original seed-42 transformer on validation, without retraining. Shuffled rows average over permutation seeds.', 'tab:order-sensitivity',['History order','Recall@10','NDCG@10'],rows,'lcc')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,help='Downloaded train_recall_results.tar.gz; imports only plotting logs')
    p.add_argument('--curves-root',type=Path,default=ROOT/'results/training_curves')
    p.add_argument('--test-summary',type=Path,default=ROOT/'results/original_test/comparison.json')
    p.add_argument('--diagnostics',type=Path,default=ROOT/'results/order_diagnostics/diagnostics.json')
    p.add_argument('--original-runs',type=Path,default=ROOT/'runs')
    p.add_argument('--output-dir',type=Path,default=ROOT/'report/generated')
    p.add_argument('--curves-only',action='store_true',help='Generate only the two figures; original test inputs not required')
    p.add_argument('--validate-only',action='store_true',help='Check inputs without generating figures or tables')
    args=p.parse_args()
    if args.archive:
        import_archive(args.archive,args.curves_root)
    runs=load_curves(args.curves_root)
    tables=None if args.curves_only else load_tables(args.test_summary,args.diagnostics,args.original_runs)
    if args.validate_only:
        print('Input validation passed; no figures or tables generated.')
        return
    out=args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output is not empty: {out}; select a new --output-dir')
    # Reuse the existing, independently usable recall plotting entry point.
    subprocess.run([sys.executable,'-m','src.plot_train_recall','--runs-root',str(args.curves_root.resolve()),'--output-dir',str(out.resolve())],cwd=ROOT,check=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(7,2.7),sharex=True,sharey=True,layout='constrained')
    for ax,model in zip(axes,MODELS):
        for m,seed,cfg,rows,best in runs:
            if m==model:
                ax.plot([r['epoch'] for r in rows],[r['train_loss'] for r in rows],label=f'Seed {seed}')
        ax.set(title=NAMES[model],xlabel='Epoch',xticks=[1,5,10,15,20]);ax.grid(alpha=.2)
    axes[0].set_ylabel('Training cross-entropy');axes[1].legend(frameon=False,fontsize=8)
    fig.savefig(out/'training_loss.pdf');fig.savefig(out/'training_loss.png',dpi=200);plt.close(fig)
    if tables:
        write_tables(out,tables)
    paths=list(args.curves_root.glob('*/*.json'))+list(args.curves_root.glob('*/history.jsonl'))
    if tables:
        paths += [args.test_summary,args.diagnostics]
        for model in MODELS:
            for seed in SEEDS:
                paths += [args.original_runs/f'{model}_20ep_seed{seed}_l4'/n for n in ('config.json','history.jsonl','validation_metrics.json')]
    provenance={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (out/'input_hashes.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Report outputs saved to {out}')


if __name__=='__main__':
    main()
