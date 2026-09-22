"""Audit saved results and snapshot the evidence used by the manuscript.

Reads the experiment directory without modifying it. Recomputes every L18
score-column AUC/AP with sklearn and all layers' aggregates/intervals with
NumPy. Does not rerun extraction, fit directions, or generate bootstrap draws.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-tree', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    snapshot = {'run': args.run_dir.name, 'datasets': {}, 'source_sha256': {}}
    audit = {'scope': __doc__, 'layers': {}, 'source_code_hashes_match': {}}
    provenance = json.loads((args.run_dir / 'code_provenance.json').read_text())
    for name, expected in provenance['source_sha256'].items():
        actual = digest(args.source_tree / name)
        audit['source_code_hashes_match'][name] = actual == expected
    # Documentation may have been updated after the run; executing code must match.
    assert all(match for name, match in audit['source_code_hashes_match'].items()
               if name.endswith('.py'))
    for name in ('code_provenance.json', 'runtime_provenance.json'):
        snapshot[name.removesuffix('.json')] = json.loads((args.run_dir / name).read_text())
        snapshot['source_sha256'][name] = digest(args.run_dir / name)
    for dataset in ('core', 'ext2'):
        folder = args.run_dir / dataset
        manifest = json.loads((folder / 'fit_manifest.json').read_text())
        saved = {'manifest': manifest, 'layers': {}}
        snapshot['datasets'][dataset] = saved
        snapshot['source_sha256'][f'{dataset}/fit_manifest.json'] = digest(folder / 'fit_manifest.json')
        labels = torch.load(folder / 'labels.pt', map_location='cpu', weights_only=False)
        for layer in (1, 9, 18, 27, 35):
            name = f'results_L{layer:02d}.json'
            result = json.loads((folder / name).read_text())
            saved['layers'][str(layer)] = result
            snapshot['source_sha256'][f'{dataset}/{name}'] = digest(folder / name)
            assert result['n_boot'] == result['minimum_valid_comparison_draws'] == 2000
            assert manifest['layers'][str(layer)]['A0_max_abs_error'] == 0
            bank = torch.load(folder / f'directions_L{layer:02d}.pt', map_location='cpu', weights_only=False)
            metrics = torch.load(folder / f'bootstrap_L{layer:02d}.pt', map_location='cpu', weights_only=False)['metrics']
            metrics = {k: v.numpy() if torch.is_tensor(v) else v for k, v in metrics.items()}
            errors = {'point_auc_max_error': 0., 'point_ap_max_error': 0.,
                      'aggregate_max_error': 0., 'interval_max_error': 0.,
                      'independent_score_columns': 0, 'comparisons_checked': 0}

            def check(key, actual, expected):
                error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
                errors[key] = max(errors[key], error)
                assert error < 1e-10, (dataset, layer, key, error)

            if layer == 18:
                scores = torch.load(folder / 'scores_L18.pt', map_location='cpu', weights_only=False, mmap=True)['S'].numpy()
                y, eligible = labels['Y'].numpy(), labels['E'].numpy()
                for k, col in enumerate(bank['columns']):
                    c = col['class']
                    mask = eligible[:, c]
                    truth, score = y[mask, c], scores[mask, k]
                    check('point_auc_max_error', roc_auc_score(truth, score), metrics['auc'][k])
                    check('point_ap_max_error', average_precision_score(truth, score), metrics['ap'][k])
                    errors['independent_score_columns'] += 1
                    if (k + 1) % 2000 == 0:
                        print(dataset, layer, 'score columns', k + 1, flush=True)
                del scores
            groups = defaultdict(list)
            for group in bank['groups']:
                groups[group['variant']].append(group)
            for scenario, block in result['scenarios'].items():
                if not block['classes']:
                    continue
                classes = [bank['names'].index(c) for c in block['classes']]
                macro_boot, seed_auc = {}, {}
                for variant, row in block['variants'].items():
                    indices = []
                    for group in groups[variant]:
                        mapping = {c: group['start'] + j for j, c in enumerate(group['classes'])}
                        indices.append([mapping[c] for c in classes])
                    ks = np.asarray(indices)
                    auc, ap = metrics['auc'][ks], metrics['ap'][ks]
                    seed_auc[variant] = auc.mean(axis=1)
                    macro_boot[variant] = metrics['bootstrap_auc'][:, ks].mean(axis=(1, 2))
                    check('aggregate_max_error', auc.mean(), row['macro_auc'])
                    check('aggregate_max_error', ap.mean(), row['macro_ap'])
                    check('aggregate_max_error', auc.mean(axis=1), row['seed_macro_auc'])
                    for j, c in enumerate(block['classes']):
                        check('aggregate_max_error', auc[:, j].mean(), row['per_class_auc'][c])
                        check('aggregate_max_error', ap[:, j].mean(), row['per_class_ap'][c])
                    if row['n_seeds'] > 1:
                        check('aggregate_max_error', auc.mean(axis=1).std(ddof=1), row['seed_macro_auc_sd'])
                    assert np.isfinite(macro_boot[variant]).all()
                    check('interval_max_error', np.quantile(macro_boot[variant], [.025, .975]), row['image_bootstrap_95']['interval'])
                for comparison, row in block['comparisons'].items():
                    lhs, rhs = comparison.split('-')
                    delta = macro_boot[lhs] - macro_boot[rhs]
                    check('aggregate_max_error', seed_auc[lhs].mean() - seed_auc[rhs].mean(), row['delta_macro_auc'])
                    check('interval_max_error', np.quantile(delta, [.025, .975]), row['image_bootstrap_95']['interval'])
                    if row['primary']:
                        check('interval_max_error', np.quantile(delta, [.0125, .9875]), row['bonferroni_simultaneous_97_5']['interval'])
                    if 'seed_delta_sd' in row:
                        check('aggregate_max_error', (seed_auc[lhs] - seed_auc[rhs]).std(ddof=1), row['seed_delta_sd'])
                    errors['comparisons_checked'] += 1
            audit['layers'][f'{dataset}_L{layer}'] = errors
            print(dataset, layer, errors, flush=True)
    out = ROOT / 'results' / 'competitor_ablation'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'verified_metrics.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    (out / 'verification.json').write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
