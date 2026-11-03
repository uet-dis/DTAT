"""Non-parametric comparison of defense variants over evaluation conditions (E6).

A block is one evaluation condition, e.g. (dataset, vector set, source case, procedure)
for the paper protocol or (dataset, experiment, constraint level, eps, source) for the
constrained attacks; the value is the variant's ASR averaged over its seeds (lower is
better). Reported:
  * Friedman test over variants observed in every block, with mean ranks and the
    Nemenyi critical difference at alpha = 0.05;
  * Wilcoxon signed-rank tests of a reference variant against each other variant,
    paired over blocks, with Holm correction.
Blocks are not independent (they share test records, checkpoints and attack
sources), so p-values describe consistency across conditions, not sampling error.
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from revision_paths import revision_dir, OUTPUT_ROOT  # noqa: E402
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
RES_DIRS = {'nslkdd': 'NSLKDD', 'cic2018': 'CIC-2018'}
TREES = ['xgb', 'catb', 'gbm', 'bagging']
# Nemenyi q_alpha (alpha = 0.05) for k = 2..10 treatments (Demsar, 2006).
Q05 = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354}


def paper_protocol_blocks(level: str) -> pd.DataFrame:
    """Blocks from the paper's evaluators: tree-mean ('comp') or ensemble ('ens') ASR."""
    rows = []
    for res, ds in RES_DIRS.items():
        for path in glob.glob(os.path.join(revision_dir(ds), 'eval', '*', f'*_{level}.csv')):
            subset = os.path.basename(os.path.dirname(path))
            variant, case = os.path.basename(path)[:-len(f'_{level}.csv')].rsplit('_', 1)
            d = pd.read_csv(path)
            if level == 'comp':
                d = d[d.model.isin(TREES)].groupby('attack').asr_mean.mean().reset_index()
            for _, r in d.iterrows():
                # Query-based rows do not depend on the source case; keep them once.
                if r.attack in ('zoo', 'hsja') and case == 'tabnet':
                    continue
                rows.append({'block': f"{res}|{subset}|{case}|{r.attack}", 'variant': variant,
                             'asr': 100 * float(r.asr_mean)})
    return pd.DataFrame(rows)


def constrained_blocks() -> pd.DataFrame:
    rows = []
    for res, ds in RES_DIRS.items():
        files = glob.glob(os.path.join(revision_dir(ds), 'constrained', '*.csv'))
        if not files:
            continue
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        ens = d[d.experiment == 'ens-rs']
        sq = d[d.experiment == 'ens-sq']
        sq_tree = d[(d.experiment == 'comp-sq') & d.target.isin(TREES)]
        sq_tree = sq_tree.groupby(['level', 'eps', 'source', 'variant', 'seed', 'attack_seed'] if 'attack_seed' in sq_tree
                                  else ['level', 'eps', 'source', 'variant', 'seed'], dropna=False).asr.mean().reset_index()
        sq_tree['source'] = 'trees'
        tree = d[(d.experiment == 'cpgd') & d.target.isin(TREES)]
        tree = tree.groupby(['level', 'eps', 'source', 'variant', 'seed'], dropna=False).asr.mean().reset_index()
        for name, part, keys in (('ens-rs', ens, ['level', 'eps']), ('ens-sq', sq, ['level', 'eps']),
                                 ('comp-sq', sq_tree, ['level', 'eps']), ('cpgd-tree', tree, ['level', 'eps', 'source'])):
            g = part.groupby(keys + ['variant'], dropna=False).asr.mean().reset_index()
            for _, r in g.iterrows():
                block = '|'.join([res, name] + [str(r[k]) for k in keys])
                rows.append({'block': block, 'variant': r.variant, 'asr': float(r.asr)})
    return pd.DataFrame(rows)


def analyse(df: pd.DataFrame, reference: str, title: str) -> str:
    if df.empty:
        return f"## {title}\n\n_no data_\n"
    wide = df.pivot_table(index='block', columns='variant', values='asr', aggfunc='mean')
    wide = wide.dropna(axis=1, thresh=int(0.9 * len(wide))).dropna(axis=0)
    out = [f"## {title}", f"Blocks: {len(wide)}; variants: {', '.join(wide.columns)}"]
    if wide.shape[1] >= 3 and len(wide) >= 3:
        chi2, p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
        ranks = wide.rank(axis=1, method='average').mean().sort_values()
        k, n = wide.shape[1], len(wide)
        cd = Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6.0 * n))
        out.append(f"Friedman chi2 = {chi2:.2f}, p = {p:.2e}; Nemenyi CD (alpha=0.05) = {cd:.2f}")
        out.append('| variant | mean rank (1 = lowest ASR) |\n|---|---|\n'
                   + '\n'.join(f"| {v} | {r:.2f} |" for v, r in ranks.items()))
    if reference in wide:
        res = []
        for v in wide.columns:
            if v == reference:
                continue
            diff = wide[v] - wide[reference]
            if np.allclose(diff, 0):
                continue
            stat, p = stats.wilcoxon(wide[v], wide[reference], zero_method='wilcox', alternative='two-sided')
            res.append((v, int((diff > 0).sum()), int((diff < 0).sum()), float(diff.median()), p))
        if res:
            ps = np.array([r[4] for r in res])
            order = np.argsort(ps)
            holm = np.empty_like(ps)
            running = 0.0
            for i, j in enumerate(order):
                running = max(running, min(1.0, (len(ps) - i) * ps[j]))
                holm[j] = running
            out.append(f"Wilcoxon signed-rank vs {reference} (other minus {reference}; Holm-adjusted p):")
            out.append('| variant | blocks other > ref | blocks other < ref | median diff (pp) | p (Holm) |\n'
                       '|---|---|---|---|---|\n'
                       + '\n'.join(f"| {v} | {a} | {b} | {m:.2f} | {h:.2e} |"
                                   for (v, a, b, m, _), h in zip(res, holm)))
    return '\n\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', default='dtat')
    ap.add_argument('--out', default=os.path.join(OUTPUT_ROOT, 'stats.md'))
    args = ap.parse_args()
    parts = ['# Revision statistics (auto-generated by cli/hedge/9_revision/stats_revision.py)',
             'Blocks are evaluation conditions; they are not independent samples.']
    comp = paper_protocol_blocks('comp')
    ens = paper_protocol_blocks('ens')
    cons = constrained_blocks()
    parts.append(analyse(comp, args.reference, 'Paper protocol, tree-mean ASR (all vector sets)'))
    parts.append(analyse(ens, args.reference, 'Paper protocol, per-branch ensemble ASR (all vector sets)'))
    parts.append(analyse(cons, args.reference, 'Constrained and defense-aware attacks'))
    for variant_set, name in ((['base', 'iat_paper', 'dtat'], 'paper variants only'),):
        sub = cons[cons.variant.isin(variant_set)]
        parts.append(analyse(sub, args.reference, f'Constrained attacks, {name}'))
    with open(args.out, 'w') as fh:
        fh.write('\n\n'.join(parts))
    print(args.out)


if __name__ == '__main__':
    main()
