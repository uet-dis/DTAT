"""Collect revision results into Markdown tables (paper/revision_results.md).

Sections:
  E1  constraint audit of the paper's evaluation vectors
  E2  post-hoc Pi_Omega: tree-mean (Table 7 analogue) and per-branch ensemble ASR
      (Table 8 analogue) for every variant and vector set
  E3/E4 constrained in-loop attacks: ensemble random search (one vector to all
      branches), constrained PGD transferred to trees, and the adaptive variants
Values are means over the seeds present; '±' is the standard deviation across
curation seeds (and attack-data seeds where the evaluator reports them).
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from revision_paths import revision_dir, OUTPUT_ROOT  # noqa: E402
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
RES_DIRS = {'nslkdd': 'NSLKDD', 'cic2018': 'CIC-2018'}
NAMES = {'nslkdd': 'NSL-KDD', 'cic2018': 'CSE-CIC-IDS2018'}
TREES = ['xgb', 'catb', 'gbm', 'bagging']
ATTACKS = ['fgsm', 'pgd', 'deepfool', 'cw', 'jsma', 'zoo', 'hsja']
VARIANTS = ['base', 'iat_paper', 'iat', 'iatvol', 'dnnonly', 'tabonly', 'pooled', 'reverse', 'random', 'dtat', 'dtat_o1', 'dtat_o2', 'dtat_o1s', 'dtat_o2s']


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join('' if pd.isna(v) else str(v) for v in r.values) + ' |')
    return '\n'.join(lines)


def e1(res: str) -> str:
    p = os.path.join(revision_dir(RES_DIRS[res]), 'audit_test.csv')
    if not os.path.exists(p):
        return ''
    d = pd.read_csv(p)
    d = d[d.seed_dir == 'seed_0'].copy()
    d['source'] = d.file.str.split('_').str[1]
    d['procedure'] = d.file.str.split('_').str[2]
    keep = ['source', 'procedure', 'viol_any', 'viol_integer', 'viol_immutable', 'viol_capability']
    keep = [c for c in keep if c in d]
    return f"### {NAMES[res]}\n\n" + _md(d[keep].round(2))


def _comp(res, subset, variant, case):
    p = os.path.join(revision_dir(RES_DIRS[res]), 'eval', subset, f"{variant}_{case}_comp.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def _ens(res, subset, variant, case):
    p = os.path.join(revision_dir(RES_DIRS[res]), 'eval', subset, f"{variant}_{case}_ens.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def e2(res: str, subsets: list[str]) -> str:
    out = []
    for subset in subsets:
        rows = []
        for case in ('dnn', 'tabnet'):
            for atk in ATTACKS:
                if atk in ('zoo', 'hsja') and case == 'tabnet':
                    continue
                row = {'case': 'D' if case == 'dnn' else 'T', 'procedure': atk}
                for v in VARIANTS:
                    c = _comp(res, subset, v, case)
                    if c is None:
                        continue
                    t = c[(c.attack == atk) & (c.model.isin(TREES))]
                    if len(t) == 4:
                        row[f"tree {v}"] = round(100 * t.asr_mean.mean(), 2)
                for v in VARIANTS:
                    e = _ens(res, subset, v, case)
                    if e is None:
                        continue
                    t = e[e.attack == atk]
                    if len(t):
                        sd = t.asr_std.iloc[0]
                        row[f"ens {v}"] = (f"{100 * t.asr_mean.iloc[0]:.2f}"
                                           + (f"±{100 * sd:.2f}" if pd.notna(sd) and sd > 0 else ''))
                rows.append(row)
        if rows and len(rows[0]) > 2:
            out.append(f"#### {NAMES[res]} — vectors: `{subset}`\n\n" + _md(pd.DataFrame(rows)))
    return '\n\n'.join(out)


def e34(res: str) -> str:
    files = glob.glob(os.path.join(revision_dir(RES_DIRS[res]), 'constrained', '*.csv'))
    if not files:
        return ''
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d['variant'] = pd.Categorical(d.variant, VARIANTS, ordered=True)
    out = []
    ens = d[d.experiment == 'ens-rs']
    if len(ens):
        g = ens.groupby(['level', 'eps', 'variant'], observed=True).asr.agg(['mean', 'std', 'count']).reset_index()
        g['ASR'] = g.apply(lambda r: f"{r['mean']:.2f}" + (f"±{r['std']:.2f}" if r['count'] > 1 else ''), axis=1)
        t = g.pivot_table(index=['level', 'eps'], columns='variant', values='ASR', aggfunc='first', observed=True)
        out.append("#### Defense-aware ensemble attack (one Omega-valid vector to every branch)\n\n"
                   + _md(t.reset_index()))
    for exp, title in (('cpgd', 'Constrained PGD from the clean network, tree-mean ASR'),):
        x = d[(d.experiment == exp) & (d.target.isin(TREES))]
        x = x[x.source.str.startswith('clean')]
        if len(x):
            per = x.groupby(['level', 'eps', 'source', 'variant', 'seed'], observed=True, dropna=False).asr.mean().reset_index()
            g = per.groupby(['level', 'eps', 'source', 'variant'], observed=True).asr.agg(['mean', 'std', 'count']).reset_index()
            g['ASR'] = g.apply(lambda r: f"{r['mean']:.2f}" + (f"±{r['std']:.2f}" if r['count'] > 1 else ''), axis=1)
            t = g.pivot_table(index=['level', 'eps', 'source'], columns='variant', values='ASR', aggfunc='first', observed=True)
            out.append(f"#### {title}\n\n" + _md(t.reset_index()))
    x = d[(d.experiment == 'cpgd') & (d.target == 'ensemble-per-branch')]
    if len(x):
        g = x.groupby(['level', 'eps', 'source', 'variant'], observed=True).asr.agg(['mean', 'std', 'count']).reset_index()
        g['ASR'] = g.apply(lambda r: f"{r['mean']:.2f}" + (f"±{r['std']:.2f}" if r['count'] > 1 else ''), axis=1)
        t = g.pivot_table(index=['level', 'eps', 'source'], columns='variant', values='ASR', aggfunc='first', observed=True)
        out.append("#### Constrained PGD, per-branch ensemble (Section 3.4 protocol)\n\n" + _md(t.reset_index()))
    x = d[d.experiment == 'comp-rs']
    if len(x):
        g = x.groupby(['level', 'eps', 'target', 'variant'], observed=True).asr.agg(['mean', 'std', 'count']).reset_index()
        g['ASR'] = g.apply(lambda r: f"{r['mean']:.2f}" + (f"±{r['std']:.2f}" if r['count'] > 1 else ''), axis=1)
        t = g.pivot_table(index=['level', 'eps', 'target'], columns='variant', values='ASR', aggfunc='first', observed=True)
        out.append("#### Defense-aware component attack (search on each retrained component)\n\n" + _md(t.reset_index()))
    viol = d['viol_pct'].max() if 'viol_pct' in d else np.nan
    out.append(f"_Max share of emitted vectors violating their constraint level: {viol}%._")
    return '\n\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(OUTPUT_ROOT, 'summary.md'))
    args = ap.parse_args()
    parts = ['# Revision results (auto-generated by cli/hedge/9_revision/summarize_revision.py)']
    for res in RES_DIRS:
        parts.append(f"## {NAMES[res]}")
        parts.append("### E1 Constraint audit of the paper's evaluation vectors (% of records)\n\n" + e1(res))
        parts.append("### E2 Paper protocol on original and Pi_Omega-repaired vectors\n\n"
                     + e2(res, ['test', 'test_omega1', 'test_omega2']))
        parts.append("### E3/E4 Constrained and defense-aware attacks\n\n" + e34(res))
    with open(args.out, 'w') as fh:
        fh.write('\n\n'.join(parts) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
