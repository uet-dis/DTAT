"""Generate the data figures of the revised manuscript (300 DPI, PDF + PNG).

All values are read from the measured result files; run make_tables.py first (it
writes the Friedman ranks used by the critical-difference diagram).

  fig_exposure_heatmap   component-level exposure of the clean-trained learners (Table 5 data)
  fig_robustness_curves  ASR versus perturbation radius under Omega_1 / Omega_2
  fig_ablation           in-policy vs out-of-policy tree exposure across curriculum arms
  fig_cd                 Nemenyi critical-difference diagrams (Friedman mean ranks)

Diagram figures (workflow, pathways, curricula) are TikZ code inside main.tex.
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import make_tables as MT  # noqa: E402
from revision_paths import revision_dir, OUTPUT_ROOT  # noqa: E402

REPO_ROOT = MT.REPO_ROOT
PLAIN = {'base': 'Base', 'iat': 'IAT', 'iatvol': 'IAT-vol', 'dnnonly': 'S={DNN}', 'tabonly': 'S={TabNet}',
         'pooled': 'Pooled', 'reverse': 'Reverse', 'random': 'Random', 'dtat': 'DTAT',
         'dtat_o1': 'DTAT-Ω1', 'dtat_o2': 'DTAT-Ω2', 'dtat_o1s': "DTAT-Ω1'", 'dtat_o2s': "DTAT-Ω2'"}
COLORS = {'base': '#7f7f7f', 'iat': '#d62728', 'iatvol': '#ff9896', 'dnnonly': '#9467bd', 'tabonly': '#c5b0d5',
          'pooled': '#8c564b', 'reverse': '#e377c2', 'random': '#bcbd22', 'dtat': '#1f77b4',
          'dtat_o1': '#2ca02c', 'dtat_o2': '#17becf', 'dtat_o1s': '#006d2c', 'dtat_o2s': '#6a3d9a'}
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'font.family': 'DejaVu Sans'})


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_exposure_heatmap(out_dir):
    targets = ['xgb', 'catb', 'gbm', 'bagging', 'dnn', 'tabnet']
    tnames = ['XGB', 'CBT', 'GBM', 'BME', 'DNN', 'TabNet']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True)
    for ax, (res, ds, name) in zip(axes, MT.RES):
        rows, labels = [], []
        xdir = os.path.join(revision_dir(ds), 'xfer')
        for src, cl in (('dnn', 'DNN'), ('tabnet', 'TabNet')):
            for proc in MT.GRAD:
                x = pd.read_csv(os.path.join(xdir, f"{res}_{src}_{proc}.csv")).set_index('target').asr
                rows.append([x.get(t, np.nan) for t in targets])
                labels.append(f"{MT.PROC_NAME[proc].replace(chr(92), '')} ({cl})")
        d = pd.read_csv(os.path.join(revision_dir(ds), 'eval', 'test', "base_dnn_comp.csv"))
        for proc in MT.QUERY:
            x = d[d.attack == proc].set_index('model').asr_mean
            rows.append([100 * x.get(t, np.nan) for t in targets])
            labels.append(f"{MT.PROC_NAME[proc]} (own)")
        m = np.array(rows)
        im = ax.imshow(m, cmap='viridis', vmin=0, vmax=100, aspect='auto')
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=4.6,
                        color='white' if m[i, j] < 60 else 'black')
        ax.set_xticks(range(len(targets)), tnames)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(name)
        ax.axvline(3.5, color='white', lw=1.5)
        ax.axhline(4.5, color='white', lw=0.8)
        ax.axhline(9.5, color='white', lw=0.8)
    fig.colorbar(im, ax=axes, shrink=0.8, label='ASR (%)')
    save(fig, out_dir, 'fig_exposure_heatmap')


def fig_robustness_curves(out_dir, variants=('base', 'iat', 'pooled', 'dtat', 'dtat_o1', 'dtat_o1s', 'dtat_o2s')):
    d = MT._cons()
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.9), sharex=True, constrained_layout=True)
    panels = [('ens-sq', 'Square on ensemble'), ('cpgd', 'cPGD→trees')]
    short = {'nslkdd': 'NSL', 'cic2018': 'CIC'}
    for r, (res, ds, name) in enumerate(MT.RES):
        x = d[d.dataset == res]
        for c, ((exp, title), level) in enumerate([(p, lv) for p in panels for lv in ('omega1', 'omega2')]):
            ax = axes[r, c]
            for v in variants:
                if exp == 'ens-sq':
                    y = x[(x.experiment == 'ens-sq') & (x.level == level) & (x.variant == v)]
                else:
                    y = x[(x.experiment == 'cpgd') & (x.source == 'clean-dnn') & x.target.isin(MT.TREES)
                          & (x.level == level) & (x.variant == v)]
                    y = y.groupby(['eps', 'seed'], dropna=False).asr.mean().reset_index()
                g = y.groupby('eps').asr.agg(['mean', 'std']).reset_index()
                if g.empty:
                    continue
                ax.errorbar(g['eps'], g['mean'], yerr=g['std'].fillna(0), marker='o', ms=3, lw=1, capsize=2,
                            color=COLORS[v], label=PLAIN[v])
            ax.set_title(f"{short[res]}, {title}, {'Ω1' if level == 'omega1' else 'Ω2'}", fontsize=7)
            ax.set_ylim(0, 100 if level == 'omega1' or res == 'cic2018' else 20)
            ax.grid(alpha=0.3)
            if r == 1:
                ax.set_xlabel('ε (× σ)')
            if c == 0:
                ax.set_ylabel('ASR (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(variants), bbox_to_anchor=(0.5, -0.06))
    save(fig, out_dir, 'fig_robustness_curves')


def fig_ablation(out_dir, arms=('iat', 'iatvol', 'dnnonly', 'tabonly', 'pooled', 'reverse', 'random', 'dtat')):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    width = 0.38
    for ax, (res, ds, name) in zip(axes, MT.RES):
        means = {case: [] for case in ('dnn', 'tabnet')}
        errs = {case: [] for case in ('dnn', 'tabnet')}
        present = []
        for v in arms:
            vals = {case: MT.ms(MT.tree_runs(ds, 'test', v, case, MT.GRAD)) for case in ('dnn', 'tabnet')}
            if any(np.isnan(vals[c][0]) for c in vals):
                continue
            present.append(v)
            for case in vals:
                means[case].append(vals[case][0])
                errs[case].append(0 if np.isnan(vals[case][1]) else vals[case][1])
        xs = np.arange(len(present))
        ax.bar(xs - width / 2, means['dnn'], width, yerr=errs['dnn'], capsize=2, label='DNN-generated vectors',
               color='#1f77b4')
        ax.bar(xs + width / 2, means['tabnet'], width, yerr=errs['tabnet'], capsize=2,
               label='TabNet-generated vectors', color='#ff7f0e')
        ax.set_xticks(xs, [PLAIN[v] for v in present], rotation=30, ha='right')
        ax.set_ylabel('Tree-mean ASR (%), 5 gradient procedures')
        ax.set_title(name)
        ax.grid(axis='y', alpha=0.3)
    axes[0].legend(loc='upper right')
    save(fig, out_dir, 'fig_ablation')


def fig_cd(out_dir, tables_dir):
    ranks = pd.read_pickle(os.path.join(tables_dir, 'ranks.pkl'))
    fig, axes = plt.subplots(len(ranks), 1, figsize=(7.2, 1.25 * len(ranks)), constrained_layout=True)
    for ax, (title, (r, cd, n)) in zip(np.atleast_1d(axes), ranks.items()):
        k = len(r)
        ax.set_xlim(0.5, k + 0.5)
        ax.set_ylim(-1.7, 1.0)
        ax.hlines(0, 1, k, color='black', lw=0.8)
        for t in range(1, k + 1):
            ax.vlines(t, -0.06, 0.06, color='black', lw=0.8)
            ax.text(t, 0.15, str(t), ha='center', fontsize=6)
        for i, (v, rank) in enumerate(r.items()):
            y = -0.3 - 0.2 * (i % 6)
            ax.plot([rank, rank], [0, y], color=COLORS.get(v, 'k'), lw=0.8)
            ax.plot(rank, 0, 'o', color=COLORS.get(v, 'k'), ms=3)
            ax.text(rank, y - 0.02, PLAIN.get(v, v), ha='center', va='top', fontsize=6, color=COLORS.get(v, 'k'))
        ax.plot([1, 1 + cd], [0.55, 0.55], color='black', lw=2)
        ax.text(1 + cd / 2, 0.68, f"CD = {cd:.2f}", ha='center', fontsize=6)
        ax.set_title(f"{title} (N = {n} conditions; lower rank = lower ASR)", fontsize=7)
        ax.axis('off')
    save(fig, out_dir, 'fig_cd')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.join(OUTPUT_ROOT, 'figures'))
    ap.add_argument('--tables-dir', default=os.path.join(OUTPUT_ROOT, 'tables'))
    args = ap.parse_args()
    fig_exposure_heatmap(args.out_dir)
    fig_robustness_curves(args.out_dir)
    fig_ablation(args.out_dir)
    fig_cd(args.out_dir, args.tables_dir)
    print(sorted(glob.glob(os.path.join(args.out_dir, '*.pdf'))))


if __name__ == '__main__':
    main()
