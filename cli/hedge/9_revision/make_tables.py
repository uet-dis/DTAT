"""Write the LaTeX result tables of the revised manuscript from the measured CSVs.

Every number in paper/revision/tables/*.tex is computed here from
resources/<DS>/revision/{eval,constrained}/ and the training logs; nothing is typed by hand.
Mean +- standard deviation is taken over all runs of an entry (curation seeds x
attack-data seeds where both apply); single runs are printed without +-.
"""
import os
import re
import sys
import glob
import argparse

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import stats_revision as SR  # noqa: E402
from revision_paths import revision_dir, OUTPUT_ROOT  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
RES = [('nslkdd', 'NSLKDD', 'NSL-KDD'), ('cic2018', 'CIC-2018', 'CSE-CIC-IDS2018')]
TREES = ['xgb', 'catb', 'gbm', 'bagging']
GRAD = ['fgsm', 'pgd', 'deepfool', 'cw', 'jsma']
QUERY = ['zoo', 'hsja']
PROC_NAME = {'fgsm': 'FGSM', 'pgd': 'PGD', 'deepfool': 'DeepFool', 'cw': 'C\\&W', 'jsma': 'JSMA',
             'zoo': 'ZOO', 'hsja': 'HSJA'}
NAME = {'base': 'Base', 'iat': 'IAT', 'iatvol': 'IAT-vol', 'dnnonly': '$\\mathcal S{=}\\{L_1\\}$',
        'tabonly': '$\\mathcal S{=}\\{L_2\\}$', 'pooled': 'Pooled', 'reverse': 'Reverse', 'random': 'Random',
        'dtat': '\\DTAT', 'dtat_o1': '\\DTAT-$\\Omega_1$', 'dtat_o2': '\\DTAT-$\\Omega_2$',
        'dtat_o1s': '\\DTAT-$\\Omega_1^{\\prime}$', 'dtat_o2s': '\\DTAT-$\\Omega_2^{\\prime}$'}
ARMS = ['iat', 'iatvol', 'dnnonly', 'tabonly', 'pooled', 'reverse', 'random', 'dtat', 'dtat_o1', 'dtat_o2']
OMEGA_ARMS = ['dtat_o1', 'dtat_o2', 'dtat_o1s', 'dtat_o2s']
ALL_VARIANTS = ['base'] + ARMS + ['dtat_o1s', 'dtat_o2s']
# Columns shown in the printed tables (all variants remain in the statistics).
ABL_ARMS = ['iat', 'iatvol', 'dnnonly', 'tabonly', 'pooled', 'reverse', 'random', 'dtat']
POSTHOC_COLS = ('base', 'iat', 'pooled', 'dtat', 'dtat_o1', 'dtat_o2', 'dtat_o1s')
CONSTRAINED_COLS = ('base', 'iat', 'pooled', 'dtat', 'dtat_o1', 'dtat_o2', 'dtat_o1s')


def f(m, s=None, n=1, bold=False):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return '--'
    txt = f"{m:.2f}" + (f"$\\pm${s:.2f}" if n > 1 and s is not None and not np.isnan(s) else '')
    return f"\\textbf{{{txt}}}" if bold else txt


def rows_path(ds, subset, variant, case, level):
    return os.path.join(revision_dir(ds), 'eval', subset, f"{variant}_{case}_{level}_rows.csv")


def tree_runs(ds, subset, variant, case, attacks):
    """Tree-mean ASR per run (model seed x attack-data seed), averaged over `attacks`."""
    p = rows_path(ds, subset, variant, case, 'comp')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d.model.isin(TREES) & d.attack.isin(attacks)]
    d['model_seed'] = d['model_seed'].fillna(0)
    per = d.groupby(['attack', 'model_seed', 'seed']).asr.mean().reset_index()   # mean over 4 trees
    per = per.groupby(['model_seed', 'seed']).asr.mean() if len(attacks) > 1 else per.set_index(['model_seed', 'seed']).asr
    return 100 * per.values


def ens_runs(ds, subset, variant, case, attacks):
    p = rows_path(ds, subset, variant, case, 'ens')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d.attack.isin(attacks)]
    if 'model_seed' not in d:
        d['model_seed'] = 0
    d['model_seed'] = d['model_seed'].fillna(0)
    per = d.groupby(['model_seed', 'seed']).asr.mean()
    return 100 * per.values


def per_seed(ds, subset, variant, case, attacks, level):
    """ASR per curation seed: mean over trees (if level='comp'), procedures and attack-data seeds."""
    p = rows_path(ds, subset, variant, case, level)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    if level == 'comp':
        d = d[d.model.isin(TREES)]
    d = d[d.attack.isin(attacks)]
    if 'model_seed' not in d:
        d['model_seed'] = 0
    d['model_seed'] = d['model_seed'].fillna(0)
    per = d.groupby(['attack', 'model_seed', 'seed']).asr.mean().reset_index()
    return 100 * per.groupby('model_seed').asr.mean()


def ms(v):
    if v is None or len(v) == 0:
        return (np.nan, np.nan, 0)
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else np.nan, len(v))


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(text)


# --------------------------------------------------------------------------- #
def tab_audit():
    lines = []
    for res, ds, name in RES:
        d = pd.read_csv(os.path.join(revision_dir(ds), 'audit_test.csv'))
        d = d[d.seed_dir == 'seed_0'].copy()
        d['proc'] = d.file.str.split('_').str[2]
        d['src'] = d.file.str.split('_').str[1]
        lines.append(f"\\multicolumn{{6}}{{c}}{{\\emph{{{name}}}}}\\\\\n\\midrule")
        for proc in GRAD + QUERY:
            x = d[d.proc == proc]
            def rng(c):
                lo, hi = f"{x[c].min():.1f}", f"{x[c].max():.1f}"
                return lo if lo == hi else f"{lo}--{hi}"
            lines.append(f"{PROC_NAME[proc]} & {len(x)} & {rng('viol_any')} & {rng('viol_integer')} & "
                         f"{rng('viol_immutable')} & {rng('viol_capability')}\\\\")
        lines.append('\\midrule')
    body = '\n'.join(lines[:-1])
    return ("\\begin{tabular}{@{}lcrrrr@{}}\n\\toprule\nProcedure & Sources & $\\Omega_1$ any & Integrality & "
            "Immutable & $\\Omega_2$ capability\\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


def tab_tree(subset='test', variants=('base', 'iat', 'iatvol', 'dtat')):
    head = ' & '.join(NAME[v] for v in variants)
    k = len(variants)
    out = [f"\\begin{{tabular}}{{@{{}}ll{'r' * k}@{{}}}}", '\\toprule',
           f"Case & Procedure & {head}\\\\", '\\midrule']
    for res, ds, name in RES:
        out.append(f"\\multicolumn{{{k + 2}}}{{c}}{{\\emph{{{name}}}}}\\\\\n\\midrule")
        for case, cl in (('dnn', 'D'), ('tabnet', 'T')):
            procs = GRAD + (QUERY if case == 'dnn' else [])
            for proc in procs:
                vals = [ms(tree_runs(ds, subset, v, case, [proc])) for v in variants]
                best = np.nanmin([m for m, _, _ in vals])
                cells = [f(m, s, n, bold=abs(m - best) < 5e-3) for m, s, n in vals]
                label = cl if proc in GRAD else 'D/T'
                out.append(f"{label} & {PROC_NAME[proc]} & " + ' & '.join(cells) + '\\\\')
        out.append('\\midrule')
    out[-1] = '\\bottomrule'
    out.append('\\end{tabular}')
    return '\n'.join(out) + '\n'


def tab_ens(subset='test', variants=('base', 'iat', 'iatvol', 'dtat')):
    head = ' & '.join(NAME[v] for v in variants)
    k = len(variants)
    out = [f"\\begin{{tabular}}{{@{{}}ll{'r' * k}r@{{}}}}", '\\toprule',
           f"Case & Procedure & {head} & $\\Delta$\\\\", '\\midrule']
    deltas = []
    for res, ds, name in RES:
        out.append(f"\\multicolumn{{{k + 3}}}{{c}}{{\\emph{{{name}}}}}\\\\\n\\midrule")
        for case, cl in (('dnn', 'D'), ('tabnet', 'T')):
            procs = GRAD + (QUERY if case == 'dnn' else [])
            for proc in procs:
                vals = [ms(ens_runs(ds, subset, v, case, [proc])) for v in variants]
                best = np.nanmin([m for m, _, _ in vals])
                cells = [f(m, s, n, bold=abs(m - best) < 5e-3) for m, s, n in vals]
                d = vals[variants.index('iat')][0] - vals[variants.index('dtat')][0]
                deltas.append(d)
                label = cl if proc in GRAD else 'D/T'
                out.append(f"{label} & {PROC_NAME[proc]} & " + ' & '.join(cells) + f" & {d:.2f}\\\\")
        out.append('\\midrule')
    out[-1] = '\\bottomrule'
    out.append('\\end{tabular}')
    return '\n'.join(out) + '\n', deltas


def tab_original(subset='test', variants=('base', 'iat', 'iatvol', 'dtat')):
    """Tree-mean and per-branch ensemble ASR on the original vectors (one table)."""
    k = len(variants)
    head = ' & '.join(NAME[v] for v in variants)
    out = [f"\\begin{{tabular}}{{@{{}}ll{'r' * k}{'r' * k}r@{{}}}}", '\\toprule',
           f" & & \\multicolumn{{{k}}}{{c}}{{Tree-mean ASR}} & \\multicolumn{{{k + 1}}}{{c}}{{Per-branch ensemble ASR}}\\\\",
           f"\\cmidrule(lr){{3-{k + 2}}}\\cmidrule(l){{{k + 3}-{2 * k + 3}}}",
           f"Case & Procedure & {head} & {head} & $\\Delta$\\\\", '\\midrule']
    deltas = []
    for res, ds, name in RES:
        out.append(f"\\multicolumn{{{2 * k + 3}}}{{c}}{{\\emph{{{name}}}}}\\\\\n\\midrule")
        for case, cl in (('dnn', 'D'), ('tabnet', 'T')):
            for proc in GRAD + (QUERY if case == 'dnn' else []):
                cells = []
                for fn in (tree_runs, ens_runs):
                    vals = [ms(fn(ds, subset, v, case, [proc])) for v in variants]
                    best = np.nanmin([m for m, _, _ in vals])
                    cells += [f(m, s, n, bold=abs(m - best) < 5e-3) for m, s, n in vals]
                    if fn is ens_runs:
                        d = vals[variants.index('iat')][0] - vals[variants.index('dtat')][0]
                        deltas.append(d)
                label = cl if proc in GRAD else 'D/T'
                out.append(f"{label} & {PROC_NAME[proc]} & " + ' & '.join(cells) + f" & {d:.2f}\\\\")
        out.append('\\midrule')
    out[-1] = '\\bottomrule'
    out.append('\\end{tabular}')
    return '\n'.join(out) + '\n', deltas


def _clean_acc(ds, v):
    p = rows_path(ds, 'test', v, 'dnn', 'ens')
    if not os.path.exists(p):
        return (np.nan, np.nan, 0)
    d = pd.read_csv(p)
    if 'model_seed' not in d:
        d['model_seed'] = 0
    return ms(100 * d.groupby(d['model_seed'].fillna(0)).plain_acc.first().values)


def tab_ablation(subset='test'):
    """Tree-mean and per-branch ensemble ASR averaged over the five gradient procedures."""
    head = ' & '.join(NAME[v] for v in ABL_ARMS)
    out = [f"\\begin{{tabular}}{{@{{}}lll{'r' * len(ABL_ARMS)}@{{}}}}", '\\toprule',
           f"Dataset & Level & Case & {head}\\\\", '\\midrule']
    for res, ds, name in RES:
        for level, fn in (('Trees', tree_runs), ('Ensemble', ens_runs)):
            for case, cl in (('dnn', 'D'), ('tabnet', 'T')):
                vals = [ms(fn(ds, subset, v, case, GRAD)) for v in ABL_ARMS]
                best = np.nanmin([m for m, _, _ in vals])
                cells = [f(m, s, n, bold=abs(m - best) < 5e-3) for m, s, n in vals]
                out.append(f"{name} & {level} & {cl} & " + ' & '.join(cells) + '\\\\')
            # query procedures do not depend on the case
            vals = [ms(fn(ds, subset, v, 'dnn', QUERY)) for v in ABL_ARMS]
            best = np.nanmin([m for m, _, _ in vals])
            cells = [f(m, s, n, bold=abs(m - best) < 5e-3) for m, s, n in vals]
            out.append(f"{name} & {level} & Q & " + ' & '.join(cells) + '\\\\')
        cells = [f(*_clean_acc(ds, v)) for v in ABL_ARMS]
        out.append(f"{name} & Clean acc. & -- & " + ' & '.join(cells) + '\\\\')
        out.append('\\midrule')
    out[-1] = '\\bottomrule'
    out.append('\\end{tabular}')
    return '\n'.join(out) + '\n'


def _cons():
    frames = []
    for res, ds, name in RES:
        files = glob.glob(os.path.join(revision_dir(ds), 'constrained', '*.csv'))
        if files:
            d = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
            d['dsname'] = name
            if 'attack_seed' not in d:
                d['attack_seed'] = 0
            d['attack_seed'] = d['attack_seed'].fillna(0)
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def tab_constrained(variants=CONSTRAINED_COLS):
    d = _cons()
    head = ' & '.join(NAME[v] for v in variants)
    k = len(variants)
    out = [f"\\begin{{tabular}}{{@{{}}lll{'r' * k}@{{}}}}", '\\toprule',
           f"Dataset & $\\Omega$ & Measurement & {head}\\\\", '\\midrule']
    for res, ds, name in RES:
        x = d[d.dataset == res]
        for level, lv in (('omega1', '$\\Omega_1$'), ('omega2', '$\\Omega_2$')):
            specs = [
                ('cPGD$\\to$trees (D)', lambda v: x[(x.experiment == 'cpgd') & (x.source == 'clean-dnn') &
                                                 x.target.isin(TREES) & (x.level == level) & (x.eps == 0.5) &
                                                 (x.variant == v)].groupby(['seed', 'attack_seed'], dropna=False).asr.mean()),
                ('cPGD$\\to$trees (T)', lambda v: x[(x.experiment == 'cpgd') & (x.source == 'clean-tabnet') &
                                                 x.target.isin(TREES) & (x.level == level) & (x.eps == 0.5) &
                                                 (x.variant == v)].groupby(['seed', 'attack_seed'], dropna=False).asr.mean()),
                ('cPGD per-branch ens.', lambda v: x[(x.experiment == 'cpgd') & x.source.str.startswith('clean') &
                                                   (x.target == 'ensemble-per-branch') & (x.level == level) &
                                                   (x.eps == 0.5) & (x.variant == v)].groupby(['seed', 'attack_seed'], dropna=False).asr.mean()),
                ('Adaptive cPGD ens.', lambda v: x[(x.experiment == 'cpgd') & x.source.str.startswith('variant') &
                                                 (x.target == 'ensemble-per-branch') & (x.level == level) &
                                                 (x.eps == 0.5) & (x.variant == v)].groupby(['seed', 'attack_seed'], dropna=False).asr.mean()),
                ('Adaptive ensemble RS', lambda v: x[(x.experiment == 'ens-rs') & (x.level == level) & (x.eps == 0.5) &
                                                   (x.variant == v)].groupby(['seed', 'attack_seed'], dropna=False).asr.mean()),
                ('Adaptive tree RS', lambda v: x[(x.experiment == 'comp-rs') & x.target.isin(TREES) & (x.level == level) &
                                               (x.eps == 0.5) & (x.variant == v)].groupby(['seed', 'attack_seed'], dropna=False).asr.mean()),
                ('Adaptive ensemble Square', lambda v: x[(x.experiment == 'ens-sq') & (x.level == level) & (x.eps == 0.5) &
                                                       (x.variant == v)].groupby(['seed', 'attack_seed'], dropna=False).asr.mean()),
                ('Adaptive tree Square', lambda v: x[(x.experiment == 'comp-sq') & x.target.isin(TREES) & (x.level == level) &
                                                   (x.eps == 0.5) & (x.variant == v)].groupby(['seed', 'attack_seed'], dropna=False).asr.mean()),
            ]
            for label, sel in specs:
                vals = [ms(sel(v).values) for v in variants]
                ok = [m for m, _, n in vals if n > 0]
                best = np.nanmin(ok) if ok else np.nan
                cells = [f(m, s, n, bold=n > 0 and abs(m - best) < 5e-3) for m, s, n in vals]
                out.append(f"{name} & {lv} & {label} & " + ' & '.join(cells) + '\\\\')
        out.append('\\midrule')
    out[-1] = '\\bottomrule'
    out.append('\\end{tabular}')
    return '\n'.join(out) + '\n'


def tab_posthoc(variants=POSTHOC_COLS):
    head = ' & '.join(NAME[v] for v in variants)
    k = len(variants)
    out = [f"\\begin{{tabular}}{{@{{}}lll{'r' * k}@{{}}}}", '\\toprule',
           f"Dataset & Vectors & Level & {head}\\\\", '\\midrule']
    for res, ds, name in RES:
        for subset, sl in (('test', 'original'), ('test_omega1', '$\\Pi_{\\Omega_1}$'), ('test_omega2', '$\\Pi_{\\Omega_2}$')):
            for level, fn in (('Trees', tree_runs), ('Ensemble', ens_runs)):
                runs = []
                lv = 'comp' if fn is tree_runs else 'ens'
                for v in variants:
                    a = [per_seed(ds, subset, v, c, GRAD, lv) for c in ('dnn', 'tabnet')]
                    # the two source cases are fixed conditions, not replicates: average them per seed
                    runs.append(ms(pd.concat(a, axis=1).mean(axis=1).values) if all(x is not None for x in a)
                                else (np.nan, np.nan, 0))
                best = np.nanmin([m for m, _, _ in runs])
                cells = [f(m, s, n, bold=abs(m - best) < 5e-3) for m, s, n in runs]
                out.append(f"{name} & {sl} & {level} & " + ' & '.join(cells) + '\\\\')
        out.append('\\midrule')
    out[-1] = '\\bottomrule'
    out.append('\\end{tabular}')
    return '\n'.join(out) + '\n'


def tab_clean(variants=tuple(ALL_VARIANTS)):
    out = [f"\\begin{{tabular}}{{@{{}}l{'r' * len(variants)}@{{}}}}", '\\toprule',
           'Dataset & ' + ' & '.join(NAME[v] for v in variants) + '\\\\', '\\midrule']
    for res, ds, name in RES:
        cells = []
        for v in variants:
            p = rows_path(ds, 'test', v, 'dnn', 'ens')
            if not os.path.exists(p):
                cells.append('--')
                continue
            d = pd.read_csv(p)
            if 'model_seed' not in d:
                d['model_seed'] = 0
            a = 100 * d.groupby(d['model_seed'].fillna(0)).plain_acc.first().values
            cells.append(f(*ms(a)))
        out.append(f"{name} & " + ' & '.join(cells) + '\\\\')
    out += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(out) + '\n'


def tab_dbal():
    out = ['\\begin{tabular}{@{}llrrr@{}}', '\\toprule',
           'Dataset & Records & $n$ & $\\Omega_1$-violating (\\%) & after strict $\\Pi_{\\Omega_1}$ (\\%)\\\\', '\\midrule']
    for res, ds, name in RES:
        d = pd.read_csv(os.path.join(revision_dir(ds), 'audit_dbal.csv'))
        d['kind'] = np.where(d.source.str.endswith(':wgan'), 'WGAN', 'real')
        d['cls'] = d.source.str.split(':').str[1]
        for _, r in d.sort_values(['kind', 'cls']).iterrows():
            lab = f"{r.cls} ({r.kind})".replace('_', '\\_')
            out.append(f"{name} & {lab} & {int(r.n):,} & {r.viol_before_pct:.1f} & {r.viol_after_pct:.1f}\\\\")
        out.append('\\midrule')
    out[-1] = '\\bottomrule'
    out.append('\\end{tabular}')
    return '\n'.join(out) + '\n'


def tab_dbal_summary():
    out = ['\\begin{tabular}{@{}llrrr@{}}', '\\toprule',
           'Dataset & Records of $\\Data_{\\mathrm{bal}}$ & Classes & $\\Omega_1$-violating & after $\\Pi^{\\star}_{\\Omega_1}$\\\\',
           '\\midrule']
    for res, ds, name in RES:
        d = pd.read_csv(os.path.join(revision_dir(ds), 'audit_dbal.csv'))
        syn = d.source.str.endswith(':wgan')
        for lab, g in (('real (K-means selected)', d[~syn]), ('synthetic (WGAN)', d[syn])):
            lo, hi = g.viol_before_pct.min(), g.viol_before_pct.max()
            rng = f"{lo:.1f}" if f"{lo:.1f}" == f"{hi:.1f}" else f"{lo:.1f}--{hi:.1f}"
            out.append(f"{name} & {lab} & {len(g)} & {rng} & {g.viol_after_pct.max():.1f}\\\\")
    out += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(out) + '\n'


def tab_runtime():
    """Fitting time per checkpoint, from fit_times.csv (written from the training logs)."""
    frames = [pd.read_csv(p) for _, ds, _ in RES
              for p in [os.path.join(revision_dir(ds), 'fit_times.csv')] if os.path.exists(p)]
    d = pd.concat(frames, ignore_index=True).rename(columns={'minutes': 'min'}) if frames else pd.DataFrame()
    if d.empty:
        return ''
    order = ['xgb', 'catb', 'gbm', 'bagging', 'dnn', 'tabnet']
    names = {'xgb': 'XGB (GPU)', 'catb': 'CBT (GPU)', 'gbm': 'GBM (1 CPU core)', 'bagging': 'BME (CPU)',
             'dnn': 'DNN (GPU)', 'tabnet': 'TabNet (GPU)'}
    out = ['\\begin{tabular}{@{}lrrrr@{}}', '\\toprule',
           'Learner & \\multicolumn{2}{c}{NSL-KDD} & \\multicolumn{2}{c}{CSE-CIC-IDS2018}\\\\',
           '\\cmidrule(lr){2-3}\\cmidrule(l){4-5}', ' & median & fits & median & fits\\\\', '\\midrule']
    for m in order:
        cells = []
        for res in ('nslkdd', 'cic2018'):
            x = d[(d.res == res) & (d.model == m)]['min']
            cells += [f"{x.median():.1f}" if len(x) else '--', str(len(x))]
        out.append(f"{names[m]} & " + ' & '.join(cells) + '\\\\')
    out += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(out) + '\n'


def stats_tables():
    """Friedman/Nemenyi and Wilcoxon over evaluation blocks, retrained variants only."""
    keep = set(ALL_VARIANTS)
    sets = [('Trees, paper protocol', SR.paper_protocol_blocks('comp')),
            ('Ensemble, paper protocol', SR.paper_protocol_blocks('ens')),
            ('Constrained/adaptive', SR.constrained_blocks())]
    fr_lines, wx_lines, ranks_all, fr_caption = [], [], {}, []
    for title, df in sets:
        df = df[df.variant.isin(keep)]
        wide = df.pivot_table(index='block', columns='variant', values='asr', aggfunc='mean')
        wide = wide.dropna(axis=1, thresh=int(0.9 * len(wide))).dropna(axis=0)
        from scipy import stats
        chi2, p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
        k, n = wide.shape[1], len(wide)
        cd = SR.Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6.0 * n))
        ranks = wide.rank(axis=1).mean().sort_values()
        ranks_all[title] = (ranks, cd, n)
        mant, expo = f"{p:.1e}".split('e')
        fr_caption.append(f"{title.lower()}: $N={n}$, $k={k}$, $\\chi^2_F={chi2:.1f}$, "
                          f"$p={mant}\\times10^{{{int(expo)}}}$, CD $={cd:.2f}$")
        fr_lines.append(f"{title} & {n} & {k} & {chi2:.1f} & {p:.1e} & {cd:.2f} & "
                        + ', '.join(f"{NAME[v]} ({r:.2f})" for v, r in ranks.items()) + '\\\\')
        res = []
        for v in wide.columns:
            if v == 'dtat':
                continue
            diff = wide[v] - wide['dtat']
            if np.allclose(diff, 0):
                continue
            _, pv = stats.wilcoxon(wide[v], wide['dtat'])
            res.append([v, int((diff > 0).sum()), int((diff < 0).sum()), float(diff.median()), pv])
        ps = np.array([r[4] for r in res])
        order = np.argsort(ps)
        holm, run = np.empty_like(ps), 0.0
        for i, j in enumerate(order):
            run = max(run, min(1.0, (len(ps) - i) * ps[j]))
            holm[j] = run
        for r, h in zip(res, holm):
            wx_lines.append(f"{title} & {NAME[r[0]]} & {r[1]}/{r[2]} & {r[3]:.2f} & {h:.1e}\\\\")
        wx_lines.append('\\midrule')
    fr = ('\\begin{tabular}{@{}lrrrrrp{8.4cm}@{}}\n\\toprule\nBlock set & $N$ & $k$ & $\\chi^2_F$ & $p$ & CD & '
          'Mean rank (1 = lowest ASR)\\\\\n\\midrule\n' + '\n'.join(fr_lines) + '\n\\bottomrule\n\\end{tabular}\n')
    wx_lines[-1] = '\\bottomrule'
    wx = ('\\begin{tabular}{@{}llrrr@{}}\n\\toprule\nBlock set & Variant & $>$/$<$ \\DTAT & Median diff. (pp) & '
          '$p_{\\mathrm{Holm}}$\\\\\n\\midrule\n' + '\n'.join(wx_lines) + '\n\\end{tabular}\n')
    return fr, wx, ranks_all, '; '.join(fr_caption)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.join(OUTPUT_ROOT, 'tables'))
    args = ap.parse_args()
    o = args.out_dir
    write(os.path.join(o, 'audit.tex'), tab_audit())
    write(os.path.join(o, 'dbal_summary.tex'), tab_dbal_summary())
    orig, _ = tab_original()
    write(os.path.join(o, 'original.tex'), orig)
    write(os.path.join(o, 'tree_transfer.tex'), tab_tree())
    ens, deltas = tab_ens()
    write(os.path.join(o, 'ensemble.tex'), ens)
    write(os.path.join(o, 'ablation.tex'), tab_ablation())
    write(os.path.join(o, 'posthoc.tex'), tab_posthoc())
    write(os.path.join(o, 'constrained.tex'), tab_constrained())
    write(os.path.join(o, 'clean.tex'), tab_clean())
    write(os.path.join(o, 'runtime.tex'), tab_runtime())
    write(os.path.join(o, 'dbal.tex'), tab_dbal())
    fr, wx, ranks, fr_cap = stats_tables()
    write(os.path.join(o, 'friedman_caption.tex'), fr_cap + '.')
    write(os.path.join(o, 'friedman.tex'), fr)
    write(os.path.join(o, 'wilcoxon.tex'), wx)
    pd.to_pickle(ranks, os.path.join(o, 'ranks.pkl'))
    print(f"Delta(IAT-DTAT) range on original vectors: {min(deltas):.2f} .. {max(deltas):.2f}")


if __name__ == '__main__':
    main()
