"""Run the paper's evaluators over vector sets x variants x source cases.

Each cell calls classifier_evaluate_adv.py (component level) or
classifier_parallel_ens_eval.py (per-branch ensemble) exactly as in
notebooks/evaluate.ipynb, only adding --subset/--adv-subset, --models-dir and
report paths. Aggregated tables land in

    resources/<DS>/revision/eval/<subset>/<variant>_<case>_<level>.csv

Cells whose aggregate table exists are skipped, so the script can be re-run as
ablation checkpoints become available.
"""
import os
import sys
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
EVAL_DIR = os.path.join(REPO_ROOT, 'cli', 'hedge', '8_attack_evaluation')
RES_DIRS = {'cic2018': 'CIC-2018', 'nslkdd': 'NSLKDD'}
ATTACKS = ['zoo', 'hsja', 'fgsm', 'jsma', 'pgd', 'deepfool', 'cw']
MODELS = ['xgb', 'catb', 'gbm', 'bagging', 'dnn', 'tabnet']
SEEDS = ['1337', '1338', '1339', '1340']


def variant_flags(res: str, variant: str) -> list[str]:
    if variant == 'base':
        return []
    if variant == 'iat_paper':
        return ['--using-isolated-robust']
    if variant == 'dtat':
        return ['--using-robust', '--model-seeds', *SEEDS]
    models_dir = os.path.join('resources', RES_DIRS[res], 'ablation', variant, 'models')
    return ['--models-dir', models_dir, '--using-robust', '--model-seeds', *SEEDS]


def cells(resources, subsets, variants, cases, levels):
    for res in resources:
        for subset in subsets:
            for variant in variants:
                for case in cases:
                    for level in levels:
                        yield res, subset, variant, case, level


def run_cell(cell, python: str, log_dir: str) -> str:
    res, subset, variant, case, level = cell
    ds = RES_DIRS[res]
    out_dir = os.path.join(REPO_ROOT, 'resources', ds, 'revision', 'eval', subset)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{variant}_{case}_{level}"
    agg = os.path.join(out_dir, f"{stem}.csv")
    if os.path.exists(agg):
        return f"skip {res}/{subset}/{stem}"
    if variant not in ('base', 'iat_paper', 'dtat'):
        need = os.path.join(REPO_ROOT, 'resources', ds, 'ablation', variant, 'models',
                            f"{res}_gbm_robust_seed1340.pkl")
        if not os.path.exists(need):
            return f"not-ready {res}/{subset}/{stem}"
    seed_dirs = [os.path.join('resources', ds, 'adv_samples', subset, f"seed_{k}") for k in (1, 2, 3)]
    common = ['-r', res, '-a', *ATTACKS, '--adv-fallback-src', case, '--seed-dirs', *seed_dirs,
              *variant_flags(res, variant)]
    if level == 'comp':
        cmd = [python, os.path.join(EVAL_DIR, 'classifier_evaluate_adv.py'), '-m', *MODELS, '--subset', subset,
               *common, '--report-out', os.path.join(out_dir, f"{stem}_rows.csv"), '--agg-report-out', agg]
    else:
        cmd = [python, os.path.join(EVAL_DIR, 'classifier_parallel_ens_eval.py'), '--adv-subset', subset,
               *common, '--report-out', os.path.join(out_dir, f"{stem}_rows.csv"), '--agg-report-out', agg]
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [os.path.join(REPO_ROOT, 'src'), env.get('PYTHONPATH')]))
    env.setdefault('OMP_NUM_THREADS', '4')
    log = os.path.join(log_dir, f"eval_{res}_{subset}_{stem}.log")
    with open(log, 'w') as fh:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    ok = rc == 0 and os.path.exists(agg)
    return f"{'OK' if ok else f'FAILED rc={rc}'} {res}/{subset}/{stem}"


def main():
    ap = argparse.ArgumentParser(description="Evaluate vector sets x variants x source cases")
    ap.add_argument('--resource', '-r', nargs='+', default=['nslkdd', 'cic2018'])
    ap.add_argument('--subsets', nargs='+', default=['test_omega1', 'test_omega2'])
    ap.add_argument('--variants', nargs='+', default=['base', 'iat_paper', 'dtat'])
    ap.add_argument('--cases', nargs='+', default=['dnn', 'tabnet'])
    ap.add_argument('--levels', nargs='+', default=['comp', 'ens'])
    ap.add_argument('--parallel', type=int, default=8)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--log-dir', default='/tmp/ablation_logs')
    args = ap.parse_args()
    setup_logging('INFO')
    os.makedirs(args.log_dir, exist_ok=True)
    jobs = list(cells(args.resource, args.subsets, args.variants, args.cases, args.levels))
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for msg in pool.map(lambda c: run_cell(c, args.python, args.log_dir), jobs):
            logger.info(msg)


if __name__ == '__main__':
    main()
