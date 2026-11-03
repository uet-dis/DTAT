"""Run constrained_attack.py over variants x seeds x constraint levels x budgets.

One CSV per job in resources/<DS>/revision/constrained/; existing files are skipped
and variants whose checkpoints are not ready yet are reported and left for a re-run.
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
ATTACK_PY = os.path.join(SCRIPT_DIR, 'constrained_attack.py')
RES_DIRS = {'cic2018': 'CIC-2018', 'nslkdd': 'NSLKDD'}
SEEDS = [1337, 1338, 1339, 1340]


def variant_seeds(variant: str) -> list:
    return [None] if variant in ('base', 'iat_paper') else SEEDS


def ready(res: str, variant: str, seed) -> bool:
    if variant in ('base', 'iat_paper', 'dtat'):
        return True
    p = os.path.join(REPO_ROOT, 'resources', RES_DIRS[res], 'ablation', variant, 'models',
                     f"{res}_gbm_robust_seed{seed}.pkl")
    return os.path.exists(p)


def jobs(resources, variants, levels, plan, attack_seeds, extra_seed_eps):
    for res in resources:
        for exp, eps_list, extra in plan:
            for variant in variants:
                if exp.endswith('adaptive') and variant == 'base':
                    continue
                for seed in variant_seeds(variant):
                    for level in levels:
                        for eps in eps_list:
                            for e in extra:
                                for k in attack_seeds:
                                    # additional attack seeds only at the main radius
                                    if k > 0 and eps not in extra_seed_eps:
                                        continue
                                    if exp == 'comp-sq' and k > 0:
                                        continue
                                    yield res, exp, variant, seed, level, eps, e, k


def run(job, python: str, log_dir: str, per_class: int, device: str) -> str:
    res, exp, variant, seed, level, eps, extra, k = job
    # attack seed k also selects the evaluation subsample; k = 0 keeps the original file names
    tag = f"{exp}_{extra or 'x'}_{variant}_{seed}_{level}_eps{eps}" + (f"_as{k}" if k else '')
    out_dir = os.path.join(REPO_ROOT, 'resources', RES_DIRS[res], 'revision', 'constrained')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{tag}.csv")
    if os.path.exists(out):
        return f"skip {res}/{tag}"
    if not ready(res, variant, seed):
        return f"not-ready {res}/{tag}"
    experiment = 'cpgd' if exp.startswith('cpgd') else exp
    cmd = [python, ATTACK_PY, '-r', res, '-e', experiment, '--variant', variant, '--level', level,
           '--eps', str(eps), '--per-class', str(per_class), '--device', device, '--out', out,
           '--attack-seed', str(k), '--sample-seed', str(k)]
    if seed is not None:
        cmd += ['--seed', str(seed)]
    if experiment == 'cpgd':
        cmd += ['--pgd-source', extra, '--iters', '100']
        if exp.endswith('adaptive'):
            cmd += ['--pgd-source-variant']
    elif experiment in ('ens-sq', 'comp-sq'):
        cmd += ['--iters', '1000']
    else:
        cmd += ['--iters', '300']
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [os.path.join(REPO_ROOT, 'src'), env.get('PYTHONPATH')]))
    env['OMP_NUM_THREADS'] = '2'
    log = os.path.join(log_dir, f"cons_{res}_{tag}.log")
    with open(log, 'w') as fh:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    return f"{'OK' if rc == 0 and os.path.exists(out) else f'FAILED rc={rc}'} {res}/{tag}"


def main():
    ap = argparse.ArgumentParser(description="Batch constrained / adaptive evaluation")
    ap.add_argument('--resource', '-r', nargs='+', default=['nslkdd', 'cic2018'])
    ap.add_argument('--variants', nargs='+',
                    default=['base', 'iat', 'iatvol', 'dnnonly', 'tabonly', 'pooled', 'reverse', 'random', 'dtat',
                             'dtat_o1', 'dtat_o2', 'dtat_o1s', 'dtat_o2s'])
    ap.add_argument('--levels', nargs='+', default=['omega1', 'omega2'])
    ap.add_argument('--experiments', nargs='+', default=['ens-rs', 'cpgd', 'cpgd-adaptive', 'comp-rs', 'ens-sq', 'comp-sq'])
    ap.add_argument('--attack-seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--extra-seed-eps', type=float, nargs='+', default=[0.5])
    ap.add_argument('--per-class', type=int, default=500)
    ap.add_argument('--parallel', type=int, default=16)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--log-dir', default='/tmp/ablation_logs/constrained')
    args = ap.parse_args()
    setup_logging('INFO')
    os.makedirs(args.log_dir, exist_ok=True)
    full = {
        'ens-rs': ('ens-rs', [0.2, 0.5, 1.0], [None]),
        'cpgd': ('cpgd', [0.2, 0.5, 1.0], ['dnn', 'tabnet']),
        'cpgd-adaptive': ('cpgd-adaptive', [0.5], ['dnn', 'tabnet']),
        'comp-rs': ('comp-rs', [0.5], [None]),
        'ens-sq': ('ens-sq', [0.2, 0.5, 1.0], [None]),
        'comp-sq': ('comp-sq', [0.5], [None]),
    }
    plan = [full[e] for e in args.experiments]
    todo = list(jobs(args.resource, args.variants, args.levels, plan, args.attack_seeds, args.extra_seed_eps))
    logger.info(f"[+] {len(todo)} jobs")
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for msg in pool.map(lambda j: run(j, args.python, args.log_dir, args.per_class, args.device), todo):
            logger.info(msg)


if __name__ == '__main__':
    main()
