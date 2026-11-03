"""Build curated curricula for the revision ablation arms.

Every arm reuses the source-indexed TRAIN pools in resources/<DS>/adv_samples/train
and the existing merge/curation code (merge_training_samples.py, postprocess_merge.py);
only the admitted pools and the per-tag quotas change. Outputs never touch the
original robust/isolated folders:

    resources/<DS>/ablation/<arm>/pools/<res>_<model>_robust_train.csv          (merged, pre-curation)
    resources/<DS>/ablation/<arm>/seed_<s>/<res>_<model>_robust_train_post.csv  (curated curriculum)

Arms (tau = per-procedure quota: 12000 for cic2018, 5000 for nslkdd):
    dtat     reproduction check of the paper's DTAT trees (own ZOO/HSJA tau each,
             DNN and TabNet gradient pools tau/2 each)
    iat      trees: own ZOO/HSJA, tau each (the paper's IAT, now with 4 curation seeds)
    iatvol   trees: own ZOO/HSJA, 3.5*tau each -> same volume (7*tau) as DTAT
    dnnonly  trees: own ZOO/HSJA tau each + DNN gradient pools tau each (S_k = {DNN})
    tabonly  trees: own ZOO/HSJA tau each + TabNet gradient pools tau each (S_k = {TabNet})
    pooled   direction-blind: every target (trees and networks) gets, per procedure, the
             pools of every source that supplies it, quota split equally across sources
    dtat_o1  the DTAT policy (trees and networks) on pools repaired with Pi_Omega1
    dtat_o2  the DTAT policy (trees and networks) on pools repaired with Pi_Omega2
    reverse  direction reversed: trees get only their own ZOO/HSJA pools; each network
             gets its own pools plus the trees' ZOO/HSJA pools (query quota split
             equally between itself and the four trees)
    dtat_o1s / dtat_o2s  the DTAT policy on pools repaired strictly with Pi_Omega against the
             strictly repaired training set D_bal' (data/<res>_merged_train_raw_processed_omega1s.csv)
    random   random pathways: for every target and every procedure, one source is drawn
             uniformly (per curation seed) among all sources that supply the procedure
             and supplies the whole quota tau
"""
import os
import sys
import random
import argparse
import subprocess

from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
GEN_DIR = os.path.join(REPO_ROOT, 'cli', 'hedge', '7_generating_samples')

RES_DIRS = {'cic2018': 'CIC-2018', 'nslkdd': 'NSLKDD'}
TAU = {'cic2018': 12000, 'nslkdd': 5000}
TREES = ['xgb', 'catb', 'gbm', 'bagging']
NETS = ['dnn', 'tabnet']
GRAD = ['fgsm', 'pgd', 'deepfool', 'cw', 'jsma']
QUERY = ['zoo', 'hsja']
ARMS = ['dtat', 'iat', 'iatvol', 'dnnonly', 'tabonly', 'pooled', 'dtat_o1', 'dtat_o2', 'reverse', 'random',
        'dtat_o1s', 'dtat_o2s']
POOL_SUBSET = {'dtat_o1': 'train_omega1', 'dtat_o2': 'train_omega2',
               'dtat_o1s': 'train_omega1s', 'dtat_o2s': 'train_omega2s'}
# Arms trained on the strictly repaired D_bal' instead of D_bal.
STRICT_ARMS = ('dtat_o1s', 'dtat_o2s')
ALL_TARGET_ARMS = ('pooled', 'dtat_o1', 'dtat_o2', 'reverse', 'random', 'dtat_o1s', 'dtat_o2s')
SEEDED_ARMS = ('random',)  # admitted pools depend on the curation seed


def _pool(train_dir: str, res: str, src: str, atk: str) -> str:
    return os.path.join(train_dir, f"{res}_{src}_{atk}_adv.csv")


def _tag(res: str, target: str, src: str, atk: str) -> str:
    # merge_training_samples.py names a pool by its attack when the source is the
    # target itself, otherwise by the file stem.
    return atk if src == target else f"{res}_{src}_{atk}_adv"


def _split(total: int, n: int) -> list[int]:
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def arm_spec(arm: str, res: str, target: str, seed: int | None = None) -> list[tuple[str, str, int]]:
    """Return (source, attack, quota) triples admitted for one target in one arm."""
    tau = TAU[res]
    is_tree = target in TREES
    spec: list[tuple[str, str, int]] = []
    if arm == 'random':
        rng = random.Random(f"{res}-{target}-{seed}")
        spec += [(rng.choice(TREES + NETS), atk, tau) for atk in QUERY]
        spec += [(rng.choice(NETS), atk, tau) for atk in GRAD]
        return spec
    if arm == 'reverse':
        if is_tree:
            return [(target, atk, tau) for atk in QUERY]
        for atk in QUERY:
            sources = [target] + TREES
            spec += [(src, atk, q) for src, q in zip(sources, _split(tau, len(sources)))]
        spec += [(target, atk, tau) for atk in GRAD]
        return spec
    if arm == 'pooled':
        for atk in QUERY:
            sources = TREES + NETS
            for src, q in zip(sources, _split(tau, len(sources))):
                spec.append((src, atk, q))
        for atk in GRAD:
            for src, q in zip(NETS, _split(tau, len(NETS))):
                spec.append((src, atk, q))
        return spec
    if not is_tree:
        if arm in ('dtat_o1', 'dtat_o2') + STRICT_ARMS:
            return [(target, atk, tau) for atk in QUERY + GRAD]
        raise ValueError(f"arm '{arm}' changes only tree targets; got {target}")
    own_q = int(3.5 * tau) if arm == 'iatvol' else tau
    spec += [(target, atk, own_q) for atk in QUERY]
    if arm in ('dtat', 'dtat_o1', 'dtat_o2') + STRICT_ARMS:
        spec += [(src, atk, q) for atk in GRAD for src, q in zip(NETS, _split(tau, 2))]
    elif arm == 'dnnonly':
        spec += [('dnn', atk, tau) for atk in GRAD]
    elif arm == 'tabonly':
        spec += [('tabnet', atk, tau) for atk in GRAD]
    return spec


def _run(cmd: list[str], log_path: str) -> None:
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [os.path.join(REPO_ROOT, 'src'), env.get('PYTHONPATH')]))
    with open(log_path, 'w') as fh:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}), see {log_path}: {' '.join(cmd)}")


def build(arm: str, res: str, target: str, seeds: list[int], python: str, log_dir: str) -> None:
    ds_dir = os.path.join(REPO_ROOT, 'resources', RES_DIRS[res])
    train_dir = os.path.join(ds_dir, 'adv_samples', POOL_SUBSET.get(arm, 'train'))
    arm_dir = os.path.join(ds_dir, 'ablation', arm)
    groups = [(seeds, None)] if arm not in SEEDED_ARMS else [([s], s) for s in seeds]
    for group_seeds, spec_seed in groups:
        spec = arm_spec(arm, res, target, spec_seed)
        pools_dir = os.path.join(arm_dir, 'pools', *([f"seed_{spec_seed}"] if spec_seed is not None else []))
        os.makedirs(pools_dir, exist_ok=True)

        adv_in = sorted({_pool(train_dir, res, src, atk) for src, atk, _ in spec})
        missing = [p for p in adv_in if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"missing pools for {arm}/{res}/{target}: {missing}")

        merged = os.path.join(pools_dir, f"{res}_{target}_robust_train.csv")
        tag = f"{arm}_{res}_{target}" + (f"_s{spec_seed}" if spec_seed is not None else '')
        if not os.path.exists(merged):
            logger.info(f"[+] merge {tag}: {len(adv_in)} pools")
            plain = (['--plain-train', os.path.join(REPO_ROOT, 'data', f"{res}_merged_train_raw_processed_omega1s.csv")]
                     if arm in STRICT_ARMS else [])
            _run([python, os.path.join(GEN_DIR, 'merge_training_samples.py'), '-r', res, '-m', target,
                  '--adv-in', *adv_in, '--output-dir', pools_dir, *plain],
                 os.path.join(log_dir, f"merge_{tag}.log"))

        targets = [f"{_tag(res, target, src, atk)}={q}" for src, atk, q in spec]
        total = sum(q for _, _, q in spec)
        for s in group_seeds:
            out = os.path.join(arm_dir, f"seed_{s}", f"{res}_{target}_robust_train_post.csv")
            if os.path.exists(out):
                logger.info(f"[=] exists, skip: {out}")
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            logger.info(f"[+] curate {tag} seed={s}: {len(spec)} quota groups, nominal {total} records | "
                        + ", ".join(f"{src}:{atk}" for src, atk, _ in spec))
            _run([python, os.path.join(GEN_DIR, 'postprocess_merge.py'), '-r', res, '-m', target,
                  '--seed', str(s), '--input', merged, '--output', out,
                  '--sample-threshold', str(TAU[res]), '--attack-targets', *targets],
                 os.path.join(log_dir, f"post_{tag}_seed{s}.log"))


def main():
    parser = argparse.ArgumentParser(description="Build curated curricula for revision ablation arms")
    parser.add_argument('--resource', '-r', nargs='+', required=True, choices=list(RES_DIRS))
    parser.add_argument('--arm', '-a', nargs='+', required=True, choices=ARMS)
    parser.add_argument('--model', '-m', nargs='+', default=None, choices=TREES + NETS,
                        help="Targets (default: trees; pooled arm also adds dnn/tabnet)")
    parser.add_argument('--seeds', type=int, nargs='+', default=[1337, 1338, 1339, 1340])
    parser.add_argument('--python', type=str, default=sys.executable)
    parser.add_argument('--log-dir', type=str, default='/tmp/ablation_logs')
    parser.add_argument('--log-level', type=str, default='INFO')
    args = parser.parse_args()
    setup_logging(args.log_level)
    os.makedirs(args.log_dir, exist_ok=True)

    for res in args.resource:
        for arm in args.arm:
            models = args.model or (TREES + NETS if arm in ALL_TARGET_ARMS else TREES)
            for target in models:
                build(arm, res, target, args.seeds, args.python, args.log_dir)


if __name__ == '__main__':
    main()
