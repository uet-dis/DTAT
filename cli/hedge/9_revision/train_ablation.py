"""Fit the ablation-arm checkpoints and assemble one models dir per arm.

For each (resource, arm, model, seed) the curriculum built by build_ablation_pools.py
is fitted with the unchanged training entry point (model_training.py --adv-train).
Checkpoints go to resources/<DS>/ablation/<arm>/models/ under the robust-seed names
the evaluators already understand, so an arm is evaluated with

    --models-dir resources/<DS>/ablation/<arm>/models --model-seeds 1337 1338 1339 1340

Components an arm does not retrain are symlinked from resources/<DS>/models/: the
clean checkpoints always, and the DTAT networks for arms that change only trees.

Lanes run concurrently: one CatBoost job on the GPU, one other GPU job (xgb, dnn,
tabnet), and a CPU lane for the single-threaded sklearn learners (gbm, bagging).
"""
import os
import sys
import argparse
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
TRAIN_PY = os.path.join(REPO_ROOT, 'cli', 'hedge', '5_training', 'model_training.py')

RES_DIRS = {'cic2018': 'CIC-2018', 'nslkdd': 'NSLKDD'}
TREES = ['xgb', 'catb', 'gbm', 'bagging']
NETS = ['dnn', 'tabnet']
LANE_OF = {'catb': 'gpu_catb', 'xgb': 'gpu', 'dnn': 'gpu', 'tabnet': 'gpu', 'gbm': 'cpu', 'bagging': 'cpu'}


def _ext(model: str) -> str:
    return '.pth' if model in NETS else '.pkl'


ALL_TARGET_ARMS = ('pooled', 'dtat_o1', 'dtat_o2', 'reverse', 'random', 'dtat_o1s', 'dtat_o2s')
STRICT_ARMS = ('dtat_o1s', 'dtat_o2s')  # fitted on D_bal' instead of D_bal


def _arm_models(arm: str) -> list[str]:
    return TREES + NETS if arm in ALL_TARGET_ARMS else TREES


def link_models(res: str, arm: str, seeds: list[int]) -> None:
    """Symlink the checkpoints an arm shares with the paper's ensembles."""
    ds_dir = os.path.join(REPO_ROOT, 'resources', RES_DIRS[res])
    src_dir = os.path.join(ds_dir, 'models')
    dst_dir = os.path.join(ds_dir, 'ablation', arm, 'models')
    os.makedirs(dst_dir, exist_ok=True)
    names = [f"{res}_{m}{_ext(m)}" for m in TREES + NETS] + [f"{res}_xgb_binary.pkl"]
    if arm not in ALL_TARGET_ARMS:
        names += [f"{res}_{m}_robust_seed{s}{_ext(m)}" for m in NETS for s in seeds]
    for name in names:
        src, dst = os.path.join(src_dir, name), os.path.join(dst_dir, name)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(os.path.relpath(src, dst_dir), dst)


def _job_paths(res: str, arm: str, model: str, seed: int) -> dict:
    arm_dir = os.path.join(REPO_ROOT, 'resources', RES_DIRS[res], 'ablation', arm)
    stem = f"{model}_robust_seed{seed}"
    return {
        'adv': os.path.join(arm_dir, f"seed_{seed}", f"{res}_{model}_robust_train_post.csv"),
        'model': os.path.join(arm_dir, 'models', f"{res}_{stem}{_ext(model)}"),
        'report': os.path.join(arm_dir, 'reports', model, f"{stem}_metrics.json"),
        'cm': os.path.join(arm_dir, 'reports', model, f"{stem}_cm.png"),
    }


def run_job(job: tuple, python: str, log_dir: str, params_dir: str) -> str:
    res, arm, model, seed = job
    p = _job_paths(res, arm, model, seed)
    tag = f"{res}/{arm}/{model}/seed{seed}"
    if os.path.exists(p['model']):
        return f"skip {tag}"
    if not os.path.exists(p['adv']):
        return f"MISSING-CURRICULUM {tag}"
    os.makedirs(os.path.dirname(p['report']), exist_ok=True)
    log = os.path.join(log_dir, f"train_{res}_{arm}_{model}_seed{seed}.log")
    device = 'cpu' if LANE_OF[model] == 'cpu' else 'auto'
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [os.path.join(REPO_ROOT, 'src'), env.get('PYTHONPATH')]))
    cmd = [python, TRAIN_PY, '-r', res, '-m', model, '--params-dir', params_dir, '--device', device,
           '--adv-train', '--adv-in', p['adv'], '--model-out', p['model'],
           '--report-out', p['report'], '--cm-out', p['cm']]
    if arm in STRICT_ARMS:
        cmd += ['--train-in', os.path.join(REPO_ROOT, 'data', f"{res}_merged_train_raw_processed_omega1s.csv")]
    t0 = time.time()
    with open(log, 'w') as fh:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    status = 'OK' if rc == 0 and os.path.exists(p['model']) else f'FAILED rc={rc}'
    return f"{status} {tag} ({(time.time() - t0) / 60:.1f} min)"


def main():
    parser = argparse.ArgumentParser(description="Train ablation-arm checkpoints in GPU/CPU lanes")
    parser.add_argument('--resource', '-r', nargs='+', required=True, choices=list(RES_DIRS))
    parser.add_argument('--arm', '-a', nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1337, 1338, 1339, 1340])
    parser.add_argument('--cpu-parallel', type=int, default=16)
    parser.add_argument('--python', type=str, default=sys.executable)
    parser.add_argument('--params-dir', type=str, default=os.path.join(REPO_ROOT, 'params', 'training'))
    parser.add_argument('--log-dir', type=str, default='/tmp/ablation_logs')
    parser.add_argument('--log-level', type=str, default='INFO')
    args = parser.parse_args()
    setup_logging(args.log_level)
    os.makedirs(args.log_dir, exist_ok=True)

    # Job order = priority order: arms as given, then resources, then seeds.
    lanes: dict[str, list] = {'gpu_catb': [], 'gpu': [], 'cpu': []}
    for arm in args.arm:
        for res in args.resource:
            link_models(res, arm, args.seeds)
            for seed in args.seeds:
                for model in _arm_models(arm):
                    lanes[LANE_OF[model]].append((res, arm, model, seed))

    width = {'gpu_catb': 1, 'gpu': 1, 'cpu': args.cpu_parallel}
    lock = threading.Lock()

    def drain(lane: str) -> None:
        with ThreadPoolExecutor(max_workers=width[lane]) as pool:
            for msg in pool.map(lambda j: run_job(j, args.python, args.log_dir, args.params_dir), lanes[lane]):
                with lock:
                    logger.info(f"[{lane}] {msg}")

    threads = [threading.Thread(target=drain, args=(lane,)) for lane in lanes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logger.info("[+] all lanes finished")


if __name__ == '__main__':
    main()
