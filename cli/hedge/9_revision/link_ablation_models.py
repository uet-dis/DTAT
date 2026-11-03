"""Recreate the shared-checkpoint links of every ablation arm.

The released archive contains only the checkpoints each arm actually fitted. An arm
also needs the clean checkpoints (resources/<DS>/models/<res>_<model>.*) and, if it
changes only the trees, the DTAT networks; the Reverse arm reuses the IAT trees.
This script restores those links (relative symlinks) after the archives are
extracted, so the evaluators can load every arm with --models-dir.
"""
import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from train_ablation import link_models, RES_DIRS, REPO_ROOT, TREES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Restore shared-checkpoint links of the ablation arms")
    ap.add_argument('--resource', '-r', nargs='+', default=list(RES_DIRS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[1337, 1338, 1339, 1340])
    args = ap.parse_args()
    for res in args.resource:
        root = os.path.join(REPO_ROOT, 'resources', RES_DIRS[res], 'ablation')
        if not os.path.isdir(root):
            print(f"[!] {root} not found; extract the checkpoint archive first")
            continue
        for arm in sorted(os.listdir(root)):
            if not os.path.isdir(os.path.join(root, arm, 'models')):
                continue
            link_models(res, arm, args.seeds)
            if arm == 'reverse':  # Reverse trees are the IAT trees (identical curricula)
                for m in TREES:
                    for s in args.seeds:
                        name = f"{res}_{m}_robust_seed{s}.pkl"
                        src = os.path.join(root, 'iat', 'models', name)
                        dst = os.path.join(root, 'reverse', 'models', name)
                        if os.path.exists(src) and not os.path.lexists(dst):
                            os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
            print(f"[+] linked {res}/{arm}")


if __name__ == '__main__':
    main()
