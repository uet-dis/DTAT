"""Audit adversarial vectors against domain constraints and write Pi_Omega-repaired copies.

For every adversarial CSV in resources/<DS>/adv_samples/<subset>/ (and its seed_k/
subfolders) the vectors are compared row-wise with the clean records they were
generated from. The script records how many violate the constraints and writes the
repaired vectors to adv_samples/<subset>_<level>/ with identical file names, so the
existing evaluators run on them unchanged via --subset / --adv-subset.

Audit table: resources/<DS>/revision/audit_<subset>.csv
"""
import os
import sys
import glob
import argparse

import pandas as pd

from utils.logging import setup_logging, get_logger
from art_generator.domain_constraints import repair, audit

logger = get_logger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
RES_DIRS = {'cic2018': 'CIC-2018', 'nslkdd': 'NSLKDD'}
CLEAN = {'cic2018': 'data/cic2018_test_random_sample_clean_merged.csv',
         'nslkdd': 'data/nslkdd_test_random_sample_clean_merged.csv'}
# Training pools are generated from D_bal and are row-aligned with it.
CLEAN_TRAIN = {'cic2018': 'data/cic2018_merged_train_raw_processed.csv',
               'nslkdd': 'data/nslkdd_merged_train_raw_processed.csv'}


def _summary(flags: pd.DataFrame) -> dict:
    return {f"viol_{k}": round(100.0 * float(flags[k].mean()), 3) for k in flags.columns}


def main():
    parser = argparse.ArgumentParser(description="Constraint audit and Pi_Omega repair of adversarial test vectors")
    parser.add_argument('--resource', '-r', required=True, choices=list(RES_DIRS))
    parser.add_argument('--subset', default='test')
    parser.add_argument('--levels', nargs='+', default=['omega1', 'omega2'], choices=['omega1', 'omega2'])
    parser.add_argument('--strict', action='store_true',
                        help="impose every relation (no carried-over deviations); outputs <subset>_<level>s/")
    parser.add_argument('--clean-in', default=None,
                        help="row-aligned clean records to repair against (default: D_bal for train, test otherwise)")
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args()
    setup_logging(args.log_level)

    ds_dir = os.path.join(REPO_ROOT, 'resources', RES_DIRS[args.resource])
    src_root = os.path.join(ds_dir, 'adv_samples', args.subset)
    clean_map = CLEAN_TRAIN if args.subset == 'train' else CLEAN
    clean = pd.read_csv(args.clean_in or os.path.join(REPO_ROOT, clean_map[args.resource]))
    suffix = 's' if args.strict else ''
    label = 'Label'

    rows = []
    folders = [''] + sorted(os.path.basename(p) for p in glob.glob(os.path.join(src_root, 'seed_*')))
    for folder in folders:
        for path in sorted(glob.glob(os.path.join(src_root, folder, '*_adv.csv'))):
            name = os.path.basename(path)
            adv = pd.read_csv(path)
            if len(adv) != len(clean) or not (adv[label].values == clean[label].values).all():
                raise SystemExit(f"{path} is not row-aligned with its clean records")
            feats = [c for c in adv.columns if c != label]
            orig = clean[adv.columns]
            changed = (adv[feats].values != orig[feats].values).any(axis=1)
            flags = audit(adv[feats], orig[feats])
            row = {'seed_dir': folder or 'seed_0', 'file': name, 'n': len(adv),
                   'changed_pct': round(100.0 * changed.mean(), 3), **_summary(flags)}
            for level in args.levels:
                rep = repair(adv[feats], orig[feats], level, strict=args.strict)
                post = audit(rep, orig[feats])
                row[f'{level}_residual_any'] = round(100.0 * float(post['any'].mean()), 3)
                row[f'{level}_still_changed_pct'] = round(
                    100.0 * float((rep.values != orig[feats].values).any(axis=1).mean()), 3)
                out_dir = os.path.join(ds_dir, 'adv_samples', f"{args.subset}_{level}{suffix}", folder)
                os.makedirs(out_dir, exist_ok=True)
                rep[label] = adv[label].values
                rep[adv.columns].to_csv(os.path.join(out_dir, name), index=False)
            rows.append(row)
            logger.info(f"[+] {row['seed_dir']}/{name}: changed {row['changed_pct']}% | "
                        f"violating {row['viol_any']}% | capability-violating {row['viol_capability']}%")

    out = os.path.join(ds_dir, 'revision', f"audit_{args.subset}{suffix}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info(f"[+] audit table: {out}")


if __name__ == '__main__':
    main()
