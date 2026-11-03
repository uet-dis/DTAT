"""Repair the balanced training set D_bal with Pi_Omega1 in strict mode.

The WGAN-generated minority records of D_bal violate the flow-meter relations far
more often than real records. The strict repair (every relation imposed, no
carried-over deviation) yields D_bal' at data/<res>_merged_train_raw_processed_omega1s.csv,
row-aligned with D_bal; columns that are not model features are copied unchanged.
"""
import os
import argparse

import pandas as pd

from utils.logging import setup_logging, get_logger
from art_generator.domain_constraints import repair, violations_absolute

logger = get_logger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
RES_DIRS = {'cic2018': 'CIC-2018', 'nslkdd': 'NSLKDD'}


def main():
    ap = argparse.ArgumentParser(description="Strict Pi_Omega1 repair of D_bal")
    ap.add_argument('--resource', '-r', nargs='+', default=['nslkdd', 'cic2018'])
    args = ap.parse_args()
    setup_logging('INFO')
    rows = []
    for res in args.resource:
        src = os.path.join(REPO_ROOT, 'data', f"{res}_merged_train_raw_processed.csv")
        dst = os.path.join(REPO_ROOT, 'data', f"{res}_merged_train_raw_processed_omega1s.csv")
        d = pd.read_csv(src, low_memory=False)
        # model features = columns of the adversarial pools (constant features are dropped)
        ref = pd.read_csv(os.path.join(REPO_ROOT, 'resources', RES_DIRS[res], 'adv_samples', 'train',
                                       f"{res}_dnn_pgd_adv.csv"), nrows=1)
        feats = [c for c in ref.columns if c != 'Label']
        before = violations_absolute(d[feats])
        rep = repair(d[feats], d[feats], 'omega1', strict=True)
        after = violations_absolute(rep)
        out = d.copy()
        out[feats] = rep[feats].values
        out.to_csv(dst, index=False)
        src_col = d['__source__'] if '__source__' in d else pd.Series('all', index=d.index)
        for name, g in pd.DataFrame({'src': src_col, 'b': before['any'], 'a': after['any']}).groupby('src'):
            rows.append({'dataset': res, 'source': name, 'n': len(g), 'viol_before_pct': round(100 * g.b.mean(), 2),
                         'viol_after_pct': round(100 * g.a.mean(), 2)})
        logger.info(f"[+] {res}: violating {100 * before['any'].mean():.2f}% -> {100 * after['any'].mean():.2f}% | {dst}")
        pd.DataFrame([r for r in rows if r['dataset'] == res]).to_csv(
            os.path.join(REPO_ROOT, 'resources', RES_DIRS[res], 'revision', 'audit_dbal.csv'), index=False)


if __name__ == '__main__':
    main()
