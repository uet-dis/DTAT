"""Locations of the revision result files.

Experiments write their results to resources/<DS>/revision/. The public repository
ships the same files under results/revision/<DS>/ (resources/ is not versioned), and
the readers fall back to that copy, so every table and figure can be rebuilt from a
fresh clone without rerunning an experiment.
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
OUTPUT_ROOT = os.path.join(REPO_ROOT, 'results', 'revision')


def revision_dir(ds: str) -> str:
    """Result directory of dataset folder `ds` ('NSLKDD' or 'CIC-2018')."""
    local = os.path.join(REPO_ROOT, 'resources', ds, 'revision')
    return local if os.path.isdir(local) else os.path.join(OUTPUT_ROOT, ds)
