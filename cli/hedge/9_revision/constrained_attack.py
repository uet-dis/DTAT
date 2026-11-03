"""Domain-constrained and defense-aware evaluation for the revision.

Experiments (all vectors satisfy Pi_Omega at every iterate, see domain_constraints.py):

  ens-rs     defense-aware end-to-end attack: score-based random search against the
             complete weighted ensemble of a variant (Eq. 2); ONE vector is fed to
             every branch. ASR is conditioned on the ensemble's clean correctness.
  comp-rs    the same search against each single component of a variant
             (defense-aware direct exposure of the retrained component).
  cpgd       constrained PGD generated against a clean-trained network (DNN or TabNet),
             transferred unchanged to every tree of a variant: the constrained version
             of the paper's gradient pathway (Table 7). With --pgd-source-variant the
             source network is the variant's own retrained network instead.
  ens-sq     Square Attack (L_inf, Andriushchenko et al. 2020) against the complete
             ensemble with Pi_Omega in the loop; one vector to every branch.
  comp-sq    the same Square Attack against each retrained component.
  validate   recompute one Table 5 entry from the original vectors to check that model
             loading and ASR here match the paper's evaluator.

Results are appended to resources/<DS>/revision/results_constrained.csv.
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger
from art_generator.domain_constraints import repair, audit, mutable_columns
from art_generator.inputnorm_adapter import InputNormStats

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources  # noqa: E402
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor  # noqa: E402
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor  # noqa: E402
from preprocessing.prepare import PrepareData  # noqa: E402
from helpers.wrapping_helper import load_model_as_wrapper, predict_one_wrapper  # noqa: E402
from helpers.ensemble_helper import ENSEMBLE_WEIGHTS  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
REGISTRY = {'cic2018': (CIC2018Resources, CIC2018Preprocessor), 'nslkdd': (NSLKDDResources, NSLKDDPreprocessor)}
RES_DIRS = {'cic2018': 'CIC-2018', 'nslkdd': 'NSLKDD'}
MODELS = ['xgb', 'catb', 'gbm', 'bagging', 'dnn', 'tabnet']
TREES = ['xgb', 'catb', 'gbm', 'bagging']


# --------------------------------------------------------------------------- #
# Variants                                                                     #
# --------------------------------------------------------------------------- #
def variant_files(res: str, variant: str, seed: int | None) -> dict[str, str]:
    """Checkpoint per component. variant: base | iat_paper | dtat | <ablation arm>."""
    ds = os.path.join(REPO_ROOT, 'resources', RES_DIRS[res])
    out = {}
    for m in MODELS:
        ext = '.pth' if m in ('dnn', 'tabnet') else '.pkl'
        if variant == 'base':
            p = os.path.join(ds, 'models', f"{res}_{m}{ext}")
        elif variant == 'iat_paper':
            name = f"{res}_{m}_isolated_robust{ext}" if m in TREES else f"{res}_{m}_robust_seed1337{ext}"
            p = os.path.join(ds, 'models', name)
        elif variant == 'dtat':
            p = os.path.join(ds, 'models', f"{res}_{m}_robust_seed{seed}{ext}")
        else:
            p = os.path.join(ds, 'ablation', variant, 'models', f"{res}_{m}_robust_seed{seed}{ext}")
        if not os.path.exists(p):
            raise SystemExit(f"missing checkpoint for {variant}/seed{seed}/{m}: {p}")
        out[m] = p
    return out


class Predictor:
    """Probability outputs of single components and of the weighted ensemble."""

    def __init__(self, files: dict[str, str], num_class: int, input_dim: int, clip_values, device: str):
        self.num_class = num_class
        self.wrappers = {m: load_model_as_wrapper(m, p, num_classes=num_class, input_dim=input_dim,
                                                  clip_values=clip_values, device=device)
                         for m, p in files.items()}
        self.queries = 0

    def proba(self, X: np.ndarray, member: str | None = None) -> np.ndarray:
        self.queries += len(X)
        if member is not None:
            return predict_one_wrapper(self.wrappers[member], member, X, self.num_class)
        total = np.zeros((len(X), self.num_class))
        wsum = 0.0
        for m, w in self.wrappers.items():
            total += ENSEMBLE_WEIGHTS[m] * predict_one_wrapper(w, m, X, self.num_class)
            wsum += ENSEMBLE_WEIGHTS[m]
        return total / wsum


# --------------------------------------------------------------------------- #
# Constraint-respecting attacks                                                #
# --------------------------------------------------------------------------- #
class OmegaSpace:
    """Pi_Omega and the eps-box over the mutable primaries, in model-input space."""

    def __init__(self, feature_names: list[str], level: str, sigma: np.ndarray, eps: float, X0: np.ndarray):
        self.cols = list(feature_names)
        self.level = level
        mut, nondec = mutable_columns(self.cols, level)
        self.mut = np.array([self.cols.index(c) for c in mut], dtype=int)
        self.nondec = np.isin(self.mut, [self.cols.index(c) for c in nondec])
        self.X0 = X0
        rad = eps * sigma[self.mut]
        self.lo = X0[:, self.mut] - np.where(self.nondec, 0.0, rad)[None, :]
        self.hi = X0[:, self.mut] + rad[None, :]
        self.step = rad

    def project(self, X: np.ndarray, rows: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[:, self.mut] = np.clip(X[:, self.mut], self.lo[rows], self.hi[rows])
        df = pd.DataFrame(X, columns=self.cols)
        orig = pd.DataFrame(self.X0[rows], columns=self.cols)
        return repair(df, orig, self.level)[self.cols].to_numpy(dtype=np.float64)


def random_search(score_fn, space: OmegaSpace, y: np.ndarray, iters: int, k_frac: float,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Score-based untargeted search; every iterate lies in Omega and the eps-box.

    Each step moves a random subset of mutable primaries by a random fraction of their
    radius (only upward for non-decreasing coordinates), repairs, queries, and keeps
    the candidate if the true-class probability does not increase.
    """
    X = space.X0.copy()
    n = len(X)
    p = score_fn(X)
    cur = p[np.arange(n), y]
    active = p.argmax(1) == y
    success = ~active
    n_mut = len(space.mut)
    k = max(1, int(round(k_frac * n_mut)))
    for _ in range(iters):
        idx = np.where(active)[0]
        if idx.size == 0:
            break
        cand = X[idx].copy()
        pick = np.argsort(rng.random((idx.size, n_mut)), axis=1)[:, :k]
        sign = rng.choice([-1.0, 1.0], size=(idx.size, k))
        sign = np.where(space.nondec[pick], 1.0, sign)
        mag = rng.uniform(0.25, 1.0, size=(idx.size, k)) * space.step[pick]
        cand[np.arange(idx.size)[:, None], space.mut[pick]] += sign * mag
        cand = space.project(cand, idx)
        pc = score_fn(cand)
        q = pc[np.arange(idx.size), y[idx]]
        flipped = pc.argmax(1) != y[idx]
        take = (q <= cur[idx] + 1e-12) | flipped
        X[idx[take]] = cand[take]
        cur[idx[take]] = q[take]
        hit = idx[take & flipped]
        success[hit] = True
        active[hit] = False
    return X, success


def _square_p(p_init: float, it: int, n_iters: int) -> float:
    """Window-size schedule of the L_inf Square Attack (rescaled to 10,000 iterations)."""
    i = int(it / n_iters * 10000)
    for thr, div in ((8000, 512), (6000, 256), (4000, 128), (2000, 64), (1000, 32),
                     (500, 16), (200, 8), (50, 4), (10, 2)):
        if i > thr:
            return p_init / div
    return p_init


def square_attack(score_fn, space: OmegaSpace, y: np.ndarray, iters: int, p_init: float,
                  rng: np.random.Generator) -> np.ndarray:
    """L_inf Square Attack adapted to tabular inputs; every iterate is repaired by Pi_Omega.

    Margin loss p_y - max_{c != y} p_c is minimized. The search starts at a random vertex
    of the eps-box of the mutable primaries; each step resets a random window of
    ceil(p_i |K|) coordinates to random vertices (only the upper vertex or the original
    value for coordinates that may only grow) and keeps the candidate if the margin
    decreases. Images use square windows; for tabular features the window is a random
    coordinate subset, as feature order carries no spatial meaning.
    """
    n, k = len(space.X0), len(space.mut)

    def vertices(shape):
        up = rng.random(shape) < 0.5
        lo = np.where(space.nondec[None, :], 0.0, -1.0) * space.step[None, :]
        return np.where(up, space.step[None, :], lo)

    def margin(p, yy):
        q = p.copy()
        q[np.arange(len(yy)), yy] = -np.inf
        return p[np.arange(len(yy)), yy] - q.max(1)

    X = space.X0.copy()
    X[:, space.mut] = space.X0[:, space.mut] + vertices((n, k))
    X = space.project(X, np.arange(n))
    cur = margin(score_fn(X), y)
    for it in range(iters):
        idx = np.where(cur > 0)[0]
        if idx.size == 0:
            break
        w = max(1, int(np.ceil(_square_p(p_init, it, iters) * k)))
        pick = np.argsort(rng.random((idx.size, k)), axis=1)[:, :w]
        cand = X[idx].copy()
        new = vertices((idx.size, k))
        rows = np.arange(idx.size)[:, None]
        cand[rows, space.mut[pick]] = space.X0[idx][rows, space.mut[pick]] + new[rows, pick]
        cand = space.project(cand, idx)
        m = margin(score_fn(cand), y[idx])
        take = m < cur[idx]
        X[idx[take]] = cand[take]
        cur[idx[take]] = m[take]
    return X


def constrained_pgd(wrapper, space: OmegaSpace, y: np.ndarray, iters: int, batch: int = 4096) -> np.ndarray:
    """Untargeted L_inf PGD on a network; Pi_Omega after every step."""
    est = wrapper.get_estimator()
    y = np.eye(est.nb_classes, dtype=np.float32)[y]
    X = space.X0.copy()
    alpha = space.step / 10.0
    for _ in range(iters):
        grads = []
        for s in range(0, len(X), batch):
            grads.append(est.loss_gradient(X[s:s + batch].astype(np.float32), y[s:s + batch]))
        g = np.concatenate(grads).astype(np.float64)[:, space.mut]
        step = alpha[None, :] * np.sign(g)
        step = np.where(space.nondec[None, :], np.maximum(step, 0.0), step)
        X[:, space.mut] += step
        X = space.project(X, np.arange(len(X)))
    return X


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def asr(pred_clean: np.ndarray, pred_adv: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    ok = pred_clean == y
    return (100.0 * float((pred_adv[ok] != y[ok]).mean()) if ok.any() else float('nan')), int(ok.sum())


def stratified(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        rows = np.where(y == c)[0]
        idx.append(rng.choice(rows, size=min(per_class, rows.size), replace=False))
    return np.sort(np.concatenate(idx))


def main():
    ap = argparse.ArgumentParser(description="Domain-constrained / defense-aware evaluation")
    ap.add_argument('--resource', '-r', required=True, choices=list(REGISTRY))
    ap.add_argument('--experiment', '-e', required=True, choices=['ens-rs', 'comp-rs', 'ens-sq', 'comp-sq', 'cpgd', 'validate'])
    ap.add_argument('--variant', default='base')
    ap.add_argument('--seed', type=int, default=None, help="model (curation) seed of the variant")
    ap.add_argument('--level', default='omega1', choices=['omega1', 'omega2'])
    ap.add_argument('--eps', type=float, default=0.2, help="radius in units of the DNN input-norm sigma")
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--k-frac', type=float, default=0.15, help="fraction of mutable primaries moved per step")
    ap.add_argument('--p-init', type=float, default=0.1, help="Square Attack initial window fraction")
    ap.add_argument('--per-class', type=int, default=500, help="stratified evaluation subsample per class")
    ap.add_argument('--sample-seed', type=int, default=0)
    ap.add_argument('--attack-seed', type=int, default=0)
    ap.add_argument('--pgd-source', default='dnn', choices=['dnn', 'tabnet'])
    ap.add_argument('--procedure', default='pgd', help="validate: procedure of the original vectors")
    ap.add_argument('--pgd-source-variant', action='store_true',
                    help="cpgd: generate on the variant's own network instead of the clean one")
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=None)
    ap.add_argument('--log-level', default='INFO')
    args = ap.parse_args()
    setup_logging(args.log_level)

    res_cls, pre_cls = REGISTRY[args.resource]
    pre = pre_cls()
    if not pre.load_encoders(fixed_label_encoder=True):
        raise SystemExit("encoders not found")
    test_csv = os.path.join(res_cls.BALENCED_DATA_FOLDER, f"{res_cls.resources_name}_test_random_sample_clean_merged.csv")
    X_all, y_all, meta = PrepareData.prepare_input_data(pd.read_csv(test_csv, low_memory=False), pre, include_label=True)
    X_all = X_all.astype(np.float64)
    num_class = len(res_cls.MAJORITY_LABELS + res_cls.MINORITY_LABELS)
    feats = list(meta['feature_names'])
    clip = meta['clip_values']

    # sigma: input-norm statistics of the clean DNN checkpoint (D_bal), as in the paper.
    base_files = variant_files(args.resource, 'base', None)
    dnn0 = load_model_as_wrapper('dnn', base_files['dnn'], num_classes=num_class, input_dim=X_all.shape[1],
                                 clip_values=clip, device=args.device)
    stats = InputNormStats.from_estimator(dnn0.get_estimator())
    sigma = np.ones(X_all.shape[1])
    sigma[stats.cont_idx] = stats.denom

    rows = []
    t0 = time.time()
    common = {'dataset': args.resource, 'experiment': args.experiment, 'variant': args.variant,
              'seed': args.seed, 'level': args.level, 'eps': args.eps,
              'attack_seed': args.attack_seed, 'sample_seed': args.sample_seed}

    if args.experiment == 'validate':
        adv = pd.read_csv(os.path.join(REPO_ROOT, 'resources', RES_DIRS[args.resource], 'adv_samples', 'test',
                                       f"{args.resource}_{args.pgd_source}_{args.procedure}_adv.csv"), low_memory=False)
        X_adv, y_adv, _ = PrepareData.prepare_input_data(adv, pre, include_label=True)
        pred = Predictor(base_files, num_class, X_all.shape[1], clip, args.device)
        for m in MODELS:
            a, n = asr(pred.proba(X_all, m).argmax(1), pred.proba(X_adv, m).argmax(1), y_all)
            rows.append({**common, 'target': m, 'source': f"{args.pgd_source}-{args.procedure}-original", 'n_clean_correct': n,
                         'asr': round(a, 2)})
            logger.info(f"[validate] {args.pgd_source} {args.procedure} -> {m}: ASR {a:.2f}%")
    else:
        sel = stratified(y_all, args.per_class, args.sample_seed)
        X0, y = X_all[sel], y_all[sel]
        space = OmegaSpace(feats, args.level, sigma, args.eps, X0)
        files = variant_files(args.resource, args.variant, args.seed)
        pred = Predictor(files, num_class, X_all.shape[1], clip, args.device)
        rng = np.random.default_rng(args.attack_seed)
        emitted: list[np.ndarray] = []

        if args.experiment == 'ens-rs':
            clean = pred.proba(X0).argmax(1)
            X_adv, _ = random_search(lambda X: pred.proba(X), space, y, args.iters, args.k_frac, rng)
            a, n = asr(clean, pred.proba(X_adv).argmax(1), y)
            rows.append({**common, 'target': 'ensemble', 'source': 'ensemble', 'n_clean_correct': n,
                         'asr': round(a, 2), 'iters': args.iters})
        elif args.experiment == 'ens-sq':
            clean = pred.proba(X0).argmax(1)
            X_adv = square_attack(lambda X: pred.proba(X), space, y, args.iters, args.p_init, rng)
            a, n = asr(clean, pred.proba(X_adv).argmax(1), y)
            rows.append({**common, 'target': 'ensemble', 'source': 'ensemble', 'n_clean_correct': n,
                         'asr': round(a, 2), 'iters': args.iters})
        elif args.experiment == 'comp-sq':
            for m in MODELS:
                clean = pred.proba(X0, m).argmax(1)
                X_adv = square_attack(lambda X, m=m: pred.proba(X, m), space, y, args.iters, args.p_init, rng)
                a, n = asr(clean, pred.proba(X_adv, m).argmax(1), y)
                rows.append({**common, 'target': m, 'source': m, 'n_clean_correct': n, 'asr': round(a, 2),
                             'iters': args.iters})
                logger.info(f"[comp-sq] {args.variant}/{m}: ASR {a:.2f}%")
                emitted.append(X_adv)
        elif args.experiment == 'comp-rs':
            for m in MODELS:
                clean = pred.proba(X0, m).argmax(1)
                X_adv, _ = random_search(lambda X, m=m: pred.proba(X, m), space, y, args.iters, args.k_frac, rng)
                a, n = asr(clean, pred.proba(X_adv, m).argmax(1), y)
                rows.append({**common, 'target': m, 'source': m, 'n_clean_correct': n, 'asr': round(a, 2),
                             'iters': args.iters})
                logger.info(f"[comp-rs] {args.variant}/{m}: ASR {a:.2f}%")
                emitted.append(X_adv)
        elif args.experiment == 'cpgd':
            src_files = files if args.pgd_source_variant else base_files
            src = load_model_as_wrapper(args.pgd_source, src_files[args.pgd_source], num_classes=num_class,
                                        input_dim=X_all.shape[1], clip_values=clip, device=args.device)
            X_adv = constrained_pgd(src, space, y, args.iters)
            tag = f"{'variant' if args.pgd_source_variant else 'clean'}-{args.pgd_source}"
            for m in MODELS:
                a, n = asr(pred.proba(X0, m).argmax(1), pred.proba(X_adv, m).argmax(1), y)
                rows.append({**common, 'target': m, 'source': tag, 'n_clean_correct': n, 'asr': round(a, 2),
                             'iters': args.iters})
            # Per-branch ensemble condition of Section 3.4 (case = pgd source): networks get
            # their own constrained PGD vectors, trees the source's vector.
            ens_clean = pred.proba(X0).argmax(1)
            own = {}
            for net in ('dnn', 'tabnet'):
                if net == args.pgd_source:
                    own[net] = X_adv
                else:
                    w = load_model_as_wrapper(net, src_files[net], num_classes=num_class, input_dim=X_all.shape[1],
                                              clip_values=clip, device=args.device)
                    own[net] = constrained_pgd(w, space, y, args.iters)
            total = np.zeros((len(X0), num_class))
            for m in MODELS:
                total += ENSEMBLE_WEIGHTS[m] * pred.proba(own.get(m, X_adv), m)
            a, n = asr(ens_clean, total.argmax(1), y)
            rows.append({**common, 'target': 'ensemble-per-branch', 'source': tag, 'n_clean_correct': n,
                         'asr': round(a, 2), 'iters': args.iters})
        # Every emitted vector must satisfy the level's constraints.
        emitted = emitted or [X_adv]
        flags = audit(pd.DataFrame(np.vstack(emitted), columns=feats),
                      pd.DataFrame(np.vstack([X0] * len(emitted)), columns=feats))
        viol = 100.0 * float(flags['any'].mean())
        cap = 100.0 * float(flags['capability'].mean()) if args.level == 'omega2' else float('nan')
        for r in rows:
            r['n_sample'] = len(X0)
            r['queries'] = pred.queries
            r['viol_pct'] = round(viol, 3)
            r['cap_viol_pct'] = round(cap, 3)

    for r in rows:
        r['minutes'] = round((time.time() - t0) / 60, 2)
    out = args.out or os.path.join(REPO_ROOT, 'resources', RES_DIRS[args.resource], 'revision', 'results_constrained.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, mode='a', header=not os.path.exists(out), index=False)
    logger.info("\n" + df.to_string(index=False))


if __name__ == '__main__':
    main()
