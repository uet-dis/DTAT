import os
import sys
import argparse
import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger
from resampling.undersampling.kmeans_reps import KMeansRepresentativeSelector



SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}

 


def _load_encoded_df(res, subset: str, label: str) -> pd.DataFrame:
    label_safe = res.get_label_name(label)
    path = res.encoded_path_for(subset, label_safe)
    if not os.path.exists(path):
        raise SystemExit(f"Encoded CSV not found for {label} in subset {subset}: {path}")
    df = pd.read_csv(path, low_memory=False)
    return df


def _run_one(res, subset: str, label: str, budget: int, n_clusters: int | None, edge_percentile: float,
             batch_size: int, out_path: str | None, strategy: str = 'diverse', avg_pts_per_cluster: float = 3.0,
             core_fallback: str = 'edge_topup', kmeans_algo: str = 'minibatch') -> None:
    # 1) Load per-label encoded data for the selected subset
    df = _load_encoded_df(res, subset, label)
    label_safe = res.get_label_name(label)

    # 2) Resolve dataset-specific preprocessor and feature schema (encoded numerical feature list)
    _, PreprocessorClass = REGISTRY[res.resources_name]
    pre = PreprocessorClass()
    tech_cols = {'__rowid__', '__subset__'}
    feat_cols = [c for c in getattr(pre, 'encoded_numerical_features', []) if c in df.columns and c not in tech_cols]
    if not feat_cols:
        raise SystemExit("No encoded numerical feature columns found in encoded CSV")
    X = df[feat_cols].to_numpy(copy=False)
    n = len(df)
    logger.info(f"[{label}] strategy={strategy} | rows={n}, dims={len(feat_cols)} | budget={budget}")

    # 3) Determine number of clusters K under the given strategy/budget
    if n_clusters is not None:
        K = int(n_clusters)
    else:
        K = KMeansRepresentativeSelector.compute_default_n_clusters(
            n_rows=n,
            budget=int(budget),
            strategy=str(strategy),
            avg_pts_per_cluster=float(avg_pts_per_cluster),
        )
    if K <= 0:
        raise SystemExit("n_clusters must be > 0")
    K = min(K, n)
    logger.info(f"[{label}] Using n_clusters={K}")

    # 4) Fit (MiniBatch)KMeans and obtain cluster assignments and centers
    ksel = KMeansRepresentativeSelector(n_clusters=K, batch_size=int(batch_size), random_state=42, algorithm=kmeans_algo)
    labels_arr = ksel.fit_predict(X)
    centers = ksel.centers_
    if centers is None:
        raise SystemExit("KMeans centers not available after fit")

    # 5) Select core representatives (closest to centroid per cluster)
    core_reps: list[int] = []
    for c in range(centers.shape[0]):
        idx = np.where(labels_arr == c)[0]
        if idx.size == 0:
            continue
        diff = X[idx] - centers[c]
        d2 = np.einsum('ij,ij->i', diff, diff)
        core_reps.append(int(idx[int(np.argmin(d2))]))
    core_reps = np.array(core_reps, dtype=np.int64)
    logger.debug(f"[{label}] core reps selected: {len(core_reps)} (K={K})")

    if strategy == 'core':
        # 6a) Strategy 'core': use only core reps; under-select -> random or edge-based top-up
        selected_idx = core_reps
        if len(selected_idx) < int(budget):
            if core_fallback == 'random':
                # Fallback to random fill
                logger.info(f"[{label}] core-only selected {len(selected_idx)} < budget {budget}. Falling back to random fill …")
                selected = set(core_reps.tolist())
                remaining = int(budget) - len(selected)
                not_sel = np.array(sorted(list(set(range(n)) - selected)), dtype=np.int64)
                take = min(remaining, not_sel.size)
                logger.debug(f"[{label}] random top-up: requested={remaining}, available={not_sel.size}, taken={take}")
                if take > 0:
                    rng = np.random.RandomState(42)
                    extra = rng.choice(not_sel, size=take, replace=False)
                    selected.update(extra.tolist())
                selected_idx = np.array(sorted(list(selected)), dtype=np.int64)
                logger.info(f"[{label}] Final selected (core+random): {len(selected_idx)} / {budget} | clusters={K}")
            else:
                # Fallback to edge + top-up
                logger.info(f"[{label}] core-only selected {len(selected_idx)} < budget {budget}. Falling back to edge + top-up …")
                # Compute distances and supplement by edge then top-up
                dist_center = KMeansRepresentativeSelector.compute_dist_to_center(X, labels_arr, centers)
                edge_candidates = KMeansRepresentativeSelector.select_edges_per_cluster(dist_center, labels_arr, percentile=float(edge_percentile))
                total_edges = int(sum(len(v) for v in edge_candidates.values())) if edge_candidates else 0
                logger.debug(f"[{label}] edge candidates total={total_edges}")

                # Select core representatives
                selected = set(core_reps.tolist())
                remaining = int(budget) - len(selected)

                # Select edges
                chosen_edges = np.array([], dtype=np.int64)
                if remaining > 0:
                    all_edge = np.concatenate(list(edge_candidates.values())) if edge_candidates else np.array([], dtype=np.int64)
                    take_edge = min(remaining, len(all_edge))
                    chosen_edges = KMeansRepresentativeSelector.distribute_edges(edge_candidates, need=take_edge)
                    if len(chosen_edges) > remaining:
                        chosen_edges = chosen_edges[:remaining]
                    selected.update(chosen_edges.tolist())
                    remaining = int(budget) - len(selected)

                # Select top-up
                if remaining > 0:
                    not_sel = np.array(sorted(list(set(range(n)) - selected)), dtype=np.int64)
                    if not_sel.size > 0:
                        d_order = np.argsort(dist_center[not_sel])
                        topup = not_sel[d_order[:remaining]]
                        logger.debug(f"[{label}] top-up chosen: requested={remaining}, available={not_sel.size}, taken={len(topup)}")
                        selected.update(topup.tolist())

                # Final selection
                selected_idx = np.array(sorted(list(selected)), dtype=np.int64)
                logger.info(f"[{label}] Final selected (core+fallback): {len(selected_idx)} / {budget} | clusters={K}")
        else:
            logger.info(f"[{label}] Final selected (core-only): {len(selected_idx)} / {budget} | clusters={K}")
    else:
        # 6b) Strategy 'diverse': core + edges + top-up; trim if exceeding budget
        dist_center = KMeansRepresentativeSelector.compute_dist_to_center(X, labels_arr, centers)
        edge_candidates = KMeansRepresentativeSelector.select_edges_per_cluster(dist_center, labels_arr, percentile=float(edge_percentile))
        total_edges = int(sum(len(v) for v in edge_candidates.values())) if edge_candidates else 0
        logger.debug(f"[{label}] edge candidates total={total_edges}")

        # Select core representatives
        selected = set(core_reps.tolist())
        remaining = int(budget) - len(selected)

        # Select edges
        chosen_edges = np.array([], dtype=np.int64)
        if remaining > 0:
            all_edge = np.concatenate(list(edge_candidates.values())) if edge_candidates else np.array([], dtype=np.int64)
            take_edge = min(remaining, len(all_edge))
            chosen_edges = KMeansRepresentativeSelector.distribute_edges(edge_candidates, need=take_edge)

            if len(chosen_edges) > remaining:
                chosen_edges = chosen_edges[:remaining]
            logger.debug(f"[{label}] edge chosen: requested={remaining}, available={len(all_edge)}, taken={len(chosen_edges)}")
            selected.update(chosen_edges.tolist())
            remaining = int(budget) - len(selected)

        # Select top-up
        if remaining > 0:
            not_sel = np.array(sorted(list(set(range(n)) - selected)), dtype=np.int64)
            d_order = np.argsort(dist_center[not_sel])
            topup = not_sel[d_order[:remaining]]
            logger.debug(f"[{label}] top-up chosen: requested={remaining}, available={not_sel.size}, taken={len(topup)}")
            selected.update(topup.tolist())

        # Trim if exceeding budget
        if len(selected) > int(budget):
            excess = len(selected) - int(budget)
            core_set = set(core_reps.tolist())
            edge_set = set(chosen_edges.tolist()) if chosen_edges.size > 0 else set()
            edge_only = np.array([i for i in selected if (i in edge_set and i not in core_set)], dtype=np.int64)
            # Drop edge-only first
            if edge_only.size > 0 and excess > 0:
                order = np.argsort(dist_center[edge_only])
                drop_k = min(excess, edge_only.size)
                to_drop = edge_only[order[:drop_k]].tolist()
                selected.difference_update(to_drop)
                excess -= drop_k
                logger.debug(f"[{label}] trim edge-only: dropped={drop_k}, remaining_excess={excess}")
            # Drop top-up second
            if excess > 0:
                topup_only = np.array([i for i in selected if (i not in core_set and i not in edge_set)], dtype=np.int64)
                if topup_only.size > 0:
                    order = np.argsort(dist_center[topup_only])
                    drop_k = min(excess, topup_only.size)
                    to_drop = topup_only[order[:drop_k]].tolist()
                    selected.difference_update(to_drop)
                    excess -= drop_k
                    logger.debug(f"[{label}] trim top-up: dropped={drop_k}, remaining_excess={excess}")
            # Drop core last
            if excess > 0:
                core_only = np.array([i for i in selected if i in core_set], dtype=np.int64)
                if core_only.size > 0:
                    order = np.argsort(dist_center[core_only])
                    drop_k = min(excess, core_only.size)
                    to_drop = core_only[order[:drop_k]].tolist()
                    selected.difference_update(to_drop)
                    logger.debug(f"[{label}] trim core: dropped={drop_k}")
                    logger.warning(f"[{label}] Trimmed {drop_k} core reps to meet budget")

        selected_idx = np.array(sorted(list(selected)), dtype=np.int64)
        logger.info(f"[{label}] Final selected (diverse): {len(selected_idx)} / {budget} | clusters={K}")

    # 7) Build output: selected encoded rows only (preserve __rowid__ and __subset__ if present)
    out_df = df.iloc[selected_idx].copy().reset_index(drop=True)

    # 8) Map selection back to clean_merged via __rowid__ and export compressed CSV
    src_path = res.clean_merged_path_for(subset, label_safe, compressed=False)
    if not os.path.exists(src_path):
        raise SystemExit(f"Source clean_merged not found: {src_path}")
    src_df = pd.read_csv(src_path, low_memory=False)
    if '__rowid__' not in out_df.columns:
        raise SystemExit("Missing __rowid__ in selection; re-run encode to embed row ids.")
    sel_ids = [int(i) for i in out_df['__rowid__'].astype(np.int64).tolist() if 0 <= int(i) < len(src_df)]
    cm_df = src_df.iloc[sel_ids].copy()
    for c in ['__rowid__', '__subset__']:
        if c in cm_df.columns:
            cm_df = cm_df.drop(columns=[c])
    out_p = out_path if out_path is not None else res.clean_merged_path_for(subset, label_safe, compressed=True)
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    cm_df.to_csv(out_p, index=False)
    logger.info(f"[{label}] Saved clean_merged compressed -> {out_p} (rows={len(cm_df)})")


def main():
    parser = argparse.ArgumentParser(description="KMeans compression on encoded CSV for major classes (multi-resource)")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--subset', '-s', type=str, default='train', choices=['train'])
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'], help='Compress all labels or only provided labels')
    parser.add_argument('--labels', '-c', type=str, nargs='+', default=None, help='Label names when --mode label')
    parser.add_argument('--exclude-labels', '-x', type=str, nargs='+', default=None, help='Labels to exclude after selection')
    parser.add_argument('--budget', '-b', type=int, required=True, help='Target compressed size per label')
    parser.add_argument('--n-clusters', '-k', type=int, default=None, help='Number of clusters (default: auto ~ budget)')
    parser.add_argument('--edge-percentile', '-e', type=float, default=95.0, help='Edge percentile per cluster (95..99)')
    parser.add_argument('--batch-size', type=int, default=100000, help='MiniBatchKMeans batch size')
    parser.add_argument('--out-path', '-o', type=str, default=None, help='Override output path for single label')
    parser.add_argument('--n-jobs-labels', '-j', type=int, default=1, help='Parallel jobs across labels when --mode all')
    parser.add_argument('--strategy', '-t', type=str, default='diverse', choices=['core', 'diverse'], help='Selection strategy: core-only or diverse (core+edge)')
    parser.add_argument('--avg-pts-per-cluster', '-p', type=float, default=3.0, help='Diverse mode: average selected points per cluster (K ≈ budget/avg)')
    parser.add_argument('--core-fallback', type=str, default='edge_topup', choices=['edge_topup', 'random'], help='Fallback method when strategy=core under-selects: edge_topup or random')
    parser.add_argument('--kmeans-algo', type=str, default='minibatch', choices=['minibatch', 'full'], help='Clustering algorithm: MiniBatchKMeans (default) or KMeans (full)')
    parser.add_argument('--log-level', '-L', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    ResourcesClass, _ = REGISTRY[args.resource]
    res = ResourcesClass

    # Resolve target labels
    if args.mode == 'label':
        if not args.labels:
            raise SystemExit("--labels is required when --mode label")
        labels = args.labels
        # validate
        valid = set(res.MAJORITY_LABELS)
        invalid = [lb for lb in labels if lb not in valid]
        if invalid:
            raise SystemExit(f"Invalid labels for {res.resources_name}: {invalid}")
    else:
        labels = res.MAJORITY_LABELS

    # Apply excludes if provided
    if args.exclude_labels:
        excludes = set(args.exclude_labels)
        labels = [lb for lb in labels if lb not in excludes]
        logger.info(f"[+] Excluding labels: {sorted(excludes)}")

    if not labels:
        raise SystemExit("No labels to process after applying excludes.")

    n_jobs = int(getattr(args, 'n_jobs_labels', 1) or 1)
    if n_jobs != 1 and len(labels) > 1:
        try:
            from joblib import Parallel, delayed
            Parallel(n_jobs=n_jobs, backend='loky')([
                delayed(_run_one)(
                    res=res,
                    subset=args.subset,
                    label=lb,
                    budget=int(args.budget),
                    n_clusters=args.n_clusters,
                    edge_percentile=float(args.edge_percentile),
                    batch_size=int(args.batch_size),
                    out_path=None,
                    strategy=args.strategy,
                    avg_pts_per_cluster=float(args.avg_pts_per_cluster),
                    core_fallback=args.core_fallback,
                    kmeans_algo=args.kmeans_algo) for lb in labels
            ])
            return
        except Exception as e:
            logger.warning(f"Parallel per-label failed ({e}); falling back to sequential")

    for lb in labels:
        try:
            _run_one(
                res=res,
                subset=args.subset,
                label=lb,
                budget=int(args.budget),
                n_clusters=args.n_clusters,
                edge_percentile=float(args.edge_percentile),
                batch_size=int(args.batch_size),
                out_path=args.out_path if (len(labels) == 1) else None,
                strategy=args.strategy,
                avg_pts_per_cluster=float(args.avg_pts_per_cluster),
                core_fallback=args.core_fallback,
                kmeans_algo=args.kmeans_algo,
            )
        except SystemExit as e:
            logger.warning(f"[{lb}] skipped: {e}")


if __name__ == "__main__":
    main()


