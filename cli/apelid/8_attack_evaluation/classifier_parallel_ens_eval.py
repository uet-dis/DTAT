import os
import sys
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger
from prettytable import PrettyTable


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources  # noqa: E402
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor  # noqa: E402
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor  # noqa: E402
from preprocessing.prepare import PrepareData  # noqa: E402
from training.model import evaluate_classification  # noqa: E402
from helpers.wrapping_helper import (
    load_model_as_wrapper,
    default_inputs as helper_default_inputs,
    find_classifier_model_files,
    predict_with_batching_wrapper,
    combine_weighted,
)  # noqa: E402


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}

MODEL_TYPES = ['xgb', 'dnn', 'tabnet', 'catb', 'bagging', 'gbm']
ATTACK_TYPES = ['zoo', 'deepfool', 'fgsm', 'cw', 'pgd', 'hsja', 'jsma']

# Default ensemble weights (align with ensemble_helper.py; self-normalizing).
# tabnet=0.1 renormalized scheme (HPO-tuned member, offline ensemble test 2026-10).
ENSEMBLE_WEIGHTS = {
    'xgb': 0.27,
    'dnn': 0.09,
    'catb': 0.18,
    'bagging': 0.18,
    'gbm': 0.18,
    'tabnet': 0.1,
}

# Metrics aggregated as mean ± std across seed-config directories.
SEED_AGG_METRICS = ('plain_acc', 'plain_f1', 'plain_precision', 'plain_recall',
                    'adv_acc', 'adv_f1', 'adv_precision', 'adv_recall', 'asr')

# The historical "default" robust checkpoint was trained from pool seed 1337.
# It used to be stored as '<res>_<model>_robust.{pkl,pth}'; it may now live
# under the explicit seed name '<res>_<model>_robust_seed1337.{pkl,pth}'. Both
# are accepted when no explicit model seed is requested.
DEFAULT_ROBUST_SEED = 1337


def _find_model_files(models_dir: str, resource_name: str, *, using_robust: bool = False, using_isolated_robust: bool = False, model_seed: int | None = None) -> dict:
    if not using_robust and not using_isolated_robust and model_seed is None:
        return find_classifier_model_files(models_dir, resource_name)
    # Build file map with preference based on flags
    seed_suffix = f"_seed{model_seed}" if model_seed is not None else ''
    model_files: dict[str, str] = {}
    for model_type in MODEL_TYPES:
        ext = '.pth' if model_type in ('dnn', 'tabnet') else '.pkl'
        stem_iso = f"{resource_name}_{model_type}_isolated_robust{seed_suffix}{ext}"
        stem_robust = f"{resource_name}_{model_type}_robust{seed_suffix}{ext}"
        stem_base = f"{resource_name}_{model_type}{seed_suffix}{ext}"
        p_iso = os.path.join(models_dir, stem_iso)
        p_robust = os.path.join(models_dir, stem_robust)
        p_base = os.path.join(models_dir, stem_base)
        chosen = None
        if using_isolated_robust and os.path.exists(p_iso):
            chosen = p_iso
        elif using_isolated_robust or using_robust:
            if os.path.exists(p_robust):
                chosen = p_robust
            elif model_seed is None:
                # No explicit seed: accept the seed-1337 name for the default robust model.
                p_default = os.path.join(
                    models_dir, f"{resource_name}_{model_type}_robust_seed{DEFAULT_ROBUST_SEED}{ext}")
                if os.path.exists(p_default):
                    chosen = p_default
        if chosen is None and os.path.exists(p_base):
            chosen = p_base
        if chosen is not None:
            model_files[model_type] = chosen
        elif seed_suffix:
            logger.warning(f"[!] No checkpoint for {model_type} at model seed {model_seed}; excluded from ensemble")
    return model_files

def _adv_path_in_dir(adv_dir: str, resource_name: str, model_type: str, attack: str) -> str:
    filename = f"{resource_name}_{model_type}_{attack}_adv.csv"
    return os.path.join(adv_dir, filename)


def _default_adv_path(res, subset: str, model_type: str, attack: str) -> str:
    return _adv_path_in_dir(os.path.join(res.DATA_FOLDER, 'adv_samples', subset),
                            res.resources_name, model_type, attack)


def _fmt_mean_std(mean: float, std: float, *, pct: bool = True) -> str:
    """Render ``mean ± std`` (single value when std is undefined)."""
    if mean != mean:
        return 'nan'
    text = f"{mean * 100:.2f}%" if pct else f"{mean:.4g}"
    if std != std:
        return text
    suffix = f"{std * 100:.2f}%" if pct else f"{std:.3g}"
    return f"{text} ± {suffix}"


def _aggregate_seed_results(results: list[dict], metrics=SEED_AGG_METRICS) -> list[dict]:
    """Collapse per-seed rows into one row per attack with mean/std."""
    grouped: dict[str, list[dict]] = {}
    for row in results:
        grouped.setdefault(row['attack'], []).append(row)

    aggregate_rows = []
    for attack, group in grouped.items():
        row = {
            'attack': attack,
            'n_seeds': len(group),
            'seeds': ','.join(str(item.get('seed', '')) for item in group),
        }
        for metric in metrics:
            values = np.array(
                [item[metric] for item in group
                 if item.get(metric) is not None and item[metric] == item[metric]],
                dtype=float,
            )
            if values.size == 0:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if values.size > 1 else float('nan')
            row[f'{metric}_mean'] = mean
            row[f'{metric}_std'] = std
            row[f'{metric}_mean_pm_std'] = _fmt_mean_std(mean, std)
        aggregate_rows.append(row)
    return aggregate_rows


def _parallel_ensemble_predict(model_files: dict, X: np.ndarray, *, num_class: int, input_dim: int,
                               clip_values: tuple | None,
                               device: str = 'auto', batch_size: int = -1, max_workers: int = 4):
    logger.debug(f"[+] Parallel ensemble prediction ({len(model_files)} models; batch_size={batch_size}, max_workers={max_workers})")
    tasks = [(mt, path) for mt, path in model_files.items()]
    results: dict[str, np.ndarray] = {}

    def _worker(mt: str, path: str):
        wrapper = load_model_as_wrapper(
            mt,
            path,
            num_classes=num_class,
            input_dim=input_dim,
            clip_values=clip_values,
            device=device,
        )
        return mt, predict_with_batching_wrapper(wrapper, mt, X, num_class, batch_size)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2mt = {ex.submit(_worker, *t): t[0] for t in tasks}
        for fut in as_completed(fut2mt):
            mt = fut2mt[fut]
            try:
                mt_out, proba = fut.result()
                if proba is not None:
                    results[mt_out] = proba
                else:
                    logger.warning(f"[!] {mt_out} prediction failed")
            except Exception as e:
                logger.error(f"[!] {mt} prediction error: {e}")

    if not results:
        raise RuntimeError("No models produced valid predictions")
    return combine_weighted(results, num_class, ENSEMBLE_WEIGHTS)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ART-wrapped ensemble under per-model adversarial inputs")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--models-dir', type=str, default=None, help="Directory containing trained models")
    parser.add_argument('--data-in', type=str, default=None, help="Plain test CSV (shared for all models)")
    parser.add_argument('--attack', '-a', type=str, nargs='+', required=True, choices=ATTACK_TYPES,
                        help="Attack name(s) to evaluate; each attack resolves per-model adv inputs")
    parser.add_argument('--exclude-models', type=str, nargs='*', default=[], help="Models to exclude from ensemble")
    parser.add_argument('--using-robust', action='store_true', help="Prefer robust models (suffix _robust) with fallback to base")
    parser.add_argument('--using-isolated-robust', action='store_true', help="Prefer isolated robust models (suffix _isolated_robust) with fallback to _robust then base")
    parser.add_argument('--model-seeds', type=int, nargs='+', default=None,
                        help="Evaluate ensembles whose members are checkpoints trained from pool seeds, i.e. "
                             "'<res>_<model>_robust_seed<S>.<ext>' (implies --using-robust). With two or more "
                             "values results are reported as mean ± std across model seeds. Combine with "
                             "--seed-dirs to also vary the adversarial data source.")
    parser.add_argument('--max-workers', type=int, default=4, help="Maximum parallel workers")
    parser.add_argument('--batch-size', type=int, default=-1, help="Batch size for prediction (-1 for full dataset)")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--adv-subset', type=str, default='test',
                        help="Subfolder under resources/<DS>/adv_samples/ holding the adversarial CSVs (e.g. 'test_v2')")
    parser.add_argument('--adv-fallback-src', type=str, default='dnn', choices=['dnn', 'tabnet'],
                        help="Source model whose white-box (gradient) adv CSVs are used for members "
                             "without their own adv file (e.g. gradient attacks on trees). "
                             "Default: dnn (backward compatible)")
    parser.add_argument('--seed-dirs', type=str, nargs='+', default=None,
                        help="Additional seed-config directories containing same-named adversarial CSVs. "
                             "Each source (primary subset dir + seed dirs) is evaluated separately and, "
                             "with two or more sources, results are reported as mean ± std across seeds")
    parser.add_argument('--seed-label', type=str, default=None,
                        help="Label for the primary (non-seed-dir) source; default 'seed_0' when "
                             "--seed-dirs is given, otherwise 'base'")
    parser.add_argument('--report-out', type=str, default=None,
                        help="Optional CSV with the per-seed evaluation rows")
    parser.add_argument('--agg-report-out', type=str, default=None,
                        help="Optional CSV with the seed-aggregated mean/std table")
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Seed-trained members live under the robust naming scheme; enable it
    # automatically unless the caller asked for isolated-robust explicitly.
    if args.model_seeds and not (args.using_robust or args.using_isolated_robust):
        args.using_robust = True
        logger.info("[i] --model-seeds implies robust checkpoints; enabling --using-robust")

    # Additional seed-config sources: each contributes same-named adv CSVs.
    seed_sources = []
    for seed_dir in (args.seed_dirs or []):
        seed_path = os.path.abspath(seed_dir)
        if not os.path.isdir(seed_path):
            raise SystemExit(f"seed directory not found: {seed_dir}")
        seed_sources.append((os.path.basename(os.path.normpath(seed_path)), seed_path))
    primary_seed_label = args.seed_label or ('seed_0' if seed_sources else 'base')

    # Setup resource and preprocessor
    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass
    pre = PreprocessorClass()

    # Preload ART modules to avoid ModuleLock deadlocks in threads
    try:
        from art.estimators.classification import (
            PyTorchClassifier, XGBoostClassifier, CatBoostARTClassifier, SklearnClassifier
        )  # noqa: F401
        from art.preprocessing.standardisation_mean_std import StandardisationMeanStd  # noqa: F401
        logger.debug("[+] Preloaded ART modules for thread safety")
    except Exception as e:
        logger.debug(f"[!] ART preload skipped or not available: {e}")

    # Class names from configs
    class_names = sorted(res.MAJORITY_LABELS + res.MINORITY_LABELS)

    # Paths
    if args.models_dir is None:
        args.models_dir = os.path.join(res.DATA_FOLDER, 'models')

    # Default test data
    if args.data_in is None:
        _, default_test = helper_default_inputs(res)
        args.data_in = default_test

    # Load plain data
    if not os.path.exists(args.data_in):
        raise SystemExit(f"Plain test CSV not found: {args.data_in}")
    logger.info(f"[+] Loading plain test data: {args.data_in}")
    df_plain = pd.read_csv(args.data_in, low_memory=False)

    if not pre.load_encoders(fixed_label_encoder=True):
        raise SystemExit("Encoders not found. Train models first.")

    X_plain, y_plain, meta_plain = PrepareData.prepare_input_data(df_plain, pre, include_label=True)

    model_seeds = args.model_seeds or [None]
    if len(model_seeds) > 1:
        logger.info(
            "[i] Evaluating %d ensemble model seeds: %s (per-seed results aggregated as mean ± std)",
            len(model_seeds), model_seeds,
        )
    results = []
    for model_seed in model_seeds:
        # Discover ensemble model files
        model_files = _find_model_files(
            args.models_dir,
            res.resources_name,
            using_robust=args.using_robust,
            using_isolated_robust=args.using_isolated_robust,
            model_seed=model_seed,
        )
        # Apply exclusions
        for ex in args.exclude_models:
            if ex in model_files:
                del model_files[ex]
                logger.info(f"[+] Excluded model: {ex}")
        if 'gbm' not in model_files:
            logger.warning("[!] 'gbm' model not found")
        if not model_files:
            logger.error("[!] No models to ensemble after exclusions for model seed %s; skipping", model_seed)
            continue
        logger.info(f"[+] Ensemble models: {list(model_files.keys())}")

        # Baseline: ensemble on plain
        try:
            _, ensemble_pred_plain = _parallel_ensemble_predict(
                model_files,
                X_plain,
                num_class=len(class_names),
                input_dim=X_plain.shape[1],
                clip_values=meta_plain['clip_values'],
                device=args.device,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
            )
            metrics_plain = evaluate_classification(y_plain, ensemble_pred_plain, class_names, metric='weighted')
            plain_acc = float(metrics_plain.get('accuracy', 0.0))
            plain_f1 = float(metrics_plain.get('f1', 0.0))
            plain_precision = float(metrics_plain.get('precision', 0.0))
            plain_recall = float(metrics_plain.get('recall', 0.0))
        except Exception as e:
            logger.error(f"[!] Plain ensemble evaluation failed: {e}")
            continue

        # Per-attack evaluation with per-model adv streams. Each seed source
        # (primary subset dir + --seed-dirs) is evaluated separately; a source
        # missing any per-model adv CSV is skipped for that attack.
        seed_candidates = [(primary_seed_label, os.path.join(res.DATA_FOLDER, 'adv_samples', args.adv_subset))]
        seed_candidates += seed_sources

        for atk in args.attack:
            for seed_label, adv_dir in seed_candidates:
                # Resolve per-model adv CSVs (fallback to --adv-fallback-src if missing)
                fallback_src = args.adv_fallback_src
                adv_paths: dict[str, str] = {}
                missing = []
                for mt in model_files.keys():
                    p = _adv_path_in_dir(adv_dir, res.resources_name, mt, atk)
                    if not os.path.exists(p) and mt != fallback_src:
                        p_fb = _adv_path_in_dir(adv_dir, res.resources_name, fallback_src, atk)
                        if os.path.exists(p_fb):
                            logger.debug(f"[+] Fallback to {fallback_src.upper()} adv for {mt}/{atk} [{seed_label}]: {p_fb}")
                            p = p_fb
                    if not os.path.exists(p):
                        missing.append(mt)
                    adv_paths[mt] = p
                if missing:
                    if seed_label == primary_seed_label:
                        logger.warning(f"[!] Skipping attack {atk} [{seed_label}]: missing adv CSV for {missing}")
                    else:
                        logger.debug(f"[+] Seed source '{seed_label}' has no adv CSV for {atk} (missing {missing}); skipping")
                    continue

                # Load and prepare per-model adv datasets in parallel
                def _load_adv(path: str):
                    if not os.path.exists(path):
                        return None
                    df = pd.read_csv(path, low_memory=False)
                    X, y, _ = PrepareData.prepare_input_data(df, pre, include_label=True)
                    return X, y

                with ThreadPoolExecutor(max_workers=min(len(adv_paths), args.max_workers)) as ex:
                    futs = {ex.submit(_load_adv, p): mt for mt, p in adv_paths.items()}
                    adv_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                    for fut in as_completed(futs):
                        mt = futs[fut]
                        try:
                            out = fut.result()
                            if out is not None:
                                adv_data[mt] = out
                            else:
                                logger.warning(f"[!] Missing adv CSV for {mt}/{atk} [{seed_label}]: {adv_paths[mt]}")
                        except Exception as e:
                            logger.warning(f"[!] Failed loading adv CSV for {mt}/{atk} [{seed_label}]: {e}")

                # Require all models to have adv data; otherwise skip this attack
                if set(adv_data.keys()) != set(model_files.keys()):
                    logger.warning(f"[!] Skipping attack {atk} [{seed_label}]: not all models have adv data ({sorted(adv_data.keys())} != {sorted(model_files.keys())})")
                    continue

                # Predict per model on its own adv X and ensemble the probabilities
                def _predict_model(mt: str, path: str, X: np.ndarray):
                    wrapper = load_model_as_wrapper(
                        mt,
                        path,
                        num_classes=len(class_names),
                        input_dim=X.shape[1],
                        clip_values=meta_plain['clip_values'],
                        device=args.device,
                    )
                    proba = predict_with_batching_wrapper(wrapper, mt, X, len(class_names), args.batch_size)
                    return proba

                with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                    futs = {ex.submit(_predict_model, mt, model_files[mt], adv_data[mt][0]): mt for mt in model_files.keys()}
                    proba_map: dict[str, np.ndarray] = {}
                    for fut in as_completed(futs):
                        mt = futs[fut]
                        try:
                            proba = fut.result()
                            if proba is not None:
                                proba_map[mt] = proba
                            else:
                                logger.warning(f"[!] {mt} adv prediction failed for attack {atk} [{seed_label}]")
                        except Exception as e:
                            logger.error(f"[!] {mt} adv prediction error ({atk} [{seed_label}]): {e}")

                if set(proba_map.keys()) != set(model_files.keys()):
                    logger.warning(f"[!] Skipping attack {atk} [{seed_label}]: some models failed to predict")
                    continue

                ensemble_proba_adv, ensemble_pred_adv = combine_weighted(proba_map, len(class_names), ENSEMBLE_WEIGHTS)
                y_adv_ref = adv_data[next(iter(adv_data))][1]
                metrics_adv = evaluate_classification(y_adv_ref, ensemble_pred_adv, class_names, metric='weighted')

                # ASR: restricted to samples correctly predicted by plain ensemble; require aligned length
                try:
                    asr_val = None
                    if ensemble_pred_plain.shape[0] != ensemble_pred_adv.shape[0]:
                        logger.warning(f"[!] Plain and adv sample counts differ: {ensemble_pred_plain.shape[0]} vs {ensemble_pred_adv.shape[0]}")
                    n = min(ensemble_pred_plain.shape[0], ensemble_pred_adv.shape[0])
                    # For reference labels, use plain y when available; if lengths differ, trim
                    y_ref = y_plain[:n]
                    plain_correct_mask = (ensemble_pred_plain[:n] == y_ref)
                    denom = int(np.sum(plain_correct_mask))
                    if denom == 0:
                        asr_val = 0.0
                    else:
                        success_mask = (ensemble_pred_adv[:n] != y_ref) & plain_correct_mask
                        num_success = int(np.sum(success_mask))
                        asr_val = float(num_success) / float(denom)
                except Exception as e:
                    logger.error(f"[!] Failed to compute ASR for [{atk}] [{seed_label}]: {e}")
                    asr_val = None

                results.append({
                    'attack': atk,
                    'model_seed': model_seed,
                    'seed': seed_label,
                    'plain_acc': plain_acc,
                    'plain_f1': plain_f1,
                    'plain_precision': plain_precision,
                    'plain_recall': plain_recall,
                    'adv_acc': float(metrics_adv.get('accuracy', 0.0)),
                    'adv_f1': float(metrics_adv.get('f1', 0.0)),
                    'adv_precision': float(metrics_adv.get('precision', 0.0)),
                    'adv_recall': float(metrics_adv.get('recall', 0.0)),
                    'asr': asr_val if asr_val is not None else float('nan'),
                })

    # Output tables
    if results:
        table = PrettyTable()
        show_seed = bool(seed_sources)
        show_model_seed = len(model_seeds) > 1
        table.field_names = (
            (['M.Seed'] if show_model_seed else []) +
            (['Seed'] if show_seed else []) +
            ['Attack/File', 'Plain Acc', 'Plain F1', 'Adv Acc', 'Adv F1', 'ASR']
        )
        for r in results:
            row = ([r.get('model_seed', '')] if show_model_seed else [])
            row += ([r.get('seed', '')] if show_seed else [])
            row += [
                r['attack'],
                f"{r['plain_acc']*100:.2f}%",
                f"{r['plain_f1']*100:.2f}%",
                f"{r['adv_acc']*100:.2f}%",
                f"{r['adv_f1']*100:.2f}%",
                (("{:.2f}%".format(r['asr']*100)) if r['asr'] == r['asr'] else 'nan'),
            ]
            table.add_row(row)
        logger.debug("\n" + table.get_string())

        # Per-seed detail CSV (if requested)
        if args.report_out:
            pd.DataFrame(results).to_csv(args.report_out, index=False)
            logger.info(f"[+] Per-seed evaluation saved: {args.report_out}")

    # Seed-aggregated data: written to CSV when requested; rendered into the
    # compact 2D table below (no separate console table).
    aggregate = _aggregate_seed_results(results) if ((seed_sources or len(model_seeds) > 1) and results) else []
    if args.agg_report_out and aggregate:
        pd.DataFrame(aggregate).to_csv(args.agg_report_out, index=False)
        logger.info(f"[+] Seed-aggregated evaluation saved: {args.agg_report_out}")

    # Compact 2D table: rows = original + attacks; columns = Acc, F1, Precision, Recall, ASR.
    # With --seed-dirs each cell shows mean ± std across the evaluated sources.
    if results:
        aggregate_map = {row['attack']: row for row in aggregate}
        primary_results = [
            r for r in results
            if not seed_sources or r.get('seed', primary_seed_label) == primary_seed_label
        ] or results
        primary_by_attack = {r['attack']: r for r in primary_results}
        tags_order = ['original'] + sorted(primary_by_attack.keys())

        table2 = PrettyTable()
        table2.field_names = ['Attack/File', 'Acc', 'F1', 'Precision', 'Recall', 'ASR']
        for tag in tags_order:
            if tag == 'original':
                def _agg_plain(field):
                    uniq = {r.get('model_seed'): r.get(field, float('nan')) for r in results}
                    vals = [v for v in uniq.values() if v == v]
                    if not vals:
                        return 'nan'
                    if len(vals) == 1:
                        return f"{vals[0] * 100:.2f}%"
                    arr = np.array(vals, dtype=float)
                    return _fmt_mean_std(float(arr.mean()), float(arr.std(ddof=1)))

                table2.add_row([
                    'original',
                    _agg_plain('plain_acc'),
                    _agg_plain('plain_f1'),
                    _agg_plain('plain_precision'),
                    _agg_plain('plain_recall'),
                    '--',
                ])
                continue
            agg = aggregate_map.get(tag)
            if agg is not None:
                table2.add_row([
                    tag,
                    agg.get('adv_acc_mean_pm_std', 'nan'),
                    agg.get('adv_f1_mean_pm_std', 'nan'),
                    agg.get('adv_precision_mean_pm_std', 'nan'),
                    agg.get('adv_recall_mean_pm_std', 'nan'),
                    agg.get('asr_mean_pm_std', 'nan'),
                ])
                continue
            r = primary_by_attack.get(tag)
            if r is None:
                continue
            asr_txt = ("{:.2f}%".format(r['asr']*100) if r['asr'] == r['asr'] else 'nan')
            table2.add_row([
                tag,
                f"{r['adv_acc']*100:.2f}%",
                f"{r['adv_f1']*100:.2f}%",
                f"{r['adv_precision']*100:.2f}%",
                f"{r['adv_recall']*100:.2f}%",
                asr_txt,
            ])
        logger.info("\n" + table2.get_string())
        if seed_sources or len(model_seeds) > 1:
            parts = []
            if len(model_seeds) > 1:
                parts.append(f"{len(model_seeds)} model seeds ({', '.join(str(s) for s in model_seeds)})")
            if seed_sources:
                parts.append(f"{len(seed_sources) + 1} data sources (primary '{primary_seed_label}' + "
                             + ', '.join(label for label, _ in seed_sources) + ")")
            logger.info(
                "[i] Cells with '±' are mean ± std across %s; values available from only one "
                "repeat are shown as plain numbers.",
                ' x '.join(parts),
            )


if __name__ == "__main__":
    main()


