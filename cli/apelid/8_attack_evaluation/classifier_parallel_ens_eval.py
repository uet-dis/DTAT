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

MODEL_TYPES = ['xgb', 'dnn', 'catb', 'bagging', 'histgbm', 'gbm']
ATTACK_TYPES = ['zoo', 'deepfool', 'fgsm', 'cw', 'pgd', 'hsja', 'jsma']

# Default ensemble weights (align with classifier_parallel_ensemble.py)
ENSEMBLE_WEIGHTS = {
    'xgb': 0.3,
    'dnn': 0.1,
    'catb': 0.2,
    'bagging': 0.2,
    'histgbm': 0.2,
    'gbm': 0.2,
}


def _find_model_files(models_dir: str, resource_name: str, *, using_robust: bool = False, using_isolated_robust: bool = False) -> dict:
    if not using_robust and not using_isolated_robust:
        return find_classifier_model_files(models_dir, resource_name)
    # Build file map with preference based on flags
    model_files: dict[str, str] = {}
    for model_type in MODEL_TYPES:
        ext = '.pth' if model_type == 'dnn' else '.pkl'
        stem_iso = f"{resource_name}_{model_type}_isolated_robust{ext}"
        stem_robust = f"{resource_name}_{model_type}_robust{ext}"
        stem_base = f"{resource_name}_{model_type}{ext}"
        p_iso = os.path.join(models_dir, stem_iso)
        p_robust = os.path.join(models_dir, stem_robust)
        p_base = os.path.join(models_dir, stem_base)
        chosen = None
        if using_isolated_robust and os.path.exists(p_iso):
            chosen = p_iso
        elif (using_isolated_robust or using_robust) and os.path.exists(p_robust):
            chosen = p_robust
        elif os.path.exists(p_base):
            chosen = p_base
        if chosen is not None:
            model_files[model_type] = chosen
    return model_files

def _default_adv_path(res, subset: str, model_type: str, attack: str) -> str:
    base_dir = os.path.join(res.DATA_FOLDER, 'adv_samples', subset)
    filename = f"{res.resources_name}_{model_type}_{attack}_adv.csv"
    return os.path.join(base_dir, filename)


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
    parser.add_argument('--using-histgbm', action='store_true', help="Use histgbm instead of gbm in ensemble")
    parser.add_argument('--using-robust', action='store_true', help="Prefer robust models (suffix _robust) with fallback to base")
    parser.add_argument('--using-isolated-robust', action='store_true', help="Prefer isolated robust models (suffix _isolated_robust) with fallback to _robust then base")
    parser.add_argument('--max-workers', type=int, default=4, help="Maximum parallel workers")
    parser.add_argument('--batch-size', type=int, default=-1, help="Batch size for prediction (-1 for full dataset)")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    # Feature Squeezing params for wrappers
    parser.add_argument('--fs-enable', action='store_true', help='Enable Feature Squeezing preprocessor')
    parser.add_argument('--fs-bit-depth', type=int, default=8, help='Bit depth for Feature Squeezing')
    parser.add_argument('--fs-config', type=str, nargs='*', default=None,
                        help='Per-model FS bit depth, e.g. dnn:6 xgb:8')
    args = parser.parse_args()

    setup_logging(args.log_level)

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

    # Discover ensemble model files
    model_files = _find_model_files(
        args.models_dir,
        res.resources_name,
        using_robust=args.using_robust,
        using_isolated_robust=args.using_isolated_robust,
    )
    # Apply exclusions and gbm/histgbm preference
    for ex in args.exclude_models:
        if ex in model_files:
            del model_files[ex]
            logger.info(f"[+] Excluded model: {ex}")
    if args.using_histgbm:
        if 'gbm' in model_files:
            del model_files['gbm']
            logger.info("[+] using_histgbm is ON: disabled 'gbm', keeping 'histgbm' if present")
        if 'histgbm' not in model_files:
            logger.warning("[!] using_histgbm is ON but 'histgbm' model not found")
    else:
        if 'histgbm' in model_files:
            del model_files['histgbm']
            logger.info("[+] using_histgbm is OFF: disabled 'histgbm', using 'gbm' if present")
        if 'gbm' not in model_files:
            logger.warning("[!] using_histgbm is OFF but 'gbm' model not found")
    if not model_files:
        raise SystemExit("No models to ensemble after exclusions")
    logger.info(f"[+] Ensemble models: {list(model_files.keys())}")

    # Parse per-model FS config
    fs_map = {}
    if args.fs_config:
        for item in args.fs_config:
            try:
                mt, depth = item.split(':', 1)
                mt = mt.strip()
                if mt not in MODEL_TYPES:
                    continue
                fs_map[mt] = int(depth)
            except Exception:
                continue

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
        metrics_plain = evaluate_classification(y_plain, ensemble_pred_plain, class_names)
        plain_acc = float(metrics_plain.get('accuracy', 0.0))
        plain_f1 = float(metrics_plain.get('f1', 0.0))
        plain_precision = float(metrics_plain.get('precision', 0.0))
        plain_recall = float(metrics_plain.get('recall', 0.0))
    except Exception as e:
        logger.error(f"[!] Plain ensemble evaluation failed: {e}")
        raise

    # Per-attack evaluation with per-model adv streams
    results = []
    for atk in args.attack:
        # Resolve per-model adv CSVs (fallback to dnn if missing)
        adv_paths: dict[str, str] = {}
        for mt in model_files.keys():
            p = _default_adv_path(res, subset='test', model_type=mt, attack=atk)
            if not os.path.exists(p) and mt != 'dnn':
                p_dnn = _default_adv_path(res, subset='test', model_type='dnn', attack=atk)
                if os.path.exists(p_dnn):
                    logger.debug(f"[+] Fallback to DNN adv for {mt}/{atk}: {p_dnn}")
                    p = p_dnn
            adv_paths[mt] = p

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
                        logger.warning(f"[!] Missing adv CSV for {mt}/{atk}: {adv_paths[mt]}")
                except Exception as e:
                    logger.warning(f"[!] Failed loading adv CSV for {mt}/{atk}: {e}")

        # Require all models to have adv data; otherwise skip this attack
        if set(adv_data.keys()) != set(model_files.keys()):
            logger.warning(f"[!] Skipping attack {atk}: not all models have adv data ({sorted(adv_data.keys())} != {sorted(model_files.keys())})")
            continue

        # Predict per model on its own adv X and ensemble the probabilities
        def _predict_model(mt: str, path: str, X: np.ndarray):
            wrapper_params = {}
            if args.fs_enable:
                bit_depth = int(fs_map.get(mt, args.fs_bit_depth))
                wrapper_params.update({
                    'feature_squeezing': True,
                    'fs_bit_depth': bit_depth,
                })
            wrapper = load_model_as_wrapper(
                mt,
                path,
                num_classes=len(class_names),
                input_dim=X.shape[1],
                clip_values=meta_plain['clip_values'],
                device=args.device,
                wrapper_params=wrapper_params,
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
                        logger.warning(f"[!] {mt} adv prediction failed for attack {atk}")
                except Exception as e:
                    logger.error(f"[!] {mt} adv prediction error ({atk}): {e}")

        if set(proba_map.keys()) != set(model_files.keys()):
            logger.warning(f"[!] Skipping attack {atk}: some models failed to predict")
            continue

        ensemble_proba_adv, ensemble_pred_adv = combine_weighted(proba_map, len(class_names), ENSEMBLE_WEIGHTS)
        y_adv_ref = adv_data[next(iter(adv_data))][1]
        metrics_adv = evaluate_classification(y_adv_ref, ensemble_pred_adv, class_names)

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
            logger.error(f"[!] Failed to compute ASR for [{atk}]: {e}")
            asr_val = None

        results.append({
            'attack': atk,
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
        table.field_names = ['Attack/File', 'Plain Acc', 'Plain F1', 'Adv Acc', 'Adv F1', 'ASR']
        for r in results:
            table.add_row([
                r['attack'],
                f"{r['plain_acc']*100:.2f}%",
                f"{r['plain_f1']*100:.2f}%",
                f"{r['adv_acc']*100:.2f}%",
                f"{r['adv_f1']*100:.2f}%",
                (("{:.2f}%".format(r['asr']*100)) if r['asr'] == r['asr'] else 'nan'),
            ])
        logger.debug("\n" + table.get_string())

        # Compact 2D table: rows = original + attacks; columns = Acc, F1, Precision, Recall, ASR
        table2 = PrettyTable()
        table2.field_names = ['Attack/File', 'Acc', 'F1', 'Precision', 'Recall', 'ASR']
        # Plain baseline
        table2.add_row([
            'original',
            f"{results[0]['plain_acc']*100:.2f}%",
            f"{results[0]['plain_f1']*100:.2f}%",
            f"{results[0]['plain_precision']*100:.2f}%",
            f"{results[0]['plain_recall']*100:.2f}%",
            '--',
        ])
        # Per-attack
        for r in results:
            asr_txt = ("{:.2f}%".format(r['asr']*100) if r['asr'] == r['asr'] else 'nan')
            table2.add_row([
                r['attack'],
                f"{r['adv_acc']*100:.2f}%",
                f"{r['adv_f1']*100:.2f}%",
                f"{r['adv_precision']*100:.2f}%",
                f"{r['adv_recall']*100:.2f}%",
                asr_txt,
            ])
        logger.info("\n" + table2.get_string())


if __name__ == "__main__":
    main()


