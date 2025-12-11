import os
import sys
import argparse
import json

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
from helpers.wrapping_helper import load_model_as_wrapper, default_inputs as helper_default_inputs  # noqa: E402


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}

MODEL_TYPES = ['xgb', 'dnn', 'catb', 'bagging', 'histgbm', 'gbm']
ATTACK_TYPES = ['zoo', 'deepfool', 'fgsm', 'cw', 'pgd', 'hsja', 'jsma']



def _resolve_model_path(models_dir: str, resource_name: str, model_type: str, *, using_robust: bool = False) -> str:
    ext = '.pth' if model_type == 'dnn' else '.pkl'
    stem = f"{resource_name}_{model_type}{'_robust' if using_robust else ''}"
    p = os.path.join(models_dir, f"{stem}{ext}")
    if not os.path.exists(p):
        # Fallback to non-robust if robust requested but missing
        if using_robust:
            fallback = os.path.join(models_dir, f"{resource_name}_{model_type}{ext}")
            if os.path.exists(fallback):
                logger.warning(f"[!] Robust model missing for {model_type}, fallback to base: {fallback}")
                return fallback
        raise SystemExit(f"Model file not found: {p}")
    return p


def _default_adv_path(res, subset: str, model_type: str, attack: str) -> str:
    base_dir = os.path.join(res.DATA_FOLDER, 'adv_samples', subset)
    filename = f"{res.resources_name}_{model_type}_{attack}_adv.csv"
    return os.path.join(base_dir, filename)


def _predict_indices(wrapper, X: np.ndarray) -> np.ndarray:
    y = wrapper.predict(X)
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1)
    return y


def main():
    parser = argparse.ArgumentParser(description="Evaluate models (ART wrappers) on plain and adversarial (test subset) samples")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--model', '-m', type=str, nargs='+', required=True, choices=MODEL_TYPES,
                        help="Model type(s) to evaluate; support multiple values")
    parser.add_argument('--attack', '-a', type=str, nargs='+', required=False, choices=ATTACK_TYPES,
                        help="Attack name(s). Required if --adv-in not provided. Support multiple values.")
    parser.add_argument('--plain-in', type=str, default=None, help="Plain test CSV. Default = dataset default test")
    parser.add_argument('--adv-in', type=str, nargs='+', default=None,
                        help="Adversarial test CSV path(s). If provided, overrides --attack. Support multiple values.")
    parser.add_argument('--models-dir', type=str, default=None, help="Directory containing trained models")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--using-histgbm', action='store_true',
                        help="Use HistGradientBoosting (histgbm) instead of GBM (default uses gbm)")
    parser.add_argument('--using-robust', action='store_true', help="Use robust models (filename suffix _robust)")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    # Feature Squeezing params for wrappers
    parser.add_argument('--fs-enable', action='store_true', help='Enable Feature Squeezing preprocessor')
    parser.add_argument('--fs-bit-depth', type=int, default=8, help='Bit depth for Feature Squeezing')
    parser.add_argument('--fs-config', type=str, nargs='*', default=None,
                        help='Per-model FS bit depth, e.g. dnn:6 xgb:8')
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Resource and preprocessor
    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass
    pre = PreprocessorClass()

    # Defaults
    if args.models_dir is None:
        args.models_dir = os.path.join(res.DATA_FOLDER, 'models')

    # Resolve inputs
    if args.plain_in is None:
        _, default_test = helper_default_inputs(res)
        args.plain_in = default_test

    if not os.path.exists(args.plain_in):
        raise SystemExit(f"Plain test CSV not found: {args.plain_in}")
    logger.debug(f"[+] Plain test CSV: {args.plain_in}")

    # Load encoders
    if not pre.load_encoders(fixed_label_encoder=True):
        raise SystemExit("Encoders not found. Train models first.")

    # Prepare data (plain)
    df_plain = pd.read_csv(args.plain_in, low_memory=False)
    X_plain, y_plain, meta_plain = PrepareData.prepare_input_data(df_plain, pre, include_label=True)

    # Class names from configs (stable order)
    class_names = sorted(res.MAJORITY_LABELS + res.MINORITY_LABELS)

    # Prepare requested models and apply gbm/histgbm preference
    models_requested = args.model if isinstance(args.model, list) else [args.model]
    models_eval = list(models_requested)
    if args.using_histgbm:
        if 'gbm' in models_eval:
            models_eval = [m for m in models_eval if m != 'gbm']
            logger.info("[+] using_histgbm is ON: disabled 'gbm'")
    else:
        if 'histgbm' in models_eval:
            models_eval = [m for m in models_eval if m != 'histgbm']
            logger.info("[+] using_histgbm is OFF: disabled 'histgbm'")

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

    results = []
    for model_type in models_eval:
        # Load model path and wrapper
        model_path = _resolve_model_path(args.models_dir, res.resources_name, model_type, using_robust=args.using_robust)
        logger.debug(f"[+] Loading wrapper: type={model_type}, path={model_path}")
        wrapper_params = {}
        if args.fs_enable:
            bit_depth = int(fs_map.get(model_type, args.fs_bit_depth))
            wrapper_params.update({
                'feature_squeezing': True,
                'fs_bit_depth': bit_depth,
            })
        wrapper = load_model_as_wrapper(
            model_type,
            model_path,
            num_classes=len(class_names),
            input_dim=X_plain.shape[1],
            clip_values=meta_plain['clip_values'],
            device=args.device,
            wrapper_params=wrapper_params,
        )

        # Evaluate on plain test
        logger.debug("[+] Evaluating on plain test set")
        y_plain_pred = _predict_indices(wrapper, X_plain)
        metrics_plain = evaluate_classification(y_plain, y_plain_pred, class_names)
        plain_acc = float(metrics_plain.get('accuracy', 0.0))
        plain_f1 = float(metrics_plain.get('f1', 0.0))

        # Resolve adversarial inputs for this model
        tasks = []
        if args.adv_in is not None:
            for p in args.adv_in:
                tasks.append((p, os.path.basename(p), False))
        else:
            if not args.attack:
                raise SystemExit("When --adv-in is not provided, --attack is required to resolve default adversarial CSV path(s).")
            for atk in args.attack:
                tasks.append((_default_adv_path(res, subset='test', model_type=model_type, attack=atk), atk, True))

        # Evaluate on adversarial inputs (multiple)
        for adv_path, tag, is_resolved in tasks:
            # Fallback: if resolved path missing and model != dnn, try DNN's adv file for same attack
            adv_path_use = adv_path
            if not os.path.exists(adv_path_use):
                if is_resolved and model_type != 'dnn':
                    try:
                        fallback_path = _default_adv_path(res, subset='test', model_type='dnn', attack=tag)
                        if os.path.exists(fallback_path):
                            logger.debug(f"[+] Missing adv file for model={model_type}, attack={tag}. Fallback to DNN: {fallback_path}")
                            adv_path_use = fallback_path
                    except Exception:
                        pass
            if not os.path.exists(adv_path_use):
                logger.warning(f"[!] Adversarial test CSV not found: {adv_path}; skipping")
                continue
            logger.debug(f"[+] Adversarial test CSV: {adv_path_use}")

            # Prepare data (adversarial)
            df_adv = pd.read_csv(adv_path_use, low_memory=False)
            X_adv, y_adv, meta_adv = PrepareData.prepare_input_data(df_adv, pre, include_label=True)

            # Evaluate on adversarial test
            logger.debug(f"[+] Evaluating on adversarial test set [{tag}]")
            y_adv_pred = _predict_indices(wrapper, X_adv)
            metrics_adv = evaluate_classification(y_adv, y_adv_pred, class_names)

            # Compute ASR restricted to plain-correct samples
            try:
                asr_val = None
                if X_plain.shape[0] != X_adv.shape[0]:
                    logger.warning(f"[!] Plain and adversarial sample counts differ: {X_plain.shape[0]} vs {X_adv.shape[0]}")
                n = min(X_plain.shape[0], X_adv.shape[0])
                y_ref = y_plain[:n]
                plain_correct_mask = (y_plain_pred[:n] == y_ref)
                denom = int(np.sum(plain_correct_mask))
                if denom == 0:
                    asr_val = 0.0
                else:
                    success_mask = (y_adv_pred[:n] != y_ref) & plain_correct_mask
                    num_success = int(np.sum(success_mask))
                    asr_val = float(num_success) / float(denom)
            except Exception as e:
                logger.error(f"[!] Failed to compute ASR for [{tag}]: {e}")
                asr_val = None

            # Collect summary row
            adv_acc = float(metrics_adv.get('accuracy', 0.0))
            adv_f1 = float(metrics_adv.get('f1', 0.0))
            results.append({
                'model': model_type,
                'attack': tag,
                'plain_acc': plain_acc,
                'plain_f1': plain_f1,
                'adv_acc': adv_acc,
                'adv_f1': adv_f1,
                'asr': asr_val if asr_val is not None else float('nan'),
            })

    # Print final pretty table
    if results:
        table = PrettyTable()
        table.field_names = [
            'Model', 'Attack/File', 'Plain Acc', 'Plain F1', 'Adv Acc', 'Adv F1', 'ASR'
        ]
        for r in results:
            table.add_row([
                r['model'],
                r['attack'],
                f"{r['plain_acc']*100:.2f}%",
                f"{r['plain_f1']*100:.2f}%",
                f"{r['adv_acc']*100:.2f}%",
                f"{r['adv_f1']*100:.2f}%",
                (("{:.2f}%".format(r['asr']*100)) if r['asr'] == r['asr'] else 'nan'),
            ])
        logger.debug("\n" + table.get_string())

    # Print compact 2D accuracy table: rows = original + attacks, cols = models
    try:
        models_order = models_eval
        # Collect plain acc per model
        plain_acc_map = {}
        for r in results:
            plain_acc_map[r['model']] = r.get('plain_acc', float('nan'))
        # Build adv acc and ASR maps (model, attack) -> value
        adv_map = {}
        asr_map = {}
        tag_set = set()
        for r in results:
            tag = r['attack']
            tag_set.add(tag)
            adv_map[(r['model'], tag)] = r.get('adv_acc', float('nan'))
            asr_map[(r['model'], tag)] = r.get('asr', float('nan'))
        tag_set.discard('original')
        tags_order = ['original'] + sorted(tag_set)

        table2 = PrettyTable()
        table2.field_names = ['Attack/File'] + models_order
        for tag in tags_order:
            row = [tag]
            for m in models_order:
                if tag == 'original':
                    acc_val = plain_acc_map.get(m, float('nan'))
                    asr_txt = '--'
                else:
                    acc_val = adv_map.get((m, tag), float('nan'))
                    asr_val = asr_map.get((m, tag), float('nan'))
                    asr_txt = ("{:.2f}%".format(asr_val*100)) if asr_val == asr_val else 'nan'
                acc_txt = ("{:.2f}%".format(acc_val*100)) if acc_val == acc_val else 'nan'
                row.append(f"{acc_txt} (asr:{asr_txt})")
            table2.add_row(row)
        logger.info("\n" + table2.get_string())
    except Exception:
        pass


if __name__ == "__main__":
    main()


