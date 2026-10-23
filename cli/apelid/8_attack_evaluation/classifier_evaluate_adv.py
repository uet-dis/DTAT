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

MODEL_TYPES = ['xgb', 'dnn', 'tabnet', 'catb', 'bagging', 'gbm']
ATTACK_TYPES = ['zoo', 'deepfool', 'fgsm', 'cw', 'pgd', 'hsja', 'jsma']

# Metrics aggregated as mean ± std across seed-config directories.
SEED_AGG_METRICS = ('plain_acc', 'plain_f1', 'adv_acc', 'adv_f1', 'asr')

# The historical "default" robust checkpoint was trained from pool seed 1337.
# It used to be stored as '<res>_<model>_robust.{pkl,pth}'; it may now live
# under the explicit seed name '<res>_<model>_robust_seed1337.{pkl,pth}'. Both
# are accepted when no explicit model seed is requested.
DEFAULT_ROBUST_SEED = 1337



def _resolve_model_path(models_dir: str, resource_name: str, model_type: str, *, using_robust: bool = False, using_isolated_robust: bool = False, model_seed: int | None = None) -> str:
    ext = '.pth' if model_type in ('dnn', 'tabnet') else '.pkl'
    seed_suffix = f"_seed{model_seed}" if model_seed is not None else ''
    # Default robust checkpoint = pool seed 1337, accepted under either name.
    default_robust = os.path.join(models_dir, f"{resource_name}_{model_type}_robust_seed{DEFAULT_ROBUST_SEED}{ext}")

    if using_isolated_robust:
        # Prefer isolated robust checkpoint (_isolated_robust); fall back to
        # robust (requested seed, then default seed 1337) then base — same
        # precedence as the ensemble evaluator.
        p_iso = os.path.join(models_dir, f"{resource_name}_{model_type}_isolated_robust{seed_suffix}{ext}")
        p_rob = os.path.join(models_dir, f"{resource_name}_{model_type}_robust{seed_suffix}{ext}")
        if os.path.exists(p_iso):
            return p_iso
        if os.path.exists(p_rob):
            logger.warning(f"[!] Isolated robust model missing for {model_type}, fallback to robust: {p_rob}")
            return p_rob
        if model_seed is None and os.path.exists(default_robust):
            logger.warning(f"[!] Isolated robust model missing for {model_type}, fallback to default robust seed "
                           f"{DEFAULT_ROBUST_SEED}: {default_robust}")
            return default_robust
        raise SystemExit(f"Isolated robust model not found: {p_iso}")

    if using_robust:
        p_req = os.path.join(models_dir, f"{resource_name}_{model_type}_robust{seed_suffix}{ext}")
        if os.path.exists(p_req):
            return p_req
        if model_seed is None:
            # Requested without an explicit seed: also accept the seed-1337 name,
            # then fall back to the plain base model.
            if os.path.exists(default_robust):
                logger.warning(f"[!] '{os.path.basename(p_req)}' missing; using default robust seed "
                               f"{DEFAULT_ROBUST_SEED}: {default_robust}")
                return default_robust
            p_base = os.path.join(models_dir, f"{resource_name}_{model_type}{ext}")
            if os.path.exists(p_base):
                logger.warning(f"[!] Robust model missing for {model_type}, fallback to base: {p_base}")
                return p_base
        if model_seed is not None:
            raise SystemExit(
                f"Model not found for pool seed {model_seed}: {p_req}\n"
                f"    Train it first, e.g.: retrain_robust.sh {resource_name} {model_type} --seed {model_seed}"
            )
        raise SystemExit(f"Model file not found: {p_req}")

    p = os.path.join(models_dir, f"{resource_name}_{model_type}{seed_suffix}{ext}")
    if not os.path.exists(p):
        raise SystemExit(f"Model file not found: {p}")
    return p


def _default_adv_path(res, subset: str, model_type: str, attack: str) -> str:
    base_dir = os.path.join(res.DATA_FOLDER, 'adv_samples', subset)
    filename = f"{res.resources_name}_{model_type}_{attack}_adv.csv"
    return os.path.join(base_dir, filename)


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
    """Collapse per-seed rows into one row per ``(model, attack)`` with mean/std."""
    grouped: dict[tuple, list[dict]] = {}
    for row in results:
        grouped.setdefault((row['model'], row['attack']), []).append(row)

    aggregate_rows = []
    for (model, attack), group in grouped.items():
        row = {
            'model': model,
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
    parser.add_argument('--subset', type=str, default='test',
                        help="Adversarial pool subset dir under adv_samples/ used when resolving "
                             "--attack without --adv-in (default: test; e.g. test_v2)")
    parser.add_argument('--adv-fallback-src', type=str, default='dnn', choices=['dnn', 'tabnet'],
                        help="Source model whose white-box (gradient) adv CSVs are used when a member "
                             "has no own adv file (e.g. gradient attacks on trees). Only applies when "
                             "resolving via --attack, not --adv-in. Default: dnn (backward compatible)")
    parser.add_argument('--seed-dirs', type=str, nargs='+', default=None,
                        help="Additional seed-config directories containing same-named adversarial CSVs. "
                             "Their files are appended to the evaluation and, with two or more sources, "
                             "results are reported as mean ± std across seeds")
    parser.add_argument('--seed-label', type=str, default=None,
                        help="Label for the primary (non-seed-dir) source; default 'seed_0' when "
                             "--seed-dirs is given, otherwise 'base'")
    parser.add_argument('--report-out', type=str, default=None,
                        help="Optional CSV with the per-seed evaluation rows")
    parser.add_argument('--agg-report-out', type=str, default=None,
                        help="Optional CSV with the seed-aggregated mean/std table")
    parser.add_argument('--models-dir', type=str, default=None, help="Directory containing trained models")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--using-robust', action='store_true', help="Use robust models (filename suffix _robust)")
    parser.add_argument('--using-isolated-robust', action='store_true', help="Prefer isolated robust models (suffix _isolated_robust); fallback to _robust then base")
    parser.add_argument('--model-seeds', type=int, nargs='+', default=None,
                        help="Evaluate checkpoints trained from pool seeds, i.e. "
                             "'<res>_<model>_robust_seed<S>.<ext>' (implies --using-robust). "
                             "With two or more values results are reported as mean ± std across model seeds. "
                             "Combine with --seed-dirs to also vary the adversarial data source.")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Seed-trained checkpoints live under the robust naming scheme; enable it
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

    # Prepare requested models
    models_requested = args.model if isinstance(args.model, list) else [args.model]
    models_eval = list(models_requested)

    model_seeds = args.model_seeds or [None]
    if len(model_seeds) > 1:
        logger.info(
            "[i] Evaluating %d model seeds: %s (per-model rows aggregated as mean ± std)",
            len(model_seeds), model_seeds,
        )
    results = []
    for model_type, model_seed in [(m, s) for m in models_eval for s in model_seeds]:
        # Load model path and wrapper
        model_path = _resolve_model_path(args.models_dir, res.resources_name, model_type,
                                         using_robust=args.using_robust,
                                         using_isolated_robust=args.using_isolated_robust,
                                         model_seed=model_seed)
        logger.debug(f"[+] Loading wrapper: type={model_type}, seed={model_seed}, path={model_path}")
        wrapper = load_model_as_wrapper(
            model_type,
            model_path,
            num_classes=len(class_names),
            input_dim=X_plain.shape[1],
            clip_values=meta_plain['clip_values'],
            device=args.device,
        )

        # Evaluate on plain test
        logger.debug("[+] Evaluating on plain test set")
        y_plain_pred = _predict_indices(wrapper, X_plain)
        metrics_plain = evaluate_classification(y_plain, y_plain_pred, class_names, metric='weighted')
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
                tasks.append((_default_adv_path(res, subset=args.subset, model_type=model_type, attack=atk), atk, True))

        # Evaluate on adversarial inputs (multiple); each task adds one file per
        # seed-config directory when --seed-dirs is given.
        for adv_path, tag, is_resolved in tasks:
            candidates = [(primary_seed_label, adv_path)]
            for seed_label, seed_dir in seed_sources:
                candidates.append((seed_label, os.path.join(seed_dir, os.path.basename(adv_path))))

            for seed_label, candidate in candidates:
                # Fallback: if resolved path missing and model != fallback source,
                # try the fallback source's adv file for the same attack
                adv_path_use = candidate
                if not os.path.exists(adv_path_use):
                    if is_resolved and model_type != args.adv_fallback_src:
                        try:
                            fallback_path = _default_adv_path(res, subset=args.subset, model_type=args.adv_fallback_src, attack=tag)
                            fallback_path = os.path.join(os.path.dirname(candidate), os.path.basename(fallback_path))
                            if os.path.exists(fallback_path):
                                logger.debug(f"[+] Missing adv file for model={model_type}, attack={tag} [{seed_label}]. Fallback to {args.adv_fallback_src.upper()}: {fallback_path}")
                                adv_path_use = fallback_path
                        except Exception:
                            pass
                if not os.path.exists(adv_path_use):
                    if seed_label == primary_seed_label:
                        logger.warning(f"[!] Adversarial test CSV not found: {adv_path}; skipping")
                    else:
                        logger.debug(f"[+] Seed source '{seed_label}' has no file for {tag} ({candidate}); skipping")
                    continue
                logger.debug(f"[+] Adversarial test CSV [{seed_label}]: {adv_path_use}")

                # Prepare data (adversarial)
                df_adv = pd.read_csv(adv_path_use, low_memory=False)
                X_adv, y_adv, meta_adv = PrepareData.prepare_input_data(df_adv, pre, include_label=True)

                # Evaluate on adversarial test
                logger.debug(f"[+] Evaluating on adversarial test set [{tag}] ({seed_label})")
                y_adv_pred = _predict_indices(wrapper, X_adv)
                metrics_adv = evaluate_classification(y_adv, y_adv_pred, class_names, metric='weighted')

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
                    'model_seed': model_seed,
                    'attack': tag,
                    'seed': seed_label,
                    'file': os.path.basename(adv_path_use),
                    'plain_acc': plain_acc,
                    'plain_f1': plain_f1,
                    'adv_acc': adv_acc,
                    'adv_f1': adv_f1,
                    'asr': asr_val if asr_val is not None else float('nan'),
                })

    # Write per-seed detail CSV (if requested)
    if args.report_out and results:
        pd.DataFrame(results).to_csv(args.report_out, index=False)
        logger.info(f"[+] Per-seed evaluation saved: {args.report_out}")

    # Print final pretty table (per-seed detail when multiple sources)
    if results:
        table = PrettyTable()
        show_seed = bool(seed_sources)
        show_model_seed = len(model_seeds) > 1
        table.field_names = (
            (['M.Seed'] if show_model_seed else []) +
            (['Seed'] if show_seed else []) +
            ['Model', 'Attack/File', 'Plain Acc', 'Plain F1', 'Adv Acc', 'Adv F1', 'ASR']
        )
        for r in results:
            row = ([r.get('model_seed', '')] if show_model_seed else [])
            row += ([r.get('seed', '')] if show_seed else [])
            row += [
                r['model'],
                r['attack'],
                f"{r['plain_acc']*100:.2f}%",
                f"{r['plain_f1']*100:.2f}%",
                f"{r['adv_acc']*100:.2f}%",
                f"{r['adv_f1']*100:.2f}%",
                (("{:.2f}%".format(r['asr']*100)) if r['asr'] == r['asr'] else 'nan'),
            ]
            table.add_row(row)
        logger.debug("\n" + table.get_string())

    # Seed-aggregated data: written to CSV when requested; rendered into the
    # compact 2D table below (no separate console table).
    aggregate = _aggregate_seed_results(results) if ((seed_sources or len(model_seeds) > 1) and results) else []
    if args.agg_report_out and aggregate:
        pd.DataFrame(aggregate).to_csv(args.agg_report_out, index=False)
        logger.info(f"[+] Seed-aggregated evaluation saved: {args.agg_report_out}")

    # Print compact 2D accuracy table: rows = original + attacks, cols = models.
    # With --seed-dirs each cell shows mean ± std across the evaluated seeds.
    try:
        aggregate_map = {(row['model'], row['attack']): row for row in aggregate}
        primary_results = [
            r for r in results
            if not seed_sources or r.get('seed', primary_seed_label) == primary_seed_label
        ]
        models_order = models_eval
        # Collect plain acc per model (mean ± std when several model seeds)
        plain_acc_map = {}
        for r in results:
            plain_acc_map.setdefault(r['model'], {})[(r['model'], r.get('model_seed'))] = r.get('plain_acc', float('nan'))
        # Build adv acc and ASR maps (model, attack) -> value
        adv_map = {}
        asr_map = {}
        tag_set = set()
        for r in primary_results:
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
                    vals = [v for v in plain_acc_map.get(m, {}).values() if v == v]
                    if not vals:
                        acc_txt = 'nan'
                    elif len(vals) > 1:
                        arr = np.array(vals, dtype=float)
                        acc_txt = _fmt_mean_std(float(arr.mean()),
                                                float(arr.std(ddof=1)) if len(vals) > 1 else float('nan'))
                    else:
                        acc_txt = "{:.2f}%".format(vals[0] * 100)
                    asr_txt = '--'
                else:
                    agg_row = aggregate_map.get((m, tag))
                    if agg_row is not None:
                        acc_txt = _fmt_mean_std(
                            agg_row.get('adv_acc_mean', float('nan')),
                            agg_row.get('adv_acc_std', float('nan')),
                        )
                        asr_txt = _fmt_mean_std(
                            agg_row.get('asr_mean', float('nan')),
                            agg_row.get('asr_std', float('nan')),
                        )
                    else:
                        acc_val = adv_map.get((m, tag), float('nan'))
                        asr_val = asr_map.get((m, tag), float('nan'))
                        asr_txt = ("{:.2f}%".format(asr_val*100)) if asr_val == asr_val else 'nan'
                        acc_txt = ("{:.2f}%".format(acc_val*100)) if acc_val == acc_val else 'nan'
                row.append(f"{acc_txt} (asr:{asr_txt})")
            table2.add_row(row)
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
    except Exception:
        pass


if __name__ == "__main__":
    main()


