"""CLI: combine pool quality, proximity, and active-ASR evaluation.

Usage from the repository root::

    python cli/apelid/8_attack_evaluation/pool_evaluation.py \
        -r nslkdd --report-out /tmp/nslkdd_pool_metrics.csv

The command is read-only.  It evaluates every ``*_adv.csv`` in the pool
directory with the model named in the filename, then joins model-independent
quality/proximity metrics from ``art_generator.pool_audit`` with
prediction-based attack metrics from ``art_generator.asr_metrics``.

Seed-config aggregation
----------------------

Passing one or more additional pool directories (e.g. ``seed_1``, ``seed_2``)
with ``--seed-dirs`` appends their ``*_adv.csv`` files to the same evaluation.
When more than one source is given, every ``(model, attack)`` present in at
least two sources is additionally reported as mean ± std across seeds::

    python cli/apelid/8_attack_evaluation/pool_evaluation.py \
        -r cic2018 --report-out /tmp/cic_pool.csv \
        --seed-dirs resources/CIC-2018/adv_samples/test_v2/seed_{1,2,3} \
        --agg-report-out /tmp/cic_pool_seedagg.csv

``--report-out`` keeps the per-seed detail (a ``seed`` column is added);
``--agg-report-out`` writes the aggregated mean/std table (console always shows
it).  Each source contributes the files it actually contains, so a directory
with only some attacks still aggregates those.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from prettytable import PrettyTable

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from configs import CIC2018Resources, NSLKDDResources  # noqa: E402
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor  # noqa: E402
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor  # noqa: E402
from preprocessing.prepare import PrepareData  # noqa: E402
from helpers.wrapping_helper import load_model_as_wrapper  # noqa: E402
from art_generator.asr_metrics import evaluate_attack_predictions  # noqa: E402
from art_generator.eps_scaling import get_inputnorm_sigma  # noqa: E402
from art_generator.pool_audit import audit_pool, budgets_from_attack_params  # noqa: E402
from utils.logging import setup_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}
MODEL_TYPES = ('xgb', 'dnn', 'tabnet', 'catb', 'bagging', 'gbm')

# Metric columns aggregated as mean ± std across seed-config sources.  Columns
# absent from a report are skipped, so the list can grow without breaking.
SEED_AGG_METRICS = (
    'active_pct',
    'unchanged_pct',
    'l2_median',
    'l2_p90',
    'linf_median',
    'linf_p90',
    'l0_median',
    'frozen_changed_rows',
    'domain_violation_cells',
    'log_rms_median',
    'log_rms_p90',
    'log_l1_mean_median',
    'log_l2_median',
    'log_l2_p90',
    'destroyed_pct',
    'clean_accuracy_pct',
    'asr_active_clean_correct_pct',
    'asr_all_clean_correct_pct',
    'active_evasion_pct',
)

# Console summary is split into several narrow tables (one per evidence group
# from src/art_generator/POOL_AUDIT.md, "How to report a result") so nothing has to be scrolled sideways.
# Names are the aggregate columns; the renderer appends ``_mean``/``_std``.
SEED_SUMMARY_GROUPS = (
    ('Proximity - InputNorm (primary)', (
        'active_pct', 'l2_median', 'l2_p90', 'linf_p90', 'l0_median',
    )),
    ('Proximity - log-space (supplementary)', (
        'log_rms_median', 'log_rms_p90', 'log_l1_mean_median',
    )),
    ('Attack effectiveness (reported separately)', (
        'asr_all_clean_correct_pct',
    )),
)

# Same split for the single-source report (raw columns, no mean/std).
REPORT_SUMMARY_GROUPS = (
    ('Proximity - InputNorm (primary)', (
        'active_pct', 'l2_median', 'l2_p90', 'linf_p90', 'l0_median',
    )),
    ('Proximity - log-space (supplementary)', (
        'log_rms_median', 'log_rms_p90', 'log_l1_mean_median',
    )),
    ('Attack effectiveness (reported separately)', (
        'asr_all_clean_correct_pct',
    )),
)



def _model_path(res, model: str, models_dir: str | None) -> str:
    directory = models_dir or os.path.join(res.DATA_FOLDER, 'models')
    extension = '.pth' if model in ('dnn', 'tabnet') else '.pkl'
    return os.path.join(directory, f'{res.resources_name}_{model}{extension}')


def _load_model(res, model: str, models_dir: str | None, *, input_dim: int,
                clip_values, num_classes: int, device: str):
    path = _model_path(res, model, models_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f'model not found for {model}: {path}')
    return load_model_as_wrapper(
        model,
        path,
        num_classes=num_classes,
        input_dim=input_dim,
        clip_values=clip_values,
        device=device,
    )


def _render_report(report: pd.DataFrame, *, full: bool = False) -> str:
    if full:
        return _pretty_table(report).get_string()
    identity = [column for column in ('seed', 'file') if column in report.columns]
    status_column = ['status'] if 'status' in report.columns else []
    blocks = []
    for title, metrics in REPORT_SUMMARY_GROUPS:
        present = [metric for metric in metrics if metric in report.columns]
        if not present:
            continue
        frame = report[identity + present + status_column]
        blocks.append(f'--- {title} ---\n{_pretty_table(frame).get_string()}')
    return '\n\n'.join(blocks)


# Console cells longer than this are truncated (CSV always keeps the full value).
_MAX_CELL_LEN = 80


def _cell_text(value, *, max_len: int = _MAX_CELL_LEN) -> str:
    """Stringify one DataFrame cell for console display."""
    if value is None:
        return ''
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ''
        text = f'{float(value):.4g}'
    else:
        text = str(value)
        if text.strip().lower() in ('nan', 'none', '<na>'):
            return ''
    if max_len and len(text) > max_len:
        text = text[: max_len - 1] + '…'
    return text


def _pretty_table(frame: pd.DataFrame, *, max_len: int = _MAX_CELL_LEN) -> PrettyTable:
    table = PrettyTable()
    table.field_names = [str(column) for column in frame.columns]
    for record in frame.itertuples(index=False, name=None):
        table.add_row([_cell_text(value, max_len=max_len) for value in record])
    return table


def _aggregate_seeds(report: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed rows into one row per ``(model, attack)`` with mean/std."""
    rows = []
    for (model, attack), group in report.groupby(['model', 'attack'], sort=False):
        statuses = [str(value) for value in group['status'].dropna().unique()] \
            if 'status' in group.columns else []
        if statuses and set(statuses) == {'PASS'}:
            status = 'PASS'
        elif 'ERROR' in statuses:
            status = 'ERROR'
        elif statuses:
            status = 'WARN'
        else:
            status = ''
        row = {
            'model': model,
            'attack': attack,
            'n_seeds': int(group['seed'].nunique()) if 'seed' in group.columns else 1,
            'seeds': ','.join(map(str, group['seed'].tolist())) if 'seed' in group.columns else '',
            'status': status,
        }
        for metric in SEED_AGG_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors='coerce').dropna()
            if values.empty:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else float('nan')
            row[f'{metric}_mean'] = mean
            row[f'{metric}_std'] = std
            row[f'{metric}_mean_pm_std'] = (
                f'{mean:.4g}' if len(values) < 2 or std != std else f'{mean:.4g} ± {std:.4g}'
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _render_seed_aggregate(aggregate: pd.DataFrame, *, full: bool = False) -> str:
    """Render mean ± std across seeds, split into one narrow table per group."""
    if full:
        return _pretty_table(aggregate).get_string()
    blocks = []
    for title, metrics in SEED_SUMMARY_GROUPS:
        present = [metric for metric in metrics if f'{metric}_mean' in aggregate.columns]
        if not present:
            continue
        display_rows = []
        for _, row in aggregate.iterrows():
            display = {
                'model': row.get('model', '?'),
                'attack': row.get('attack', '?'),
            }
            for metric in present:
                mean = row.get(f'{metric}_mean', float('nan'))
                std = row.get(f'{metric}_std', float('nan'))
                if pd.isna(mean):
                    display[metric] = '-'
                elif pd.isna(std):
                    display[metric] = f'{mean:.4g}'
                else:
                    display[metric] = f'{mean:.4g} ± {std:.3g}'
            display['status'] = row.get('status', '')
            display_rows.append(display)
        blocks.append(
            f'--- {title} ---\n{_pretty_table(pd.DataFrame(display_rows)).get_string()}'
        )
    return '\n\n'.join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Read-only combined quality/proximity/active-ASR pool evaluation'
    )
    parser.add_argument('--resource', '-r', required=True, choices=list(REGISTRY))
    parser.add_argument('--pool-dir', default=None,
                        help='Pool directory (default: {DATA_FOLDER}/adv_samples/test_v2)')
    parser.add_argument('--orig-csv', default=None,
                        help='Clean source CSV (default: balanced test CSV)')
    parser.add_argument('--models-dir', default=None,
                        help='Directory containing trained model files')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--ztol', type=float, default=1e-3,
                        help='InputNorm tolerance for L0/domain checks')
    parser.add_argument('--unchanged-tol', type=float, default=1e-5,
                        help='InputNorm tolerance for active/unchanged rows')
    parser.add_argument('--destruction-threshold', type=float, default=10.0)
    parser.add_argument('--report-out', default=None,
                        help='Optional output CSV containing per-seed audit and ASR metrics. '
                             'Omit to run without writing any file (console output only)')
    parser.add_argument('--seed-dirs', nargs='+', default=None,
                        help='Additional seed-config pool directories. Their *_adv.csv files '
                             'are evaluated and appended to the report; with two or more '
                             'sources, per-(model, attack) results are aggregated as mean ± std '
                             'across seeds')
    parser.add_argument('--pool-label', default=None,
                        help="Label for the primary pool dir (default: 'seed_0' when "
                             '--seed-dirs is given, otherwise the directory basename)')
    parser.add_argument('--agg-report-out', default=None,
                        help='Optional CSV for the seed-aggregated mean/std table '
                             '(written when --seed-dirs is given)')
    parser.add_argument('--print-report', default='summary',
                        choices=['summary', 'full', 'none'],
                        help="Console output: 'summary' (headline table via the logger), "
                             "'full' (every column of the per-seed report and, with multiple "
                             "sources, of the aggregate; printed to stdout), 'none' (CSV only). "
                             "Default: summary")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    resources_class, preprocessor_class = REGISTRY[args.resource]
    res = resources_class
    pre = preprocessor_class()
    if not pre.load_encoders(fixed_label_encoder=True):
        raise SystemExit('Encoders not found. Train models first.')

    pool_dir = args.pool_dir or os.path.join(res.DATA_FOLDER, 'adv_samples', 'test_v2')
    orig_csv = args.orig_csv or os.path.join(
        res.BALENCED_DATA_FOLDER,
        f'{res.resources_name}_test_random_sample_clean_merged.csv',
    )
    if not os.path.isdir(pool_dir):
        raise SystemExit(f'pool directory not found: {pool_dir}')
    if not os.path.exists(orig_csv):
        raise SystemExit(f'original test CSV not found: {orig_csv}')

    seed_dirs = list(args.seed_dirs or [])
    if args.pool_label is not None:
        primary_label = args.pool_label
    elif seed_dirs:
        primary_label = 'seed_0'
    else:
        primary_label = os.path.basename(os.path.normpath(pool_dir))
    sources = [(primary_label, pool_dir)]
    for seed_dir in seed_dirs:
        seed_path = os.path.abspath(seed_dir)
        if not os.path.isdir(seed_path):
            raise SystemExit(f'seed directory not found: {seed_dir}')
        sources.append((os.path.basename(os.path.normpath(seed_path)), seed_path))

    df_orig = pd.read_csv(orig_csv, low_memory=False)
    x_orig, y_orig, meta = PrepareData.prepare_input_data(
        df_orig, pre, include_label=True
    )
    num_classes = len(res.MAJORITY_LABELS + res.MINORITY_LABELS)

    dnn_wrapper = _load_model(
        res, 'dnn', args.models_dir, input_dim=x_orig.shape[1],
        clip_values=meta['clip_values'], num_classes=num_classes, device=args.device,
    )
    cont_idx, sigma = get_inputnorm_sigma(dnn_wrapper)

    budgets = {}
    for attack_name, attack_params in res.ATTACK_PARAMETERS.items():
        budget = budgets_from_attack_params(attack_params)
        if budget is not None:
            budgets[attack_name] = budget

    model_cache = {'dnn': dnn_wrapper}
    audit_frames = []
    asr_rows = []
    for label, directory in sources:
        logger.info('[evaluation] source %s -> %s', label, directory)
        source_audit = audit_pool(
            directory,
            orig_csv,
            pre,
            meta,
            cont_idx,
            sigma,
            budgets=budgets,
            ztol=args.ztol,
            unchanged_tol=args.unchanged_tol,
            destruction_threshold=args.destruction_threshold,
        )
        source_audit.insert(0, 'seed', label)
        audit_frames.append(source_audit)

        for audit_row in source_audit.to_dict('records'):
            filename = audit_row.get('file')
            model = audit_row.get('model')
            if model not in MODEL_TYPES or not filename:
                continue
            if model not in model_cache:
                model_cache[model] = _load_model(
                    res,
                    model,
                    args.models_dir,
                    input_dim=x_orig.shape[1],
                    clip_values=meta['clip_values'],
                    num_classes=num_classes,
                    device=args.device,
                )

            adv_path = Path(directory) / filename
            df_adv = pd.read_csv(adv_path, low_memory=False)
            if len(df_adv) != len(df_orig):
                logger.warning('[evaluation] skipping %s/%s: row mismatch', label, filename)
                continue
            x_adv, _, _ = PrepareData.prepare_input_data(
                df_adv, pre, include_label=True
            )
            metrics = evaluate_attack_predictions(
                model_cache[model],
                x_orig,
                x_adv,
                y_orig,
                cont_idx,
                sigma,
                unchanged_tol=args.unchanged_tol,
            )
            asr_rows.append({'seed': label, 'file': filename, **metrics})

    report = pd.concat(audit_frames, ignore_index=True)
    asr_frame = pd.DataFrame(asr_rows)
    if not asr_frame.empty:
        report = report.merge(asr_frame, on=['seed', 'file'], how='left')
    if args.report_out:
        report.to_csv(args.report_out, index=False)

    if len(sources) > 1:
        aggregate = _aggregate_seeds(report)
        if args.agg_report_out:
            aggregate.to_csv(args.agg_report_out, index=False)
            logger.info('[+] Seed-aggregated report saved: %s', args.agg_report_out)
        if args.print_report == 'full':
            print(f'=== Per-seed detail ({len(sources)} sources) ===')
            print(_render_report(report, full=True))
            print('\n=== Mean ± std across seeds (grouped by model × attack) ===')
            print(_render_seed_aggregate(aggregate, full=True))
        elif args.print_report == 'summary':
            logger.debug('\n=== Per-seed detail (%d sources) ===\n%s',
                         len(sources), _render_report(report))
            logger.info('\n=== Mean ± std across seeds (grouped by model × attack) ===\n%s',
                        _render_seed_aggregate(aggregate))
    else:
        if args.print_report == 'full':
            print(_render_report(report, full=True))
        elif args.print_report == 'summary':
            logger.info('\n%s', _render_report(report))
    if args.report_out:
        logger.info('[+] Combined report saved: %s', args.report_out)


if __name__ == '__main__':
    main()
