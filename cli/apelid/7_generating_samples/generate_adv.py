import os
import sys
import shutil
import time
import argparse
import json
import multiprocessing as mp
from pathlib import Path

# S4 (E0/E2): force a deterministic cuBLAS workspace BEFORE any torch/CUDA
# initialization. Must happen before the imports below (wrapping_helper pulls
# torch in at module import time).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources  # noqa: E402
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor  # noqa: E402
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor  # noqa: E402
from preprocessing.prepare import PrepareData  # noqa: E402
from helpers.wrapping_helper import load_model_as_wrapper  # noqa: E402
from art_generator.zoo import ZooAttackGenerator  # noqa: E402
from art_generator.deepfool import DeepFoolAttackGenerator  # noqa: E402
from art_generator.fgsm import FGSMAttackGenerator  # noqa: E402
from art_generator.cw import CWAttackGenerator  # noqa: E402
from art_generator.pgd import PGDAttackGenerator  # noqa: E402
from art_generator.hsja import HSJAAttackGenerator  # noqa: E402
from art_generator.jsma import JSMAAttackGenerator  # noqa: E402
from art_generator.eps_scaling import build_per_feature_eps_arrays  # noqa: E402
from art_generator.inputnorm_adapter import data_zspace_clip  # noqa: E402


logger = get_logger(__name__)

# Default robust checkpoint = pool seed 1337; accepted under either the
# historical unsuffixed name or the explicit seed name.
DEFAULT_ROBUST_SEED = 1337


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}

MODEL_TYPES = ['xgb', 'dnn', 'tabnet', 'catb', 'bagging', 'gbm']
ATTACK_TYPES = ['zoo', 'deepfool', 'fgsm', 'cw', 'pgd', 'hsja', 'jsma']  # extend later
# Minimum-norm attacks run in InputNorm coordinates (scale-coherent minimal
# perturbation, see art_generator/inputnorm_adapter.py).
INPUTNORM_ATTACKS = ['deepfool', 'cw', 'zoo', 'hsja']

# Sharded generation supports attacks whose per-sample optimization is
# independent (batch_size=1 semantics) and whose domain box is attached
# per generate() call: zoo and hsja.
SHARDABLE_ATTACKS = ['zoo', 'hsja']

# Keep the unpatched constructor available.  ``_patch_randomstate`` is called
# in the main process and in spawned workers, so reading ``np.random.RandomState``
# inside that function would otherwise wrap the wrapper again.
_ORIGINAL_RANDOMSTATE = np.random.RandomState


def _patch_randomstate(seed: int):
    """Make unseeded ``RandomState()`` constructions use the current RNG stream.

    HSJA creates an UNSEEDED RandomState for its initial sample
    (art hop_skip_jump.py:295), which ``np.random.seed`` does NOT cover.
    Each unseeded construction therefore receives a child seed drawn from the
    globally seeded legacy RNG.  In sharded mode ``_worker_chunk`` reseeds that
    global stream with ``seed + chunk_idx`` before every chunk, so samples have
    independent but reproducible initialization streams.  A caller that passes
    an explicit seed keeps the normal RandomState(seed) behavior.
    """
    def _seeded_randomstate(rs=None):
        if rs is None:
            child_seed = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
            return _ORIGINAL_RANDOMSTATE(child_seed)
        return _ORIGINAL_RANDOMSTATE(rs)

    np.random.RandomState = _seeded_randomstate


def _inputnorm_stats_for(res, models_dir: str, X_in: np.ndarray, estimator, model_type: str,
                         *, using_robust: bool = False):
    """InputNorm statistics for z-space minimum-norm attacks.

    For the DNN itself, from its own checkpoint (the adapter shares the
    model's buffers). For other model types, from the trained DNN checkpoint
    — the canonical sigma units of the project.
    """
    from art_generator.inputnorm_adapter import InputNormStats
    if model_type == 'dnn':
        return InputNormStats.from_estimator(estimator)
    suffix = '_robust' if using_robust else ''
    dnn_path = os.path.join(models_dir, f"{res.resources_name}_dnn{suffix}.pth")
    if not os.path.exists(dnn_path) and using_robust:
        for cand in (f"{res.resources_name}_dnn_robust_seed{DEFAULT_ROBUST_SEED}.pth",
                     f"{res.resources_name}_dnn.pth"):
            alt = os.path.join(models_dir, cand)
            if os.path.exists(alt):
                logger.warning(f"[!] Robust DNN missing for InputNorm stats, fallback to {alt}")
                dnn_path = alt
                break
    if not os.path.exists(dnn_path):
        return None
    dnn_wrapper = load_model_as_wrapper(
        'dnn', dnn_path,
        num_classes=len(res.MAJORITY_LABELS + res.MINORITY_LABELS),
        input_dim=X_in.shape[1],
        clip_values=(0.0, 1.0),  # buffers only; clip values irrelevant here
        device='cpu',
    )
    return InputNormStats.from_estimator(dnn_wrapper.get_estimator())


def _default_inputs(res) -> tuple[str, str]:
    data_dir = res.BALENCED_DATA_FOLDER
    train_path = os.path.join(data_dir, f"{res.resources_name}_merged_train_raw_processed.csv")
    test_path = os.path.join(data_dir, f"{res.resources_name}_test_random_sample_clean_merged.csv")
    return train_path, test_path


def _resolve_model_path(models_dir: str, resource_name: str, model_type: str, *,
                        using_robust: bool = False) -> str:
    ext = '.pth' if model_type in ('dnn', 'tabnet') else '.pkl'
    stem = f"{resource_name}_{model_type}{'_robust' if using_robust else ''}"
    p = os.path.join(models_dir, f"{stem}{ext}")
    if not os.path.exists(p):
        # Fallback chain when robust is requested but the exact name is absent:
        # default seed 1337 checkpoint first, then the plain base model.
        if using_robust:
            for cand in (f"{resource_name}_{model_type}_robust_seed{DEFAULT_ROBUST_SEED}{ext}",
                         f"{resource_name}_{model_type}{ext}"):
                alt = os.path.join(models_dir, cand)
                if os.path.exists(alt):
                    logger.warning(f"[!] Robust model missing for {model_type}, fallback to {alt}")
                    return alt
        raise SystemExit(f"Model file not found: {p}")
    return p


def _parse_attack_param_override(raw: str, params: dict) -> tuple[str, object]:
    """Parse one ``KEY=VALUE`` attack-parameter override from the CLI."""
    if '=' not in raw:
        raise ValueError(f"expected KEY=VALUE, got: {raw}")
    key, value = raw.split('=', 1)
    key = key.strip()
    if key not in params:
        raise ValueError(f"unknown parameter '{key}'; available: {', '.join(sorted(params))}")
    current = params[key]
    value = value.strip()
    if isinstance(current, bool):
        lowered = value.lower()
        if lowered not in ('true', 'false'):
            raise ValueError(f"boolean parameter '{key}' requires true or false")
        parsed = lowered == 'true'
    elif isinstance(current, int) and not isinstance(current, bool):
        parsed = int(value)
    elif isinstance(current, float):
        parsed = float(value)
    else:
        parsed = value
    return key, parsed


# ---------------------------------------------------------------------------
# Sharded generation (--workers > 1)
#
# Why sharding is safe for zoo/hsja: ART runs these attacks with
# batch_size=1 semantics (ZOO even rejects batch_size > 1 for feature
# vectors), so each sample's optimization is independent and contiguous
# chunks reproduce the sequential pool modulo the per-chunk RNG stream.
#
# The z-space clip box and the raw clamp floor are computed ONCE on the
# FULL input and handed to every worker: inside generate() the box would
# otherwise be derived from the chunk itself, which is tighter than the
# full-set box and inconsistent between shards (audit compares the pool
# against the FULL original CSV).
# ---------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _quiet_worker_tqdm():
    """Silence tqdm bars inside shard worker processes.

    ART renders its own bars (zoo.py:32 ``from tqdm.auto import trange``,
    hop_skip_jump.py:30 ``from tqdm.auto import tqdm``), constructed at
    generate() call time. With --workers > 1 every worker would draw its
    own bar on the shared stderr and they stomp each other (garbled
    output). Patching ``disable=True`` into the constructor keeps tqdm
    iterable but silent; the main process renders one aggregated bar
    instead.
    """
    import tqdm
    _orig_init = tqdm.std.tqdm.__init__

    def _quiet_init(self, *args, **kwargs):
        # Force-disable even when the caller passes disable explicitly:
        # ART always passes disable=not self.verbose (zoo.py:232,
        # hop_skip_jump.py:195) with verbose=True, which would defeat
        # setdefault() — shard workers must never render bars.
        kwargs['disable'] = True
        _orig_init(self, *args, **kwargs)

    tqdm.std.tqdm.__init__ = _quiet_init


def _worker_init(cfg: dict):
    """Per-process setup: thread limits, model, attack generator (once)."""
    import torch
    torch.set_num_threads(1)
    _quiet_worker_tqdm()
    if cfg['seed'] is not None:
        np.random.seed(cfg['seed'])
        _patch_randomstate(cfg['seed'])
    ResClass, _ = REGISTRY[cfg['resource']]
    model_path = _resolve_model_path(cfg['models_dir'], ResClass.resources_name, cfg['model_type'],
                                     using_robust=cfg['using_robust'])
    wrapper = load_model_as_wrapper(
        cfg['model_type'], model_path,
        num_classes=cfg['num_classes'],
        input_dim=cfg['input_dim'],
        clip_values=cfg['clip_values'],
        device=cfg['device'],
    )
    estimator = wrapper.get_estimator()
    # _inputnorm_stats_for only reads X_in.shape[1] — a dummy row suffices.
    inputnorm_stats = _inputnorm_stats_for(
        ResClass, cfg['models_dir'], np.zeros((1, cfg['input_dim'])), estimator, cfg['model_type'],
        using_robust=cfg['using_robust'],
    )
    if inputnorm_stats is None or not inputnorm_stats.cont_idx:
        raise RuntimeError(
            f"Attack {cfg['attack']} requires the DNN checkpoint's InputNorm statistics"
        )
    if cfg['attack'] == 'zoo':
        generator = ZooAttackGenerator(estimator, generator_params=cfg['attack_params'],
                                       inputnorm_stats=inputnorm_stats)
    elif cfg['attack'] == 'hsja':
        generator = HSJAAttackGenerator(estimator, generator_params=cfg['attack_params'],
                                        inputnorm_stats=inputnorm_stats)
    else:  # guarded in main(); unreachable via the CLI
        raise RuntimeError(f"Sharding not supported for attack: {cfg['attack']}")
    _WORKER_STATE['generator'] = generator
    _WORKER_STATE['input_metadata'] = {'feature_names': cfg['feature_names'],
                                       'label_column': cfg['label_column']}
    _WORKER_STATE['mutate_indices'] = cfg['mutate_indices']
    _WORKER_STATE['box'] = cfg['box']
    _WORKER_STATE['raw_floor'] = cfg['raw_floor']
    _WORKER_STATE['generate_kwargs'] = cfg['generate_kwargs']
    _WORKER_STATE['seed'] = cfg['seed']


def _worker_chunk(task: tuple):
    """Attack one contiguous chunk; returns (chunk_idx, df_adv)."""
    chunk_idx, X_chunk, y_chunk = task
    if _WORKER_STATE.get('seed') is not None:
        # Per-chunk reseed: chunk results are reproducible regardless of
        # which worker ran them or how the scheduler interleaved chunks.
        np.random.seed(_WORKER_STATE['seed'] + chunk_idx)
    generator = _WORKER_STATE['generator']
    df = generator.generate(
        X_chunk, y_chunk,
        input_metadata=_WORKER_STATE['input_metadata'],
        mutate_indices=_WORKER_STATE['mutate_indices'],
        zspace_clip=_WORKER_STATE['box'],
        raw_floor=_WORKER_STATE['raw_floor'],
        **_WORKER_STATE['generate_kwargs'],
    )
    return chunk_idx, df


def _generate_sharded(res, args, X_in: np.ndarray, y_in: np.ndarray, meta: dict,
                      attack_params: dict, inputnorm_stats) -> pd.DataFrame:
    """Shard the input into contiguous chunks attacked by parallel processes."""
    n_rows = X_in.shape[0]
    n_chunks = max(1, args.workers * args.chunks_per_worker)
    chunk_rows = int(np.ceil(n_rows / n_chunks))
    chunks = [
        (idx, X_in[start:start + chunk_rows], y_in[start:start + chunk_rows])
        for idx, start in enumerate(range(0, n_rows, chunk_rows))
    ]

    # Domain box + raw clamp floor computed on the FULL input: identical for
    # every shard (see module comment above).
    raw_clip = meta['clip_values']
    raw_clip_max = raw_clip[1] if raw_clip is not None else None
    box = data_zspace_clip(inputnorm_stats, inputnorm_stats.to_z(X_in), raw_clip_max=raw_clip_max)
    raw_floor = np.asarray(X_in, dtype=np.float64).min(axis=0)

    shard_dir = os.path.join(
        args.output_dir, "_shards", f"{res.resources_name}_{args.model}_{args.attack}"
    )
    os.makedirs(shard_dir, exist_ok=True)

    # Limit child-process thread pools BEFORE their imports: spawn children
    # inherit this environment at interpreter startup (nthread=1 for model
    # predict is set separately by the generator adapters).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ctx = mp.get_context("spawn")

    generate_kwargs = {}
    if args.attack in ['hsja', 'jsma']:
        generate_kwargs.update({
            'batch_size': args.batch_size,
            'max_retries': args.max_retries,
            'timeout': args.timeout,
            'placeholder': args.placeholder,
            'verbose': args.verbose,
        })

    worker_cfg = {
        'resource': args.resource,
        'models_dir': args.models_dir,
        'using_robust': args.using_robust,
        'model_type': args.model,
        'attack': args.attack,
        'device': args.device,
        'attack_params': attack_params,
        'num_classes': len(res.MAJORITY_LABELS + res.MINORITY_LABELS),
        'input_dim': X_in.shape[1],
        'clip_values': tuple(meta['clip_values']) if meta['clip_values'] is not None else None,
        'feature_names': list(meta['feature_names']),
        'label_column': meta['label_column'],
        'mutate_indices': list(meta['cat_feature_indices'] + meta['binary_feature_indices']),
        'box': box,
        'raw_floor': raw_floor,
        'seed': args.seed,
        'generate_kwargs': generate_kwargs,
    }

    logger.info(
        f"[+] Sharded generation: {len(chunks)} chunks over {args.workers} workers "
        f"({chunk_rows} rows/chunk); shards -> {shard_dir}"
    )

    # Resume: chunks whose shard CSV already exists (with the right row
    # count) are reloaded, not re-attacked.
    parts: list = [None] * len(chunks)
    tasks = []
    for idx, X_chunk, y_chunk in chunks:
        shard_csv = os.path.join(shard_dir, f"shard_{idx:05d}.csv")
        if os.path.exists(shard_csv):
            try:
                df_shard = pd.read_csv(shard_csv, low_memory=False)
                if len(df_shard) == X_chunk.shape[0]:
                    logger.info(f"[+] shard {idx}: resume from {shard_csv} ({len(df_shard)} rows)")
                    parts[idx] = df_shard
                    continue
                logger.warning(
                    f"[!] shard {idx}: saved row count {len(df_shard)} != {X_chunk.shape[0]} — regenerating"
                )
            except Exception as e:  # corrupt/partial shard file: regenerate
                logger.warning(f"[!] shard {idx}: unreadable shard file ({e}) — regenerating")
        tasks.append((idx, X_chunk, y_chunk))

    t0 = time.time()
    if tasks:
        from tqdm.auto import tqdm
        # Explicit close()/join() instead of `with Pool(...)` semantics: the
        # context manager calls terminate() on exit, which SIGTERMs workers
        # before they release the Pool's internal semaphores — the resource
        # tracker then reports "N leaked semaphore objects" at shutdown.
        pool = ctx.Pool(processes=min(args.workers, len(tasks)),
                        initializer=_worker_init, initargs=(worker_cfg,))
        try:
            pbar = tqdm(total=len(tasks), desc="Shards", unit="shard", leave=True)
            for done_idx, df_shard in pool.imap_unordered(_worker_chunk, tasks):
                shard_csv = os.path.join(shard_dir, f"shard_{done_idx:05d}.csv")
                df_shard.to_csv(shard_csv, index=False)
                parts[done_idx] = df_shard
                pbar.update(1)
                logger.info(f"[+] shard {done_idx}/{len(chunks) - 1} done "
                            f"({len(df_shard)} rows, {time.time() - t0:.1f}s elapsed)")
            pbar.close()
            pool.close()   # no more tasks; workers exit after finishing theirs
            pool.join()
        except BaseException:
            pool.terminate()  # hard-stop on error; saved shards allow resume
            pool.join()
            raise
    if any(p is None for p in parts):
        missing = [i for i, p in enumerate(parts) if p is None]
        raise SystemExit(f"Shards {missing} missing; rerun the same command to resume from saved shards.")

    df_adv = pd.concat(parts, ignore_index=True)
    if len(df_adv) != n_rows:
        raise SystemExit(f"Sharded merge produced {len(df_adv)} rows, expected {n_rows}")
    logger.info(f"[+] Sharded generation complete: {n_rows} rows in {time.time() - t0:.1f}s")

    if not args.keep_shards:
        shutil.rmtree(shard_dir, ignore_errors=True)
        logger.info(f"[+] Shard directory removed ({shard_dir}); use --keep-shards to retain it")
    return df_adv


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial samples using ART attacks")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--subset', '-s', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--model', '-m', type=str, required=True, choices=MODEL_TYPES, help="Model type to attack")
    parser.add_argument('--attack', '-a', type=str, required=True, choices=ATTACK_TYPES, help="Attack type")
    parser.add_argument('--data-in', '-i', type=str, default=None, help="Input data file to perturb (default: test)")
    parser.add_argument('--models-dir', type=str, default=None, help="Directory containing trained models")
    parser.add_argument('--using-robust', action='store_true',
                        help="Attack the robust model (filename suffix _robust) instead of the base model; "
                             "falls back to base with a warning when the robust file is missing. "
                             "InputNorm statistics follow the same suffix (z-space attacks)")
    parser.add_argument('--output-dir', type=str, default=None, help="Directory to save adversarial CSV")
    parser.add_argument('--device', '-d', type=str, default='cpu', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--batch-size', type=int, default=-1, help="Optional batching for large inputs (-1 = full)")
    parser.add_argument('--samples', type=int, default=-1, help="Limit number of rows to attack (-1 = all)")
    parser.add_argument('--sampling-mode', type=str, default='sequential', choices=['sequential', 'random'],
                        help="How to pick samples when limiting: sequential head or random without replacement")
    parser.add_argument('--max-retries', type=int, default=3, help="Max retries for batch processing (HSJA/JSMA)")
    parser.add_argument('--timeout', type=int, default=-1, help="Timeout in seconds for each batch (HSJA/JSMA, -1 = no timeout)")
    parser.add_argument('--placeholder', type=str, default='original', choices=['original', 'drop'],
                        help="Policy for failed batches: use original samples or drop them")
    parser.add_argument('--seed', type=int, default=None,
                        help="Seed for the global numpy RNG (attack-level randomness: ZOO/HSJA). None = unseeded (previous behavior)")
    parser.add_argument('--targeted', action='store_true',
                        help="Force targeted mode for attacks that support it (fgsm/pgd/cw). Target = the highest-scoring class other than the true label. JSMA is always targeted.")
    parser.add_argument('--attack-param', action='append', default=[], metavar='KEY=VALUE',
                        help="Override one configured attack parameter without editing configs; repeatable, e.g. --attack-param init_size=50")
    parser.add_argument('--hsja-fallback-radius', type=float, default=None,
                        help="Optional HSJA hybrid fallback: retry unchanged rows at this z-space radius")
    parser.add_argument('--verbose', type=int, default=0,
                        help="Log batch progress every N batches for HSJA/JSMA (0 = disable)")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help="Parallel worker processes (sharded generation; per-sample independent, "
                             "batch_size=1 semantics). 1 = sequential single process (previous behavior).")
    parser.add_argument('--chunks-per-worker', type=int, default=4,
                        help="Contiguous chunks per worker for sharded generation (dynamic load balancing; "
                             "hard samples may skew contiguous splits)")
    parser.add_argument('--keep-shards', action='store_true',
                        help="Keep per-chunk shard CSVs after a successful merge (default: delete them; "
                             "they are always kept when the run fails, so the same command resumes)")
    args = parser.parse_args()

    setup_logging(args.log_level)

    # S1: seed the global numpy RNG (covers ZOO/HSJA attack randomness and
    # random sampling mode). None keeps the previous unseeded behavior.
    if args.seed is not None:
        np.random.seed(args.seed)
        logger.info(f"[+] Global numpy RNG seeded with {args.seed}")
        # HSJA creates an UNSEEDED RandomState for its initial adversarial image
        # (art hop_skip_jump.py:295) which np.random.seed does NOT cover. Patch
        # RandomState so unseeded constructions default to --seed (R2 in plan).
        _patch_randomstate(args.seed)
        logger.info("[+] np.random.RandomState patched: unseeded RandomState() now defaults to --seed (covers HSJA init)")

    # S4: hard determinism flags for DNN gradient attacks on GPU (E0/E2).
    try:
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        logger.info("[+] Determinism flags enabled (torch deterministic algorithms, cudnn.deterministic)")
    except Exception as e:
        logger.warning(f"[!] Could not enable full determinism flags: {e}")

    # Setup resource and preprocessor
    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass
    pre = PreprocessorClass()

    # Paths
    if args.models_dir is None:
        args.models_dir = os.path.join(res.DATA_FOLDER, 'models')
    if args.output_dir is None:
        args.output_dir = os.path.join(res.DATA_FOLDER, 'adv_samples', args.subset)

    # Default test data
    if args.data_in is None:
        default_train, default_test = _default_inputs(res)
        if args.subset == 'train':
            args.data_in = default_train
        else:
            args.data_in = default_test


    # Load data
    if not os.path.exists(args.data_in):
        raise SystemExit(f"Input data file not found: {args.data_in}")
    logger.info(f"[+] Loading input data: {args.data_in}")
    df_in = pd.read_csv(args.data_in, low_memory=False)
    logger.info(f"[+] Input data shape: {df_in.shape}")

    # Load encoders
    if not pre.load_encoders(fixed_label_encoder=True):
        raise SystemExit("Encoders not found. Train models first.")

    # Prepare features and labels (unified)
    X_in, y_in, meta = PrepareData.prepare_input_data(
        df_in, pre, include_label=True
    )

    # Optional sampling
    if args.samples is not None and args.samples > 0:
        n = min(args.samples, X_in.shape[0])
        if args.sampling_mode == 'random':
            idx = np.random.permutation(X_in.shape[0])[:n]
            X_in = X_in[idx]
            y_in = y_in[idx]
            logger.info(f"[+] Using {n} random samples for attack")
        else:
            X_in = X_in[:n]
            y_in = y_in[:n]
            logger.info(f"[+] Using first {n} samples for attack")

    # Attack parameters (config), with optional --targeted override
    attack_params = dict(res.ATTACK_PARAMETERS[args.attack])
    for raw_override in args.attack_param:
        try:
            key, value = _parse_attack_param_override(raw_override, attack_params)
        except ValueError as exc:
            raise SystemExit(f"Invalid --attack-param: {exc}") from exc
        attack_params[key] = value
        logger.info("[+] Attack parameter override: %s=%s", key, value)
    if args.hsja_fallback_radius is not None:
        if args.attack != 'hsja':
            raise SystemExit("--hsja-fallback-radius is only supported for --attack hsja")
        if args.hsja_fallback_radius <= 0:
            raise SystemExit("--hsja-fallback-radius must be positive")
        attack_params['fallback_radius'] = args.hsja_fallback_radius
        logger.info("[+] HSJA hybrid fallback radius: %s", args.hsja_fallback_radius)
    if args.targeted:
        if args.attack in ['pgd', 'fgsm', 'cw']:
            attack_params['targeted'] = True
            logger.info(f"[+] --targeted flag: forcing targeted mode for {args.attack}")
        elif args.attack == 'jsma':
            logger.info("[+] --targeted: JSMA is inherently targeted (masked second-most-likely targets) — no change needed")
        else:
            raise SystemExit(f"--targeted is not supported for attack: {args.attack} (supported: fgsm, pgd, cw, jsma)")

    # Load ART wrapper for the selected model
    model_path = _resolve_model_path(args.models_dir, res.resources_name, args.model,
                                     using_robust=args.using_robust)
    logger.info(f"[+] Loading classifier wrapper: type={args.model}, path={model_path}")
    wrapper = load_model_as_wrapper(
        args.model,
        model_path,
        num_classes=len(res.MAJORITY_LABELS + res.MINORITY_LABELS),
        input_dim=X_in.shape[1],
        clip_values=meta['clip_values'],
        device=args.device,
    )
    estimator = wrapper.get_estimator()

    # Option B: per-feature eps scaled by the trained DNN's InputNorm sigma.
    # Budget is defined in InputNorm space (k standard deviations per
    # continuous feature) — see art_generator/eps_scaling.py for rationale.
    # NOTE: conversion happens here (needs the loaded wrapper + meta); the
    # wrapper itself cannot resolve it at __init__ time.
    if attack_params.get('eps_per_std') is not None:
        if args.attack not in ('fgsm', 'pgd'):
            raise SystemExit(f"eps_per_std is only supported for fgsm/pgd, got: {args.attack}")
        if args.model not in ('dnn', 'tabnet'):
            raise SystemExit(f"eps_per_std requires a torch tabular model with InputNorm (dnn|tabnet), got model: {args.model}")
        k_eps = float(attack_params.pop('eps_per_std'))
        k_step = float(attack_params.pop('eps_step_per_std', k_eps / 10.0))
        eps_arr, step_arr = build_per_feature_eps_arrays(
            wrapper, meta, k_eps, k_step, feature_dim=X_in.shape[1]
        )
        attack_params['eps'] = eps_arr
        attack_params['eps_step'] = step_arr

    # InputNorm statistics for z-space minimum-norm attacks (deepfool/cw/zoo/hsja)
    if args.attack in INPUTNORM_ATTACKS:
        inputnorm_stats = _inputnorm_stats_for(res, args.models_dir, X_in, estimator, args.model,
                                               using_robust=args.using_robust)
        if inputnorm_stats is None or not inputnorm_stats.cont_idx:
            raise SystemExit(
                f"Attack {args.attack} runs in InputNorm coordinates and requires "
                f"the trained DNN checkpoint's InputNorm statistics; "
                f"no DNN model found under {args.models_dir}"
            )
        logger.info("[+] InputNorm stats ready: %d continuous features (source: %s)",
                    len(inputnorm_stats.cont_idx),
                    'own model' if args.model == 'dnn' else 'DNN checkpoint')
    else:
        inputnorm_stats = None

    # Sharded generation (--workers > 1): per-sample independent attacks.
    if args.workers > 1:
        if args.attack not in SHARDABLE_ATTACKS:
            raise SystemExit(
                f"--workers > 1 supports sharding for {SHARDABLE_ATTACKS} only, got: {args.attack}"
            )
        logger.info(f"[+] Sharded mode: workers={args.workers}, chunks/worker={args.chunks_per_worker}")
        df_adv = _generate_sharded(res, args, X_in, y_in, meta, attack_params, inputnorm_stats)
    else:
        # Select attack generator
        if args.attack == 'zoo':
            generator = ZooAttackGenerator(estimator, generator_params=attack_params,
                                           inputnorm_stats=inputnorm_stats)
        elif args.attack == 'deepfool':
            if args.model not in ('dnn', 'tabnet'):
                raise SystemExit(f"DeepFool attack only supports torch tabular models (dnn|tabnet), got: {args.model}")
            generator = DeepFoolAttackGenerator(estimator, generator_params=attack_params)
        elif args.attack == 'fgsm':
            if args.model not in ('dnn', 'tabnet'):
                raise SystemExit(f"FGSM attack only supports torch tabular models (dnn|tabnet), got: {args.model}")
            generator = FGSMAttackGenerator(estimator, generator_params=attack_params)
        elif args.attack == 'cw':
            if args.model not in ('dnn', 'tabnet'):
                raise SystemExit(f"CW attack only supports torch tabular models (dnn|tabnet), got: {args.model}")
            generator = CWAttackGenerator(estimator, generator_params=attack_params)
        elif args.attack == 'pgd':
            if args.model not in ('dnn', 'tabnet'):
                raise SystemExit(f"PGD attack only supports torch tabular models (dnn|tabnet), got: {args.model}")
            generator = PGDAttackGenerator(estimator, generator_params=attack_params)
        elif args.attack == 'hsja':
            # HSJA supports all model types (no restriction)
            generator = HSJAAttackGenerator(estimator, generator_params=attack_params,
                                            inputnorm_stats=inputnorm_stats)
        elif args.attack == 'jsma':
            if args.model not in ('dnn', 'tabnet'):
                raise SystemExit(f"JSMA attack only supports torch tabular models (dnn|tabnet), got: {args.model}")
            generator = JSMAAttackGenerator(estimator, generator_params=attack_params)
        else:
            raise SystemExit(f"Unsupported attack: {args.attack}")

        # Generate adversarial samples; y is mandatory per our interface
        logger.info(f"[+] Generating adversarial samples: attack={args.attack}")

        # Prepare kwargs for batch processing (HSJA/JSMA)
        generate_kwargs = {}
        if args.attack in ['hsja', 'jsma']:
            generate_kwargs.update({
                'batch_size': args.batch_size,
                'max_retries': args.max_retries,
                'timeout': args.timeout,
                'placeholder': args.placeholder,
                'verbose': args.verbose
            })

        df_adv = generator.generate(
            X_in, y_in,
            input_metadata=meta,
            mutate_indices=meta['cat_feature_indices'] + meta['binary_feature_indices'],
            **generate_kwargs
        )

    # inverse transform the adversarial samples (ordinal and label)
    df_adv = pre.inverse_transform_ordinal_features(df_adv)
    df_adv = pre.inverse_transform_label(df_adv)

    # label distribution
    logger.info(f"[+] Label distribution of adversarial samples: {df_adv[pre.label_column].value_counts()}")

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(
        args.output_dir, f"{res.resources_name}_{args.model}_{args.attack}_adv.csv"
    )
    df_adv.to_csv(out_csv, index=False)
    logger.info(f"[+] Adversarial CSV saved to: {out_csv}")


if __name__ == "__main__":
    main()
