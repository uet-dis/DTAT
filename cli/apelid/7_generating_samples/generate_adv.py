import os
import sys
import argparse
import json
from pathlib import Path

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


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}

MODEL_TYPES = ['xgb', 'dnn', 'catb', 'bagging', 'histgbm', 'gbm']
ATTACK_TYPES = ['zoo', 'deepfool', 'fgsm', 'cw', 'pgd', 'hsja', 'jsma']  # extend later


def _default_inputs(res) -> tuple[str, str]:
    data_dir = res.BALENCED_DATA_FOLDER
    train_path = os.path.join(data_dir, f"{res.resources_name}_merged_train_raw_processed.csv")
    test_path = os.path.join(data_dir, f"{res.resources_name}_test_random_sample_clean_merged.csv")
    return train_path, test_path


def _resolve_model_path(models_dir: str, resource_name: str, model_type: str) -> str:
    ext = '.pth' if model_type == 'dnn' else '.pkl'
    p = os.path.join(models_dir, f"{resource_name}_{model_type}{ext}")
    if not os.path.exists(p):
        raise SystemExit(f"Model file not found: {p}")
    return p


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial samples using ART attacks")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--subset', '-s', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--model', '-m', type=str, required=True, choices=MODEL_TYPES, help="Model type to attack")
    parser.add_argument('--attack', '-a', type=str, required=True, choices=ATTACK_TYPES, help="Attack type")
    parser.add_argument('--data-in', '-i', type=str, default=None, help="Input data file to perturb (default: test)")
    parser.add_argument('--models-dir', type=str, default=None, help="Directory containing trained models")
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
    parser.add_argument('--verbose', type=int, default=0,
                        help="Log batch progress every N batches for HSJA/JSMA (0 = disable)")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

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

    # Load ART wrapper for the selected model
    model_path = _resolve_model_path(args.models_dir, res.resources_name, args.model)
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

    # Select attack generator
    if args.attack == 'zoo':
        generator = ZooAttackGenerator(estimator, generator_params=res.ATTACK_PARAMETERS[args.attack])
    elif args.attack == 'deepfool':
        if args.model != 'dnn':
            raise SystemExit(f"DeepFool attack only supports DNN models, got: {args.model}")
        generator = DeepFoolAttackGenerator(estimator, generator_params=res.ATTACK_PARAMETERS[args.attack])
    elif args.attack == 'fgsm':
        if args.model != 'dnn':
            raise SystemExit(f"FGSM attack only supports DNN models, got: {args.model}")
        generator = FGSMAttackGenerator(estimator, generator_params=res.ATTACK_PARAMETERS[args.attack])
    elif args.attack == 'cw':
        if args.model != 'dnn':
            raise SystemExit(f"CW attack only supports DNN models, got: {args.model}")
        generator = CWAttackGenerator(estimator, generator_params=res.ATTACK_PARAMETERS[args.attack])
    elif args.attack == 'pgd':
        if args.model != 'dnn':
            raise SystemExit(f"PGD attack only supports DNN models, got: {args.model}")
        generator = PGDAttackGenerator(estimator, generator_params=res.ATTACK_PARAMETERS[args.attack])
    elif args.attack == 'hsja':
        # HSJA supports all model types (no restriction)
        generator = HSJAAttackGenerator(estimator, generator_params=res.ATTACK_PARAMETERS[args.attack])
    elif args.attack == 'jsma':
        if args.model != 'dnn':
            raise SystemExit(f"JSMA attack only supports DNN models, got: {args.model}")
        generator = JSMAAttackGenerator(estimator, generator_params=res.ATTACK_PARAMETERS[args.attack])
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


