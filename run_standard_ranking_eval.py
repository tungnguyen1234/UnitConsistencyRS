"""
Main runner script for standard ranking evaluation.

This script addresses reviewer concerns by evaluating UC and baseline methods
on conventional ranking metrics (Precision@k, Recall@k, NDCG@k) using
standard random train/test splits.

Usage:
    python run_standard_ranking_eval.py --dataset ML-1M --methods UC BPR-MF LightGCN

Arguments:
    --dataset: Dataset name (ML-100K, ML-1M, ML-20M, Douban_monti, Netflix)
    --methods: Methods to evaluate (UC, TC, BPR-MF, LightGCN, NCF, NGCF, SGL, NCL, SimpleX)
    --k_values: K values for top-k metrics (default: 5 10 20)
    --data_path: Path to data directory
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from standard_ranking_experiment import run_standard_ranking_evaluation
from utils.experiment_utils import (
    load_data_ML1M_exp_9,
    load_data_ML20M_exp_9,
    load_data_monti_exp_9,
    load_data_Netflix_exp_9
)

# Load JSON utility (check if utils.py exists and what it contains)
try:
    from utils import load_json
except ImportError:
    import json
    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)


class Logger:
    """Simple logger that writes to both console and file."""

    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Run standard ranking evaluation for UC benchmarking'
    )
    parser.add_argument('--dataset', type=str, default='ML-1M',
                        choices=['ML-100K', 'ML-1M', 'ML-20M', 'Douban_monti', 'Netflix'],
                        help='Dataset name')
    parser.add_argument('--methods', nargs='+', default=['UC'],
                        help='Methods to evaluate (UC, TC)')
    parser.add_argument('--k_values', nargs='+', type=int, default=[5, 10, 20],
                        help='K values for top-k metrics')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data directory (if not specified, uses paths.json)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, auto-detect if not specified)')
    parser.add_argument('--temp_dir', type=str, default='temp_standard_ranking',
                        help='Temporary directory for intermediate files')
    parser.add_argument('--config_dir', type=str, default='configs/recbole',
                        help='Config directory for RecBole models')
    parser.add_argument('--eval_mode', type=str, default='rerank',
                        choices=['rerank', 'full_ranking', 'negative_sampling'],
                        help='Evaluation mode: rerank (only test items), full_ranking (all items - training), negative_sampling (test + N negatives)')
    parser.add_argument('--num_negatives', type=int, default=100,
                        help='Number of negative samples per user (only for negative_sampling mode)')
    parser.add_argument('--use_uc_reranking', action='store_true',
                        help='Evaluate UC as a re-ranking module on top of base models (BaseModel vs BaseModel+UC)')
    parser.add_argument('--rerank_top_n', type=int, default=100,
                        help='Number of top candidates to re-rank with UC (only for --use_uc_reranking)')
    parser.add_argument('--compute_uc_metrics', action='store_true',
                        help='Compute UC-specific metrics (CVR and Kendall-tau) for all methods')
    parser.add_argument('--rating_threshold', type=float, default=4.0,
                        help='Minimum rating to consider as positive feedback (default: 4.0)')

    args = parser.parse_args()

    # Numpy compatibility
    if not hasattr(np, "float"):
        np.float = np.float64
    if not hasattr(np, "int"):
        np.int = np.int64
    if not hasattr(np, "bool"):
        np.bool = np.bool_

    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Get data path
    if args.data_path is None:
        paths_json = os.path.join(os.path.dirname(__file__), 'paths.json')
        if os.path.exists(paths_json):
            PATHS = load_json(paths_json)
            data_path = PATHS.get('data_path', 'data')
        else:
            data_path = 'data'
    else:
        data_path = args.data_path

    # Set up logger
    log_file = os.path.join('logs', f'{args.dataset}_standard_ranking_results', f'experiment_seed_{args.random_state}_{args.methods}.log')
    logger = Logger(log_file)

    logger.log("=" * 80)
    logger.log("STANDARD RANKING EVALUATION FOR UC BENCHMARKING")
    logger.log("=" * 80)
    logger.log(f"Dataset: {args.dataset}")
    logger.log(f"Methods: {args.methods}")
    logger.log(f"K values: {args.k_values}")
    logger.log(f"Evaluation mode: {args.eval_mode}")
    if args.eval_mode == 'negative_sampling':
        logger.log(f"Num negatives: {args.num_negatives}")
    if args.use_uc_reranking:
        logger.log(f"UC Re-ranking: ON (top-{args.rerank_top_n} candidates)")
    if args.compute_uc_metrics:
        logger.log(f"UC-specific metrics: ON (CVR, Kendall-tau)")
    logger.log(f"Rating threshold: {args.rating_threshold} (ratings >= {args.rating_threshold} as positive)")
    logger.log(f"Data path: {data_path}")
    logger.log(f"Device: {device}")
    logger.log(f"Temp dir: {args.temp_dir}")
    logger.log("")

    # Load data
    logger.log("Loading dataset...")
    dataset_path = os.path.join(data_path, args.dataset)

    try:
        if args.dataset == 'ML-100K':
            ratings = load_data_ML100K_exp_9(path=dataset_path)
        elif args.dataset == 'ML-1M':
            ratings = load_data_ML1M_exp_9(path=dataset_path, delimiter='::', seed=1234)
        elif args.dataset == 'ML-20M':
            ratings = load_data_ML20M_exp_9(path=dataset_path)
        elif args.dataset == 'Douban_monti':
            ratings = load_data_monti_exp_9(path=dataset_path)
        elif args.dataset == 'Netflix':
            ratings = load_data_Netflix_exp_9(path=dataset_path, files=[2, 4])
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        logger.log(f"Dataset loaded: {len(ratings)} ratings")
        logger.log(f"Users: {ratings['UserID'].nunique()}, Items: {ratings['MovieID'].nunique()}")
        logger.log("")

    except Exception as e:
        logger.log(f"Error loading dataset: {e}")
        raise

    # Apply dataset-specific FairGAN defaults from YAML if not overridden by CLI
    fairgan_config_path = os.path.join(args.config_dir, args.dataset, 'fairgan_local.yaml')
    if os.path.exists(fairgan_config_path):
        import yaml
        with open(fairgan_config_path, 'r') as f:
            fg_defaults = yaml.safe_load(f)
        args.fairgan_hidden_size = fg_defaults.get('hidden_size', None)
        args.fairgan_batch_size = fg_defaults.get('batch_size', None)
        args.fairgan_fairness_sample_items = fg_defaults.get('fairness_sample_items', None)
        fairgan_epochs = fg_defaults.get('epochs', None)
        logger.log(f"FairGAN config loaded from: {fairgan_config_path}")

    # Use dataset-specific config directory if it exists, otherwise fall back to base
    dataset_config_dir = os.path.join(args.config_dir, args.dataset)
    if os.path.isdir(dataset_config_dir):
        config_dir = dataset_config_dir
        logger.log(f"Using dataset-specific configs: {config_dir}")
    else:
        config_dir = args.config_dir
        logger.log(f"Using default configs: {config_dir}")

    # Run evaluation
    try:
        results = run_standard_ranking_evaluation(
            dataset_name=args.dataset,
            ratings=ratings,
            methods=args.methods,
            k_values=args.k_values,
            device=device,
            random_state=args.random_state,
            logger=logger,
            temp_dir=args.temp_dir,
            config_dir=config_dir,
            eval_mode=args.eval_mode,
            num_negatives=args.num_negatives,
            use_uc_reranking=args.use_uc_reranking,
            rerank_top_n=args.rerank_top_n,
            compute_uc_metrics=args.compute_uc_metrics,
            rating_threshold=args.rating_threshold,
            fairgan_hidden_size=args.fairgan_hidden_size,
            fairgan_batch_size=args.fairgan_batch_size,
            fairgan_fairness_sample_items=args.fairgan_fairness_sample_items,
            fairgan_epochs=fairgan_epochs
        )

        logger.log("\n" + "=" * 80)
        logger.log("SUCCESS: Standard ranking evaluation completed!")
        logger.log("=" * 80)

        return 0

    except Exception as e:
        logger.log(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
