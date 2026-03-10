#!/usr/bin/env python3
"""
Unit Consistency (UC) Recommender Systems — Main Experiment Runner

Three experiment modes
----------------------
  8  — Rank-Preference Consistency: All Items
       Strong preferences (1 vs. 5 ratings) and subtle preferences (4 vs. 5).
       Dense matrices for small/medium datasets; sparse for large datasets.

  9  — Rank-Preference Consistency: Long-Tail Items
       Evaluates UC on the least-frequently rated 67% of items, for both
       strong and subtle preference cases.

  ranking — Standard Ranking Evaluation (Precision@k, Recall@k, NDCG@k)
       Follows the standard implicit-feedback protocol (ratings >= threshold
       as positive, 80/20 per-user random split, full-ranking protocol).
       S=10 seeds for small datasets, S=5 for large datasets.

Usage
-----
    python main.py --dataset ML-1M --seed 42 --experiment 9
    python main.py --dataset ML-1M --seed 42 --experiment 8
    python main.py --dataset ML-1M --seed 42 --experiment ranking --output_dir results
    python main.py --dataset ML-20M --seed 0  --experiment 9

Supported datasets
------------------
    ML-100K, ML-1M, Douban_monti, ML-20M, Netflix
"""

import argparse
import os
import sys
import gc
import json
import numpy as np
from datetime import datetime

# NumPy 2.0 compatibility
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "int"):
    np.int = np.int64
if not hasattr(np, "bool"):
    np.bool = np.bool_

import torch

# Utils package (all utilities live here)
from utils import (
    Logger,
    save_data,
    load_json,
    run_UC_TC_easy,
    run_UC_TC_hard,
    preprocess_data,
    train_test_split,
    load_data_ML100K_exp_9,
    load_data_ML1M_exp_9,
    load_data_ML20M_exp_9,
    load_data_monti_exp_9,
    load_data_Netflix_exp_9,
    sample_products_all_users_by_indices,
    sample_products_all_users_by_indices_sparse,
    get_users_from_indices,
    get_users_from_indices_sparse,
)
from utils.dataloader import (
    load_data_100k,
    load_data_1m,
    load_data_monti,
    load_big_data_ML_20M,
    load_big_data_Netflix,
)
from utils.ranking_eval import run_uc_ranking_evaluation

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

EASY_SETS = {'ML-100K', 'ML-1M', 'Douban_monti'}
HARD_SETS = {'ML-20M', 'Netflix'}


def _get_data_path(dataset, data_root):
    return os.path.join(data_root, dataset)


def load_ratings_exp9(dataset, data_path):
    """Load ratings DataFrame (exp-9 / standard ranking format)."""
    if dataset == 'ML-100K':
        return load_data_ML100K_exp_9(path=data_path)
    elif dataset == 'ML-1M':
        return load_data_ML1M_exp_9(path=data_path, delimiter='::')
    elif dataset == 'Douban_monti':
        return load_data_monti_exp_9(path=data_path)
    elif dataset == 'ML-20M':
        return load_data_ML20M_exp_9(path=data_path)
    elif dataset == 'Netflix':
        return load_data_Netflix_exp_9(path=data_path, files=[2, 4])
    raise ValueError(f"Unknown dataset: {dataset}")


def load_matrix_exp8(dataset, data_path, seed):
    """Load dense/sparse rating matrix (exp-8 format)."""
    if dataset == 'ML-100K':
        return load_data_100k(path=data_path)
    elif dataset == 'ML-1M':
        return load_data_1m(path=data_path, delimiter='::', seed=seed)
    elif dataset == 'Douban_monti':
        return load_data_monti(path=data_path)
    elif dataset == 'ML-20M':
        return load_big_data_ML_20M(path=data_path, seed=seed)
    elif dataset == 'Netflix':
        return load_big_data_Netflix(path=data_path, seed=seed)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Experiment 8 — All-Items Rank-Preference Consistency
# ---------------------------------------------------------------------------

def run_experiment_8(dataset, data_path, seed, output_dir, logger):
    """
    Evaluate UC on all items for both strong (1 vs 5) and subtle (4 vs 5) preferences.
    Iterates over S seeds; S=10 for easy datasets, S=5 for hard datasets.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    logger.log(f"[Exp 8] Dataset={dataset}  seed={seed}  device={device}")

    os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)

    if dataset in EASY_SETS:
        n_m, n_u, train_r, train_m = load_matrix_exp8(dataset, data_path, seed)
        train_r = get_users_from_indices(train_r, [1, 2, 3, 4, 5])

        ratings_arr = [[1, 5], [4, 5]]
        for ratings in ratings_arr:
            logger.log(f"[Exp 8] Rating pair: {ratings}")
            train_r_curr, train_r_N, train_m_N, samples_products = \
                sample_products_all_users_by_indices(train_r, ratings)
            n_m2, n_u2 = train_r_curr.shape
            r_train = torch.tensor(train_r_N).to(device)
            test_r = train_r_curr - train_r_N

            hash_UCTC = run_UC_TC_easy(n_u2, r_train, test_r, samples_products, device)
            path = os.path.join(output_dir, dataset,
                                f'exp8_{ratings}_{dataset}_seed{seed}_UC.json')
            save_data(path, hash_UCTC)
            logger.log(f"[Exp 8] Saved: {path}")

            del train_r_N, train_m_N, test_r, train_r_curr, r_train
            gc.collect()
            torch.cuda.empty_cache()

    else:  # HARD_SETS
        from scipy import sparse
        n_m, n_u, train_r, train_m = load_matrix_exp8(dataset, data_path, seed)
        train_r_sparse = sparse.csr_matrix(train_r) if not sparse.issparse(train_r) else train_r
        train_r_sparse = get_users_from_indices_sparse(train_r_sparse, [1, 2, 3, 4, 5])

        ratings_arr = [[1, 5], [4, 5]]
        for ratings in ratings_arr:
            logger.log(f"[Exp 8] Rating pair: {ratings}")
            train_r_curr, train_r_N, train_m_N, samples_products = \
                sample_products_all_users_by_indices_sparse(train_r_sparse, ratings)
            n_m2, n_u2 = train_r_curr.shape
            test_r = train_r_curr - train_r_N

            hash_UCTC = run_UC_TC_hard(n_u2, train_r_N, test_r, samples_products, device)
            path = os.path.join(output_dir, dataset,
                                f'exp8_{ratings}_{dataset}_seed{seed}_UC.json')
            save_data(path, hash_UCTC)
            logger.log(f"[Exp 8] Saved: {path}")

            del train_r_N, train_m_N, test_r, train_r_curr
            gc.collect()
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Experiment 9 — Long-Tail Rank-Preference Consistency
# ---------------------------------------------------------------------------

def run_experiment_9(dataset, data_path, seed, output_dir, logger):
    """
    Evaluate UC on long-tail items for both strong (1 vs 5) and subtle (4 vs 5)
    preferences.  Long-tail = least-frequently rated 67% of items.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    logger.log(f"[Exp 9] Dataset={dataset}  seed={seed}  device={device}")

    os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)

    mode = "easy" if dataset in EASY_SETS else "hard"
    ratings = load_ratings_exp9(dataset, data_path)
    long_tail_ratings = preprocess_data(ratings)

    for start_rating in [1, 4]:
        pairs = [start_rating, 5]
        logger.log(f"[Exp 9] Rating pair: {pairs}")

        training_ratings, test_ratings, train_r, test_r, samples_products = \
            train_test_split(start_rating, long_tail_ratings, ratings, mode=mode)

        n_m, n_u = train_r.shape
        logger.log(f"[Exp 9] Matrix: {n_m} items x {n_u} users")

        if mode == "easy":
            r_train = torch.tensor(train_r).to(device)
            hash_UCTC = run_UC_TC_easy(n_u, r_train, test_r, samples_products, device)
            del r_train
        else:
            hash_UCTC = run_UC_TC_hard(n_u, train_r, test_r, samples_products, device)

        path = os.path.join(output_dir, dataset,
                            f'exp9_{pairs}_{dataset}_seed{seed}_UC.json')
        save_data(path, hash_UCTC)
        logger.log(f"[Exp 9] Saved: {path}")

        del train_r, test_r
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Standard Ranking Evaluation
# ---------------------------------------------------------------------------

def run_experiment_ranking(dataset, data_path, seed, output_dir, logger):
    """
    Evaluate UC on Precision@k, Recall@k, NDCG@k using the standard
    implicit-feedback protocol (ratings >= 4.0, 80/20 split, full ranking).
    """
    ratings = load_ratings_exp9(dataset, data_path)
    logger.log(f"[Ranking] Dataset={dataset}  seed={seed}")
    logger.log(f"[Ranking] Loaded {len(ratings)} ratings  "
               f"({ratings['UserID'].nunique()} users, "
               f"{ratings['MovieID'].nunique()} items)")

    run_uc_ranking_evaluation(
        dataset_name=dataset,
        ratings=ratings,
        methods=['UC', 'TC'],
        k_values=[5, 10, 20],
        random_state=seed,
        logger=logger,
        eval_mode='full_ranking',
        compute_uc_metrics=True,
        rating_threshold=4.0,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "UC Recommender Systems — run experiments 8, 9, or standard ranking."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='ML-1M',
        choices=['ML-100K', 'ML-1M', 'Douban_monti', 'ML-20M', 'Netflix'],
        help='Dataset to use.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility.',
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='9',
        choices=['8', '9', 'ranking'],
        help=(
            '8  = All-Items rank-preference consistency, '
            '9  = Long-Tail rank-preference consistency, '
            'ranking = standard P@k/R@k/NDCG@k evaluation.'
        ),
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to write result JSON / CSV files.',
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help=(
            'Root directory containing dataset folders. '
            'If not given, falls back to paths.json or ./data.'
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve data path
    if args.data_path is not None:
        data_root = args.data_path
    else:
        paths_json = os.path.join(os.path.dirname(__file__), 'paths.json')
        if os.path.exists(paths_json):
            PATHS = load_json(paths_json)
            data_root = PATHS.get('data_path', 'data')
        else:
            data_root = 'data'

    dataset_path = os.path.join(data_root, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    # Logger
    log_dir = os.path.join('logs', args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f'exp{args.experiment}_seed{args.seed}_{dt}.log')
    logger = Logger(log_path)

    logger.log("=" * 70)
    logger.log("UC RECOMMENDER SYSTEMS — EXPERIMENT RUNNER")
    logger.log("=" * 70)
    logger.log(f"Dataset:    {args.dataset}")
    logger.log(f"Seed:       {args.seed}")
    logger.log(f"Experiment: {args.experiment}")
    logger.log(f"Output dir: {args.output_dir}")
    logger.log(f"Data path:  {dataset_path}")
    logger.log("=" * 70)
    logger.log("")

    if args.experiment == '8':
        run_experiment_8(args.dataset, dataset_path, args.seed, args.output_dir, logger)
    elif args.experiment == '9':
        run_experiment_9(args.dataset, dataset_path, args.seed, args.output_dir, logger)
    elif args.experiment == 'ranking':
        run_experiment_ranking(args.dataset, dataset_path, args.seed, args.output_dir, logger)

    logger.log("")
    logger.log("Done.")
    logger.close()


if __name__ == '__main__':
    main()
