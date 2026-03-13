#!/usr/bin/env python3
"""
Unit Consistency (UC) Recommender Systems — Main Experiment Runner

Experiment modes
----------------
  strong_and_subtle — All-items rank-preference consistency.
                      Strong (1 vs 5) and subtle (4 vs 5) preference pairs.

  long_tail         — Long-tail rank-preference consistency.
                      Same preference pairs on the least-rated 67% of items.

  ranking           — Standard ranking evaluation (P@k, R@k, NDCG@k).
                      80/20 split, full-ranking protocol, ratings >= 4.0.

Usage
-----
    python main.py --dataset ML-1M --seed 42 --experiment strong_and_subtle
    python main.py --dataset ML-1M --seed 42 --experiment long_tail
    python main.py --dataset ML-1M --seed 42 --experiment ranking
    python main.py --dataset ML-20M --seed 0  --experiment long_tail

Datasets: ML-100K, ML-1M, Douban_monti, ML-20M, Netflix
"""

import argparse
import os
import gc
import numpy as np
from datetime import datetime

if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "int"):
    np.int = np.int64
if not hasattr(np, "bool"):
    np.bool = np.bool_

import torch
from scipy import sparse

from utils import (
    Logger, save_data, load_json,
    run_UC_easy, run_UC_hard,
    preprocess_data, train_test_split,
    load_data_ML100K_exp_9, load_data_ML1M_exp_9,
    load_data_ML20M_exp_9, load_data_monti_exp_9, load_data_Netflix_exp_9,
    sample_products_all_users_by_indices, sample_products_all_users_by_indices_sparse,
    get_users_from_indices, get_users_from_indices_sparse,
)
from utils.dataloader import (
    load_data_100k, load_data_1m, load_data_monti,
    load_big_data_ML_20M, load_big_data_Netflix,
)
from utils.ranking_eval import run_uc_ranking_evaluation

EASY_SETS = {'ML-100K', 'ML-1M', 'Douban_monti'}
HARD_SETS = {'ML-20M', 'Netflix'}


def load_ratings(dataset, data_path):
    """Load ratings DataFrame."""
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


def load_matrix(dataset, data_path, seed):
    """Load dense/sparse rating matrix."""
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
# strong_and_subtle — all-items, strong (1v5) and subtle (4v5) preferences
# ---------------------------------------------------------------------------

def run_strong_and_subtle(dataset, data_path, seed, output_dir, logger):
    """UC on all items: strong (1 vs 5) and subtle (4 vs 5) preference pairs."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    logger.log(f"[strong_and_subtle] {dataset}  seed={seed}  device={device}")
    os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)

    if dataset in EASY_SETS:
        train_r = load_matrix(dataset, data_path, seed)
        train_r = get_users_from_indices(train_r, [1, 2, 3, 4, 5])
        for ratings in [[1, 5], [4, 5]]:
            train_r_curr, train_r_N, train_m_N, samples_products = \
                sample_products_all_users_by_indices(train_r, ratings)
            n_m2, n_u2 = train_r_curr.shape
            r_train = torch.tensor(train_r_N).to(device)
            test_r = train_r_curr - train_r_N
            result = run_UC_easy(n_u2, r_train, test_r, samples_products, device)
            path = os.path.join(output_dir, dataset,
                                f'strong_and_subtle_{ratings}_{dataset}_seed{seed}_UC.json')
            save_data(path, result)
            logger.log(f"Saved: {path}")
            del train_r_N, train_m_N, test_r, train_r_curr, r_train
            gc.collect(); torch.cuda.empty_cache()

    else:
        train_r = load_matrix(dataset, data_path, seed)
        train_r_sp = sparse.csr_matrix(train_r) if not sparse.issparse(train_r) else train_r
        train_r_sp = get_users_from_indices_sparse(train_r_sp, [1, 2, 3, 4, 5])
        for ratings in [[1, 5], [4, 5]]:
            train_r_curr, train_r_N, train_m_N, samples_products = \
                sample_products_all_users_by_indices_sparse(train_r_sp, ratings)
            n_m2, n_u2 = train_r_curr.shape
            test_r = train_r_curr - train_r_N
            result = run_UC_hard(n_u2, train_r_N, test_r, samples_products, device)
            path = os.path.join(output_dir, dataset,
                                f'strong_and_subtle_{ratings}_{dataset}_seed{seed}_UC.json')
            save_data(path, result)
            logger.log(f"Saved: {path}")
            del train_r_N, train_m_N, test_r, train_r_curr
            gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# long_tail — long-tail items, strong (1v5) and subtle (4v5) preferences
# ---------------------------------------------------------------------------

def run_long_tail(dataset, data_path, seed, output_dir, logger):
    """UC on long-tail items (least-rated 67%): strong and subtle preference pairs."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    logger.log(f"[long_tail] {dataset}  seed={seed}  device={device}")
    os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)

    mode = "easy" if dataset in EASY_SETS else "hard"
    ratings = load_ratings(dataset, data_path)
    long_tail_ratings = preprocess_data(ratings)

    for start_rating in [1, 4]:
        pairs = [start_rating, 5]
        logger.log(f"[long_tail] Pair: {pairs}")
        train_r, test_r, samples_products = train_test_split(start_rating, long_tail_ratings, ratings, mode=mode)
        n_m, n_u = train_r.shape

        if mode == "easy":
            r_train = torch.tensor(train_r).to(device)
            result = run_UC_easy(n_u, r_train, test_r, samples_products, device)
            del r_train
        else:
            result = run_UC_hard(n_u, train_r, test_r, samples_products, device)

        path = os.path.join(output_dir, dataset,
                            f'long_tail_{pairs}_{dataset}_seed{seed}_UC.json')
        save_data(path, result)
        logger.log(f"Saved: {path}")
        del train_r, test_r
        gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# ranking — standard P@k / R@k / NDCG@k (full-ranking protocol)
# ---------------------------------------------------------------------------

def run_ranking(dataset, data_path, seed, output_dir, logger):
    """Standard ranking evaluation: full-ranking, ratings >= 4.0, 80/20 split."""
    ratings = load_ratings(dataset, data_path)
    logger.log(f"[ranking] {dataset}  seed={seed}  "
               f"{ratings['UserID'].nunique()} users  {ratings['MovieID'].nunique()} items")
    run_uc_ranking_evaluation(
        dataset_name=dataset,
        ratings=ratings,
        methods=['UC'],
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
        description='UC Recommender Systems — experiment runner.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--dataset', type=str, default='ML-1M',
                        choices=['ML-100K', 'ML-1M', 'Douban_monti', 'ML-20M', 'Netflix'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment', type=str, default='long_tail',
                        choices=['strong_and_subtle', 'long_tail', 'ranking'],
                        help='strong_and_subtle | long_tail | ranking')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--data_path', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.data_path is not None:
        data_root = args.data_path
    else:
        paths_json = os.path.join(os.path.dirname(__file__), 'paths.json')
        if os.path.exists(paths_json):
            data_root = load_json(paths_json).get('data_path', 'data')
        else:
            data_root = 'data'

    dataset_path = os.path.join(data_root, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    log_dir = os.path.join('logs', args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f'{args.experiment}_seed{args.seed}_{dt}.log')
    logger = Logger(log_path)

    logger.log(f"Dataset={args.dataset}  seed={args.seed}  experiment={args.experiment}")
    logger.log(f"output_dir={args.output_dir}  data_path={dataset_path}")
    logger.log("")

    if args.experiment == 'strong_and_subtle':
        run_strong_and_subtle(args.dataset, dataset_path, args.seed, args.output_dir, logger)
    elif args.experiment == 'long_tail':
        run_long_tail(args.dataset, dataset_path, args.seed, args.output_dir, logger)
    elif args.experiment == 'ranking':
        run_ranking(args.dataset, dataset_path, args.seed, args.output_dir, logger)

    logger.log("Done.")
    logger.close()


if __name__ == '__main__':
    main()
