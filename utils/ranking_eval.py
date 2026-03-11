"""
Standard Ranking Evaluation for UC (Unit Consistency).

Evaluates UC and TC on Precision@k, Recall@k, and NDCG@k using a standard
80/20 per-user random train/test split and full-ranking protocol.

Only UC/TC methods are evaluated here. Baseline methods were run using
RecBole; hyperparameter configurations are reported in Tables 15-18 of
the paper.
"""

import sys
import os
import numpy as np
import torch
import gc
from time import time
from scipy import sparse
import pandas as pd
import json

# Ensure repo root is on path so UCTC_sparse is importable
_repo_root = os.path.dirname(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from .preprocessing import standard_train_test_split
from .metric import calculate_ranking_metrics, calculate_global_kendall_tau
from .lazy_candidates import LazyCandidateGenerator

from UCTC_sparse import SparseUC, SparseTC


# ---------------------------------------------------------------------------
# Lazy UC prediction wrapper
# ---------------------------------------------------------------------------

class LazyUCPredictions:
    """
    Wraps UC/TC latent factors so predictions are computed on demand
    rather than materialising a full n_items × n_users matrix.
    """

    def __init__(self, user_factors, item_factors, model_type='UC'):
        self.user_factors = user_factors  # (n_users,)
        self.item_factors = item_factors  # (n_items,)
        self.model_type = model_type
        self.shape = (len(item_factors), len(user_factors))

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            # Single user: return all item scores
            return self.item_factors * self.user_factors[key]
        if isinstance(key, tuple) and len(key) == 2:
            user, items = key
            return self.item_factors[items] * self.user_factors[user]
        raise NotImplementedError("Only single-user or (user, items) indexing is supported.")

    def get_user_scores(self, user_idx):
        return self.item_factors * self.user_factors[user_idx]


def get_full_predictions_uc(model_type, train_matrix, device, epsilon=1e-9):
    """
    Compute UC/TC latent factors and return a LazyUCPredictions object.

    Args:
        model_type: 'UC' or 'TC'
        train_matrix: scipy sparse matrix (n_items, n_users)
        device: torch device
        epsilon: numerical stability constant

    Returns:
        LazyUCPredictions
    """
    print(f"[{model_type}] Computing latent factors...")
    tic = time()

    if not sparse.issparse(train_matrix):
        raise ValueError("train_matrix must be a scipy sparse matrix.")

    coo = train_matrix.tocoo()
    i = torch.LongTensor(np.vstack((coo.row, coo.col)))
    v = torch.FloatTensor(coo.data)
    train_tensor = torch.sparse_coo_tensor(i, v, torch.Size(coo.shape)).to(device)

    if model_type == 'UC':
        model = SparseUC(device, train_tensor, epsilon, mode_full=False)
        latent_1, latent_2 = model.UC()
    elif model_type == 'TC':
        model = SparseTC(device, train_tensor, epsilon, mode_full=False)
        latent_1, latent_2 = model.TC()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    item_factors = latent_1.float().cpu().detach().numpy()  # (n_items,)
    user_factors = latent_2.float().cpu().detach().numpy()  # (n_users,)

    del model, train_tensor
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f"[{model_type}] Latent factors ready in {time() - tic:.2f}s")
    return LazyUCPredictions(user_factors, item_factors, model_type=model_type)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_uc_ranking_evaluation(
    dataset_name,
    ratings,
    methods=None,
    k_values=None,
    device=None,
    random_state=42,
    logger=None,
    eval_mode='full_ranking',
    num_negatives=100,
    compute_uc_metrics=False,
    rating_threshold=4.0,
    output_dir='results',
    test_ratio=0.2,
):
    """
    Evaluate UC and TC on standard ranking metrics.

    Args:
        dataset_name: e.g. 'ML-1M'
        ratings: pd.DataFrame with columns [UserID, MovieID, Rating]
        methods: list, subset of ['UC'] (default: ['UC'])
        k_values: list of k values (default: [5, 10, 20])
        device: torch device (auto-detect if None)
        random_state: random seed
        logger: object with .log(str) method
        eval_mode: 'full_ranking' | 'rerank' | 'negative_sampling'
        num_negatives: negatives per user for negative_sampling mode
        compute_uc_metrics: whether to compute Kendall-tau
        rating_threshold: minimum rating treated as positive
        output_dir: directory to save CSV / JSON results
        test_ratio: fraction of interactions held out per user

    Returns:
        dict: per-method results
    """
    if methods is None:
        methods = ['UC']
    if k_values is None:
        k_values = [5, 10, 20]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class _SimpleLogger:
        def log(self, msg):
            print(msg)

    if logger is None:
        logger = _SimpleLogger()

    logger.log("=" * 80)
    logger.log(f"UC RANKING EVALUATION: {dataset_name}")
    logger.log("=" * 80)
    logger.log(f"Methods:   {methods}")
    logger.log(f"K values:  {k_values}")
    logger.log(f"Eval mode: {eval_mode}")
    logger.log(f"Seed:      {random_state}")
    logger.log(f"Device:    {device}")
    logger.log("")

    # ---- train/test split ----
    logger.log("Creating train/test split...")
    split = standard_train_test_split(
        ratings,
        test_ratio=test_ratio,
        rating_threshold=rating_threshold,
        random_state=random_state,
    )
    train_matrix = split['train_matrix']
    test_matrix = split['test_matrix']
    n_users = split['n_users']
    n_items = split['n_items']
    logger.log(f"Train: {train_matrix.shape}, nnz={train_matrix.nnz}")
    logger.log(f"Test:  {test_matrix.shape},  nnz={test_matrix.nnz}")
    logger.log(f"Users: {n_users}, Items: {n_items}")
    logger.log("")

    # ---- candidate generator ----
    candidates = LazyCandidateGenerator(
        train_matrix=train_matrix,
        test_matrix=test_matrix,
        n_users=n_users,
        n_items=n_items,
        eval_mode=eval_mode,
        num_negatives=num_negatives,
        rating_threshold=rating_threshold,
        random_state=random_state,
    )

    results = {}

    for method in methods:
        logger.log("-" * 80)
        logger.log(f"Evaluating: {method}")
        logger.log("-" * 80)

        try:
            t_start = time()
            predictions = get_full_predictions_uc(method, train_matrix, device)
            train_time = time() - t_start

            logger.log(f"[{method}] Computing ranking metrics...")
            t_inf = time()
            ranking_results = calculate_ranking_metrics(
                n_users, test_matrix, predictions, candidates, k_values=k_values
            )
            inf_time = time() - t_inf

            results[method] = {
                'metrics': ranking_results,
                'training_time': train_time,
                'inference_time': inf_time,
                'total_time': train_time + inf_time,
            }

            # Optional Kendall-tau
            if compute_uc_metrics:
                logger.log(f"[{method}] Computing Kendall-tau...")
                tau_result = calculate_global_kendall_tau(
                    train_matrix, test_matrix, predictions, min_common_users=5
                )
                results[method]['kendall_tau'] = tau_result['global_tau']
                logger.log(f"[{method}] Kendall-tau: {tau_result['global_tau']:.4f}")

            for k in k_values:
                p = ranking_results['precision'][f'at_{k}']
                r = ranking_results['recall'][f'at_{k}']
                n = ranking_results['ndcg'][f'at_{k}']
                logger.log(f"  k={k:2d}: P={p:.4f}  R={r:.4f}  NDCG={n:.4f}")
            logger.log(f"  Train: {train_time:.2f}s  Inference: {inf_time:.2f}s")
            logger.log("")

        except Exception as e:
            logger.log(f"[{method}] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Summary table ----
    logger.log("=" * 80)
    logger.log("SUMMARY")
    logger.log("=" * 80)
    summary_rows = []
    for method, res in results.items():
        row = {'Method': method}
        for k in k_values:
            row[f'P@{k}'] = res['metrics']['precision'][f'at_{k}']
            row[f'R@{k}'] = res['metrics']['recall'][f'at_{k}']
            row[f'N@{k}'] = res['metrics']['ndcg'][f'at_{k}']
        row['Train(s)'] = res['training_time']
        row['Infer(s)'] = res['inference_time']
        if 'kendall_tau' in res:
            row['Kendall-K'] = res['kendall_tau']
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    logger.log("\n" + summary_df.to_string(index=False))
    logger.log("")

    # ---- Save results ----
    result_dir = os.path.join(output_dir, f'{dataset_name}_ranking_results')
    os.makedirs(result_dir, exist_ok=True)

    csv_path = os.path.join(result_dir, f'summary_seed_{random_state}.csv')
    summary_df.to_csv(csv_path, index=False)
    logger.log(f"Results saved to: {csv_path}")

    json_path = os.path.join(result_dir, f'detailed_seed_{random_state}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.number) else x)
    logger.log(f"Details saved to: {json_path}")

    return results
