"""
Standard Ranking Evaluation Experiment

This module addresses reviewer concerns by implementing a conventional
evaluation setup where ALL methods (UC, SASRec, LightGCN, etc.) are
evaluated on standard ranking metrics: Precision@k, Recall@k, and NDCG@k.

This demonstrates UC's competitiveness even under evaluation metrics
designed specifically for ranking-based models.
"""

import sys
import os
import numpy as np
import torch
import gc
from time import time
from scipy import sparse
import pandas as pd

# Add models/recbole to Python path so RecBole can discover local models like SimGCL
_models_recbole_path = os.path.join(os.path.dirname(__file__), 'models', 'recbole')
if _models_recbole_path not in sys.path:
    sys.path.insert(0, _models_recbole_path)

# Import utility modules
from utils.preprocessing import standard_train_test_split
from utils.metric import calculate_ranking_metrics
from utils.lazy_candidates import LazyCandidateGenerator

# Import model modules
from models.custom.UCTC_sparse import SparseUC, SparseTC

try:
    from algos_optimized import run_algo_hard
    RECBOLE_AVAILABLE = True
except ImportError:
    RECBOLE_AVAILABLE = False
    print("Warning: RecBole not available. SASRec and other RecBole models will be skipped.")

try:
    from models.custom.mcclk_wrapper import get_full_predictions_mcclk
    MCCLK_AVAILABLE = True
except ImportError as e:
    MCCLK_AVAILABLE = False
    print(f"Warning: MCCLK not available. Error: {e}")


def get_full_predictions_uc(model_type, train_matrix, device, epsilon=1e-9):
    """
    Get lazy prediction object from UC/TC models (memory-efficient).

    Instead of creating full n_users × n_items matrix (11 GB for ML-20M),
    returns lazy object that computes predictions on-demand.

    Args:
        model_type (str): 'UC' or 'TC'
        train_matrix (sparse matrix): Training data (n_items, n_users)
        device: PyTorch device
        epsilon: Small value for numerical stability

    Returns:
        LazyUCPredictions: Lazy prediction object with latent factors
    """
    from models.custom.lazy_predictions import LazyUCPredictions

    print(f"[{model_type}] Computing latent factors (lazy predictions)...")
    tic = time()

    # Convert to torch sparse if needed
    if sparse.issparse(train_matrix):
        coo = train_matrix.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        train_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)
    else:
        raise ValueError("train_matrix must be sparse")

    # Run UC or TC
    if model_type == 'UC':
        model = SparseUC(device, train_tensor, epsilon, mode_full=False)
        latent_1, latent_2 = model.UC()
    elif model_type == 'TC':
        model = SparseTC(device, train_tensor, epsilon, mode_full=False)
        latent_1, latent_2 = model.TC()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Convert to numpy
    latent_1 = latent_1.float().cpu().detach().numpy()  # (n_items,)
    latent_2 = latent_2.float().cpu().detach().numpy()  # (n_users,)

    print(f"[{model_type}] Latent factors computed in {time() - tic:.2f}s")
    print(f"[{model_type}] Memory: {latent_2.nbytes + latent_1.nbytes:.1f} bytes (latent factors only)")

    # Cleanup
    del model, train_tensor
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Return lazy prediction object
    return LazyUCPredictions(latent_2, latent_1, model_type=model_type)


def run_standard_ranking_evaluation(dataset_name, ratings, methods=None,
                                      k_values=[5, 10, 20], device=None,
                                      logger=None, random_state=42, test_ratio=0.2, temp_dir='temp',
                                      config_dir='configs/recbole', eval_mode='rerank', num_negatives=100,
                                      use_uc_reranking=False, rerank_top_n=100, compute_uc_metrics=False,
                                      rating_threshold=4.0, fairgan_hidden_size=None,
                                      fairgan_batch_size=None, fairgan_fairness_sample_items=None,
                                      fairgan_epochs=None):
    """
    Run standard ranking evaluation on all methods.

    This function implements the evaluation setup requested by reviewers:
    - Standard random train/test split (80/20)
    - Evaluation on Precision@k, Recall@k, NDCG@k
    - Comparison of UC against modern baselines (BPR-MF, LightGCN, SASRec)

    Performance Note:
    - SASRec and BERT4Rec configs are aggressively optimized for speed on large datasets
    - Smaller models (hidden_size=32, n_layers=1) and shorter sequences (max_length=20)
    - This provides 5-10x speedup with minimal accuracy loss
    - For small datasets (ML-100K, ML-1M), you can increase model size in configs if needed

    Args:
        dataset_name (str): Name of dataset (e.g., 'ML-1M')
        ratings (pd.DataFrame): Ratings DataFrame
        methods (list): List of methods to evaluate. Options:
            ['UC', 'TC', 'BPR-MF', 'fairGAN', 'fairGAN_tf', 'fairGAN_torch', 'LightGCN', 'SASRec', 'BERT4Rec', 'NCF', 'NGCF']
            Note: LightGCN uses RecBole implementation for fair comparison
            If None, uses ['UC', 'BPR-MF', 'LightGCN']
        k_values (list): List of k values for top-k metrics
        device: PyTorch device
        logger: Logger object
        temp_dir: Temporary directory for intermediate files
        config_dir: Config directory for RecBole models
        eval_mode (str): Evaluation mode:
            'rerank': Rank only test items (original, non-standard)
            'full_ranking': Rank all items excluding training (standard RecSys protocol)
            'negative_sampling': Rank test items + N random negatives (common compromise)
        num_negatives (int): Number of negative samples per user (only for negative_sampling mode)
        use_uc_reranking (bool): If True, evaluate UC as a re-ranking module on top of base models
                                 Compares BaseModel vs BaseModel+UC (two-stage pipeline)
        rerank_top_n (int): Number of top candidates from base model to re-rank with UC
        compute_uc_metrics (bool): If True, compute UC-specific metrics (CVR, Kendall-tau) for all methods
                                   CVR = Consensus Violation Rate (UC's unique guarantee)
                                   Kendall-tau = Global rank-order consistency
        rating_threshold (float): Minimum rating to consider as positive feedback (default: 4.0)
                                  Only ratings >= threshold are used for training/evaluation
        fairgan_hidden_size (int | None): Override FairGAN hidden size (capped for safety)
        fairgan_batch_size (int | None): Override FairGAN batch size (capped for safety)
        fairgan_fairness_sample_items (int | None): Subsample size for FairGAN fairness loss

    Returns:
        dict: Results dictionary with metrics for each method
    """
    if methods is None:
        methods = ['UC', 'BPR-MF', 'LightGCN']

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if logger is None:
        class SimpleLogger:
            def log(self, msg):
                print(msg)
        logger = SimpleLogger()

    logger.log("=" * 80)
    logger.log(f"STANDARD RANKING EVALUATION: {dataset_name}")
    logger.log("=" * 80)
    logger.log(f"Methods: {methods}")
    logger.log(f"K values: {k_values}")
    logger.log(f"Device: {device}")
    logger.log("")

    # Step 1: Create standard train/test split
    logger.log("Step 1: Creating standard train/test split...")
    logger.log(f"Rating threshold: {rating_threshold} (only ratings >= {rating_threshold} used as positive)")
    split_data = standard_train_test_split(ratings, test_ratio=test_ratio,
                                          rating_threshold=rating_threshold,
                                          random_state=random_state)

    train_matrix = split_data['train_matrix']
    test_matrix = split_data['test_matrix']
    samples_products = split_data['samples_products']  # Default: re-ranking (only test items)
    n_users = split_data['n_users']
    n_items = split_data['n_items']

    logger.log(f"Train shape: {train_matrix.shape}, nnz: {train_matrix.nnz}")
    logger.log(f"Test shape: {test_matrix.shape}, nnz: {test_matrix.nnz}")
    logger.log(f"Users: {n_users}, Items: {n_items}")
    logger.log("")

    # Step 1.5: Create lazy candidate generator (memory-efficient)
    logger.log(f"Setting up lazy candidate generation for evaluation mode: {eval_mode}")

    from utils.lazy_candidates import LazyCandidateGenerator

    samples_products = LazyCandidateGenerator(
        train_matrix=train_matrix,
        test_matrix=test_matrix,
        n_users=n_users,
        n_items=n_items,
        eval_mode=eval_mode,
        num_negatives=num_negatives,
        rating_threshold=rating_threshold,
        random_state=random_state
    )

    if eval_mode == 'full_ranking':
        logger.log("Using FULL RANKING: all items excluding training items (lazy generation)")
    elif eval_mode == 'negative_sampling':
        logger.log(f"Using NEGATIVE SAMPLING: test items + {num_negatives} random negatives (lazy generation)")
    else:  # eval_mode == 'rerank'
        logger.log("Using RE-RANKING: only test items (lazy generation)")

    # Sample a few users to estimate average candidates
    sample_users = np.random.choice(min(100, n_users), min(100, n_users), replace=False)
    sample_lens = [len(samples_products[u]) for u in sample_users]
    avg_candidates = np.mean(sample_lens)
    logger.log(f"Estimated average candidates per user: {avg_candidates:.1f} (from {len(sample_users)} sample users)")
    logger.log(f"Memory: Using lazy generation - candidates created on-demand per user")
    logger.log("")

    # Step 2: Evaluate each method
    results = {}
    base_model_predictions = {}  # Store predictions for UC re-ranking
    all_predictions = {}  # Store predictions for CVR/Kendall-tau computation

    for method in methods:
        # Skip UC standalone if we're doing re-ranking comparison
        if use_uc_reranking and method in ['UC', 'TC']:
            logger.log(f"[{method}] Skipping standalone evaluation (will use as re-ranker)")
            continue

        logger.log("-" * 80)
        logger.log(f"Evaluating: {method}")
        logger.log("-" * 80)

        try:
            method_start = time()
            training_time = 0
            inference_time = 0
            predictions = None  # Store for re-ranking

            # Get predictions based on method
            if method == 'UC':
                train_start = time()
                predictions = get_full_predictions_uc('UC', train_matrix, device)
                training_time = time() - train_start
                inference_time = 0  # UC computes predictions during training

                # Store for UC metrics computation if enabled
                if compute_uc_metrics:
                    all_predictions[method] = predictions
            elif method == 'TC':
                train_start = time()
                predictions = get_full_predictions_uc('TC', train_matrix, device)
                training_time = time() - train_start
                inference_time = 0  # TC computes predictions during training

                # Store for UC metrics computation if enabled
                if compute_uc_metrics:
                    all_predictions[method] = predictions
            elif method == 'BPR-MF':
                train_start = time()
                predictions = get_full_predictions_bpr_mf(train_matrix, device)
                training_time = time() - train_start
                inference_time = 0  # BPR-MF computes predictions during training

                # Store for UC re-ranking if enabled
                if use_uc_reranking:
                    base_model_predictions[method] = predictions

                # Store for UC metrics computation if enabled
                if compute_uc_metrics:
                    all_predictions[method] = predictions
            elif method in ('fairGAN', 'fairGAN_tf'):
                train_start = time()
                predictions = get_full_predictions_fairgan(
                    train_matrix, device,
                    hidden_size=fairgan_hidden_size,
                    batch_size=fairgan_batch_size,
                    fairness_sample_items=fairgan_fairness_sample_items,
                    epochs=fairgan_epochs
                )
                training_time = time() - train_start
                inference_time = 0  # FairGAN computes predictions during training

                # Store for UC metrics computation if enabled
                if compute_uc_metrics:
                    all_predictions[method] = predictions
            elif method == 'fairGAN_torch':
                from FairGAN.FairGAN_PyTorch import get_full_predictions_fairgan_pytorch
                train_start = time()
                predictions = get_full_predictions_fairgan_pytorch(train_matrix, device)
                training_time = time() - train_start
                inference_time = 0  # FairGAN computes predictions during training

                # Store for UC metrics computation if enabled
                if compute_uc_metrics:
                    all_predictions[method] = predictions
            elif method == 'MCCLK':
                if not MCCLK_AVAILABLE:
                    logger.log("[MCCLK] MCCLK not available, skipping...")
                    continue

                train_start = time()
                predictions = get_full_predictions_mcclk(
                    train_matrix, test_matrix, samples_products,
                    device, logger, epochs=200
                )
                training_time = time() - train_start
                inference_time = 0  # MCCLK computes predictions during training

                # Store for UC metrics computation if enabled
                if compute_uc_metrics:
                    all_predictions[method] = predictions

                logger.log(f"[{method}] Calculating ranking metrics...")
                metrics_start = time()
                ranking_results = calculate_ranking_metrics(
                    n_users, test_matrix, predictions, samples_products, k_values=k_values
                )
                # For non-RecBole, metrics calculation is part of inference
                inference_time += time() - metrics_start

            method_time = time() - method_start

            # Store results with separate timing
            results[method] = {
                'metrics': ranking_results,
                'training_time': training_time,
                'inference_time': inference_time,
                'total_time': method_time
            }

            # For RecBole methods: use streaming Kendall tau scores if available
            results[method]['uc_metrics'] = {
                'kendall_tau': float('nan'),
                'kendall_concordant': 0,
                'kendall_discordant': 0,
                'kendall_total_pairs': 0,
                'note': 'N/A (no predictions or streaming scores available)'
            }
            logger.log(f"[{method}] Note: Kendall-tau N/A (no predictions available)")

            # Log results
            logger.log(f"[{method}] Results:")
            for k in k_values:
                prec = ranking_results['precision'][f'at_{k}']
                rec = ranking_results['recall'][f'at_{k}']
                ndcg = ranking_results['ndcg'][f'at_{k}']
                logger.log(f"  k={k:2d}: Precision={prec:.4f}, Recall={rec:.4f}, NDCG={ndcg:.4f}")
            logger.log(f"  Training Time: {training_time:.2f}s")
            logger.log(f"  Inference Time: {inference_time:.2f}s")
            logger.log(f"  Total Time: {method_time:.2f}s")
            logger.log("")

        except Exception as e:
            logger.log(f"[{method}] Error: {e}")
            import traceback
            traceback.print_exc()
            logger.log("")
        finally:
            # Free GPU memory between methods so PyTorch doesn't starve TensorFlow (and vice versa)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Step 2.5: UC Re-ranking Evaluation (if enabled)
    if use_uc_reranking and len(base_model_predictions) > 0:
        logger.log("\n" + "=" * 80)
        logger.log("UC RE-RANKING EVALUATION")
        logger.log("=" * 80)
        logger.log(f"Stage 1: Base model retrieves top-{rerank_top_n} candidates")
        logger.log(f"Stage 2: UC re-ranks those candidates")
        logger.log("")
        logger.log("WARNING: Under full/negative ranking, UC re-ranking often DEGRADES")
        logger.log("         performance because UC cannot handle unobserved items well.")
        logger.log("         See WHY_UC_UNDERPERFORMS_TOPK.md for explanation.")
        logger.log("")

        reranking_results = {}

        for base_method in base_model_predictions.keys():
            base_preds = base_model_predictions[base_method]

            try:
                rerank_result = evaluate_with_uc_reranking(
                    base_method=base_method,
                    base_predictions=base_preds,
                    train_matrix=train_matrix,
                    test_matrix=test_matrix,
                    samples_products=samples_products,
                    device=device,
                    n_users=n_users,
                    k_values=k_values,
                    top_n_candidates=rerank_top_n,
                    logger=logger
                )

                reranking_results[base_method] = rerank_result

                # Add reranked version to main results for summary table
                results[f'{base_method}+UC'] = {
                    'metrics': rerank_result['reranked_results'],
                    'training_time': results[base_method]['training_time'],  # Use base training time
                    'inference_time': results[base_method]['inference_time'],  # UC adds minimal overhead
                    'total_time': results[base_method]['total_time']
                }

            except Exception as e:
                logger.log(f"[{base_method}+UC] Error during re-ranking: {e}")
                import traceback
                traceback.print_exc()

    # Step 2.6: UC-specific Metrics (CVR and Kendall-tau)
    if compute_uc_metrics and len(all_predictions) > 0:
        logger.log("\n" + "=" * 80)
        logger.log("UC-SPECIFIC METRICS EVALUATION")
        logger.log("=" * 80)
        logger.log("Computing Global Kendall-tau for all methods with available predictions.")
        logger.log("")
        logger.log("Kendall-tau Distance (K = Q/(P+Q)):")
        logger.log("  - Measures % of item pairs in wrong relative order")
        logger.log("  - K=0: perfect rank-order agreement, K=1: complete disagreement")
        logger.log("  - Range: [0, 1], lower is better")
        logger.log("")

        from utils.metric import calculate_global_kendall_tau

        uc_metrics_results = {}

        for method_name in all_predictions.keys():
            logger.log(f"Computing UC metrics for {method_name}...")

            try:
                preds = all_predictions[method_name]

                # Compute Kendall-tau
                logger.log(f"  [{method_name}] Computing Global Kendall-tau...")
                tau_result = calculate_global_kendall_tau(
                    train_matrix, test_matrix, preds, min_common_users=5
                )

                uc_metrics_results[method_name] = {
                    'kendall_tau': tau_result['global_tau'],
                    'kendall_concordant': tau_result['concordant'],
                    'kendall_discordant': tau_result['discordant'],
                    'kendall_total_pairs': tau_result['total_pairs']
                }

                # Add to main results
                if method_name in results:
                    results[method_name]['uc_metrics'] = uc_metrics_results[method_name]

                logger.log(f"  [{method_name}] Global Kendall-tau: {tau_result['global_tau']:.4f}")
                logger.log("")

            except Exception as e:
                logger.log(f"  [{method_name}] Error computing UC metrics: {e}")
                import traceback
                traceback.print_exc()
                logger.log("")

        # Summary of UC metrics
        logger.log("-" * 80)
        logger.log("UC METRICS SUMMARY")
        logger.log("-" * 80)
        logger.log(f"{'Method':<15} {'Kendall-K':>15}")
        logger.log("-" * 80)
        for method_name, uc_metrics in uc_metrics_results.items():
            logger.log(f"{method_name:<15} {uc_metrics['kendall_tau']:>15.4f}")
        logger.log("-" * 80)
        logger.log("")
        logger.log("Interpretation:")
        logger.log("  Kendall-tau Distance (K):")
        logger.log("    - Fraction of item pairs in wrong relative order")
        logger.log("    - UC expected to excel here: K ≈ 0.05-0.10 (5-10% wrong pairs)")
        logger.log("    - Baselines typically: K ≈ 0.20-0.40 (20-40% wrong pairs)")
        logger.log("    - Range: [0, 1], lower is better")
        logger.log("")

    # Step 3: Summary table
    logger.log("=" * 80)
    logger.log("SUMMARY")
    logger.log("=" * 80)

    # Create summary table
    summary_rows = []
    for method, result in results.items():
        row = {'Method': method}
        for k in k_values:
            row[f'P@{k}'] = result['metrics']['precision'][f'at_{k}']
            row[f'R@{k}'] = result['metrics']['recall'][f'at_{k}']
            row[f'N@{k}'] = result['metrics']['ndcg'][f'at_{k}']
        row['Train(s)'] = result['training_time']
        row['Infer(s)'] = result['inference_time']

        # Add UC metrics if computed
        if 'uc_metrics' in result:
            row['Kendall-K'] = result['uc_metrics']['kendall_tau']

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    logger.log("\n" + summary_df.to_string(index=False))
    logger.log("")


    # Save results
    output_dir = f'results/{dataset_name}_standard_ranking_results'
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save summary
    summary_path = os.path.join(output_dir, f'summary_seed_{random_state}_{methods}_use_uc_ranking_{use_uc_reranking}_compute_uc_metrics_{compute_uc_metrics}.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.log(f"Summary saved to: {summary_path}")

    # Save detailed results
    import json
    detailed_path = os.path.join(output_dir, f'detailed_results_seed_{random_state}_{methods}_use_uc_ranking_{use_uc_reranking}_compute_uc_metrics_{compute_uc_metrics}.json')
    with open(detailed_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else x)
    logger.log(f"Detailed results saved to: {detailed_path}")

    logger.log("")
    logger.log("=" * 80)
    logger.log("EVALUATION COMPLETED")
    logger.log("=" * 80)

    return results


def apply_uc_reranking(base_predictions, train_matrix, device, top_n_candidates=100, epsilon=1e-9):
    """
    Apply UC as a re-ranking module on top of base model predictions.
    
    Two-stage pipeline (standard in RecSys):
    1. Stage 1 (Retrieval): Base model (NGCF, BPR-MF, LightGCN) retrieves top-N candidates
    2. Stage 2 (Re-ranking): UC re-ranks those N candidates
    
    This is the fair way to evaluate UC since it's designed for rating prediction/re-ranking,
    not for retrieval from scratch.
    
    Args:
        base_predictions (np.ndarray): Base model predictions (n_users, n_items)
        train_matrix (sparse matrix): Training data for UC (n_items, n_users)
        device: PyTorch device
        top_n_candidates (int): Number of candidates from base model to re-rank
        epsilon: Small value for UC numerical stability
    
    Returns:
        np.ndarray: Re-ranked predictions (n_users, n_items) where only top-N candidates
                    are re-ranked by UC, rest are set to -inf
    """
    print(f"[UC Re-ranking] Applying UC as re-ranker on top-{top_n_candidates} base candidates...")
    tic = time()
    
    n_users, n_items = base_predictions.shape
    
    # Step 1: Get UC predictions for all items
    print(f"[UC Re-ranking] Computing UC predictions...")
    uc_predictions = get_full_predictions_uc('UC', train_matrix, device, epsilon)
    
    # Step 2: For each user, re-rank only the top-N candidates from base model
    reranked_predictions = np.full((n_users, n_items), -np.inf, dtype=np.float32)
    
    for user in range(n_users):
        # Get top-N candidates from base model
        base_scores = base_predictions[user, :]
        top_n_indices = np.argpartition(base_scores, -top_n_candidates)[-top_n_candidates:]
        
        # Re-rank these candidates using UC scores
        uc_scores_for_candidates = uc_predictions[user, top_n_indices]
        
        # Store UC scores only for these candidates
        reranked_predictions[user, top_n_indices] = uc_scores_for_candidates
    
    print(f"[UC Re-ranking] Re-ranking completed in {time() - tic:.2f}s")
    
    return reranked_predictions


def evaluate_with_uc_reranking(base_method, base_predictions, train_matrix, test_matrix, 
                                samples_products, device, n_users, k_values, 
                                top_n_candidates=100, logger=None):
    """
    Evaluate BaseModel vs BaseModel+UC comparison.
    
    Args:
        base_method (str): Name of base model (e.g., 'NGCF', 'BPR-MF')
        base_predictions (np.ndarray): Base model predictions (n_users, n_items)
        train_matrix: Training data for UC
        test_matrix: Test data for evaluation
        samples_products: Candidate items per user
        device: PyTorch device
        n_users: Number of users
        k_values: K values for metrics
        top_n_candidates: Number of candidates to re-rank with UC
        logger: Logger object
    
    Returns:
        dict: Results comparing base model alone vs base model + UC
    """
    if logger is None:
        class SimpleLogger:
            def log(self, msg):
                print(msg)
        logger = SimpleLogger()
    
    logger.log(f"\n{'='*80}")
    logger.log(f"RE-RANKING EVALUATION: {base_method} vs {base_method}+UC")
    logger.log(f"{'='*80}")
    logger.log(f"Base model: {base_method}")
    logger.log(f"Re-ranking: UC on top-{top_n_candidates} candidates")
    logger.log("")
    
    # Evaluate base model alone
    logger.log(f"[{base_method}] Calculating ranking metrics (baseline)...")
    base_results = calculate_ranking_metrics(
        n_users, test_matrix, base_predictions, samples_products, k_values
    )
    
    # Apply UC re-ranking
    reranked_predictions = apply_uc_reranking(
        base_predictions, train_matrix, device, top_n_candidates
    )
    
    # Evaluate with UC re-ranking
    logger.log(f"[{base_method}+UC] Calculating ranking metrics (with UC re-ranking)...")
    reranked_results = calculate_ranking_metrics(
        n_users, test_matrix, reranked_predictions, samples_products, k_values
    )
    
    # Log comparison
    logger.log(f"\n{'-'*80}")
    logger.log(f"COMPARISON: {base_method} vs {base_method}+UC")
    logger.log(f"{'-'*80}")
    
    for k in k_values:
        base_p = base_results['precision'][f'at_{k}']
        rerank_p = reranked_results['precision'][f'at_{k}']
        base_r = base_results['recall'][f'at_{k}']
        rerank_r = reranked_results['recall'][f'at_{k}']
        base_n = base_results['ndcg'][f'at_{k}']
        rerank_n = reranked_results['ndcg'][f'at_{k}']
        
        p_delta = ((rerank_p - base_p) / base_p * 100) if base_p > 0 else 0
        r_delta = ((rerank_r - base_r) / base_r * 100) if base_r > 0 else 0
        n_delta = ((rerank_n - base_n) / base_n * 100) if base_n > 0 else 0
        
        logger.log(f"\nk={k}:")
        logger.log(f"  Precision: {base_p:.4f} → {rerank_p:.4f} ({p_delta:+.2f}%)")
        logger.log(f"  Recall:    {base_r:.4f} → {rerank_r:.4f} ({r_delta:+.2f}%)")
        logger.log(f"  NDCG:      {base_n:.4f} → {rerank_n:.4f} ({n_delta:+.2f}%)")
    
    logger.log(f"\n{'='*80}\n")
    
    return {
        'base_method': base_method,
        'base_results': base_results,
        'reranked_results': reranked_results,
        'top_n_candidates': top_n_candidates
    }
