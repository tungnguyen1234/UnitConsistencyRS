"""
Optimized RecBole algorithms for ranking-based evaluation.
Key optimizations:
1. Use BPR loss for faster ranking-focused training
2. Compute predictions ONLY for sampled test items (not full matrix)
3. Reduce epochs and use early stopping
4. Better batch size configuration
"""
import numpy as np
import torch
import gc
from scipy import sparse
from time import time
import os
import sys
import tempfile
import pandas as pd

# Import metric functions
from utils.metric import (
    calculate_macro_stats,
    calculate_bootstrap_stats
)

def get_predictions_for_samples(
    model,
    dataset,
    samples_products,  # dict: {user_ext: [item_ext, ...]}
    device,
    test_data=None,
    skip_empty_seq=True,        # <--- key: avoid SASRec crash
    min_items_per_user=2,
):
    """
    Compute predictions ONLY for sampled items per user.

    Supports:
      - Sequential models: SASRec, BERT4Rec, GRU4Rec, Caser, NARM, STAMP
        -> uses full_sort_predict(inter) then indexes sampled items
      - General models: LightGCN, NCF, etc.
        -> uses predict(inter) directly on (user,item) pairs

    Fixes:
      - External->internal mapping done via dataset.token2id (no giant id2token loops)
      - Sequential empty history: skip or clamp seq_len to 1
      - Range checks for user/item indices (prevent device-side assert)
      - Moves Interaction to device cleanly
    """
    model.eval()
    predictions = {}

    # --------- detect sequential model ----------
    model_type = model.__class__.__name__
    sequential_names = {'SASRec', 'BERT4Rec', 'GRU4Rec', 'Caser', 'NARM', 'STAMP'}
    is_sequential_model = model_type in sequential_names

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    # --------- sequential field names ----------
    item_list_field = None
    item_len_field = None

    if is_sequential_model:
        item_list_field = getattr(model, 'ITEM_SEQ', f'{iid_field}_list')
        item_len_field = getattr(model, 'ITEM_SEQ_LEN', 'item_length')

    # --------- build user interaction histories (internal user_id -> item_seq tensor) ----------
    user_interactions = {}
    if is_sequential_model and test_data is not None:
        # Iterate over test_data once to cache each user's item sequence.
        # NOTE: Depending on RecBole version, batches might already be on CPU.
        try:
            for batch in test_data:
                # batch is an Interaction
                if uid_field not in batch.interaction or item_list_field not in batch.interaction:
                    # If fields aren't present, we can't build histories from this dataloader
                    break

                batch_users = batch[uid_field].detach().cpu().tolist()
                batch_seqs = batch[item_list_field].detach().cpu()  # (B, L)

                for u, seq in zip(batch_users, batch_seqs):
                    user_interactions[int(u)] = seq.long()
        except Exception:
            # If anything goes wrong, proceed without cache; users will fall back to empty seq
            pass

    # --------- helpers: external -> internal id ----------
    def ext_user_to_internal(user_ext):
        # RecBole tokens are usually strings in the mapping vocab
        token = str(user_ext)
        try:
            return dataset.token2id(uid_field, token)
        except Exception:
            return None

    def ext_item_to_internal(item_ext):
        token = str(item_ext)
        try:
            return dataset.token2id(iid_field, token)
        except Exception:
            return None

    item_num = dataset.item_num
    user_num = dataset.user_num

    # --------- main loop ----------
    with torch.no_grad():
        for user_ext, item_ext_list in samples_products.items():
            user_internal = ext_user_to_internal(user_ext)
            if user_internal is None:
                continue
            if not (0 <= user_internal < user_num):
                continue

            # Map sampled items ext->internal (filter invalid + padding id 0)
            valid_items = []
            valid_items_ext = []
            for item_ext in item_ext_list:
                iid = ext_item_to_internal(item_ext)
                if iid is None:
                    continue
                # 0 is padding in RecBole; also guard range
                if 0 < iid < item_num:
                    valid_items.append(iid)
                    valid_items_ext.append(item_ext)

            if len(valid_items) < min_items_per_user:
                continue

            # ---- sequential models ----
            if is_sequential_model:
                # Retrieve history sequence for this internal user id
                item_seq = user_interactions.get(user_internal, None)
                if item_seq is None:
                    max_len = getattr(model, 'max_seq_length', 50)
                    item_seq = torch.zeros(max_len, dtype=torch.long)

                item_seq = item_seq.long().view(1, -1)  # (1, L)

                # Sequence length = count of non-zero tokens
                seq_len = int((item_seq != 0).sum().item())

                # Critical: many sequential models crash when seq_len == 0
                if seq_len == 0:
                    if skip_empty_seq:
                        continue
                    else:
                        seq_len = 1  # clamp; predictions will be low-signal but won't crash

                inter = Interaction({
                    uid_field: torch.tensor([user_internal], dtype=torch.long),
                    item_list_field: item_seq,
                    item_len_field: torch.tensor([seq_len], dtype=torch.long),
                }).to(device)

                # full_sort_predict returns shape (1, item_num) or (item_num,)
                all_scores = model.full_sort_predict(inter)
                if all_scores.dim() == 2:
                    all_scores = all_scores[0]
                all_scores = all_scores.detach().cpu().numpy()

                # Index sampled items
                scores = all_scores[valid_items]

            # ---- general models ----
            else:
                # Use full_sort_predict (one forward pass for all items) then index
                inter = Interaction({
                    uid_field: torch.tensor([user_internal], dtype=torch.long),
                }).to(device)
                try:
                    all_scores = model.full_sort_predict(inter)
                    if all_scores.dim() == 2:
                        all_scores = all_scores[0]
                    all_scores = all_scores.detach().cpu().numpy()
                    scores = all_scores[valid_items]
                except (NotImplementedError, AttributeError, RuntimeError):
                    user_tensor = torch.tensor([user_internal] * len(valid_items), dtype=torch.long)
                    item_tensor = torch.tensor(valid_items, dtype=torch.long)
                    inter = Interaction({
                        uid_field: user_tensor,
                        iid_field: item_tensor,
                    }).to(device)
                    scores = model.predict(inter).detach().cpu().numpy()

            predictions[user_ext] = {"items": valid_items_ext, "scores": scores}

    return predictions



def calculate_kendall_tau_from_predictions(predictions, test_r, samples_products):
    """
    Calculate Kendall tau scores from prediction dict (optimized version).
    """
    from utils.metric import norm_kendall_tau

    macro_scores = []

    for user, pred_data in predictions.items():
        if user not in samples_products:
            continue

        items = pred_data['items']
        scores = pred_data['scores']

        if len(items) < 2:
            continue

        # Get actual ratings
        if sparse.issparse(test_r):
            # test_r is (n_items, n_users), so we need test_r[items, user]
            actual = test_r[items, user].toarray().flatten()
        else:
            actual = test_r[items, user]

        # Calculate kendall tau for this user
        score = norm_kendall_tau(actual, scores)
        macro_scores.append(score)

    return macro_scores


def compute_predictions_and_metrics_streaming(
    model, dataset, samples_products, test_r, n_users, device,
    test_data=None, k_values=[5, 10, 20], logger=None, algo_name='',
):
    """
    Compute predictions and ranking metrics in a streaming per-user fashion.
    Never accumulates all predictions in memory.

    For ML-20M full_ranking: avoids storing 136K users × 20K items (~32GB).
    Instead, processes one user at a time and accumulates metric scores.

    Returns:
        tuple: (ranking_metrics, kendall_tau_scores, elapsed_time)
    """
    tic = time()

    model.eval()

    # ---- detect model type ----
    model_type = model.__class__.__name__
    sequential_names = {'SASRec', 'BERT4Rec', 'GRU4Rec', 'Caser', 'NARM', 'STAMP'}
    is_sequential_model = model_type in sequential_names

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    item_num = dataset.item_num
    user_num = dataset.user_num

    # ---- sequential model setup ----
    item_list_field = None
    item_len_field = None
    user_interactions = {}

    if is_sequential_model:
        item_list_field = getattr(model, 'ITEM_SEQ', f'{iid_field}_list')
        item_len_field = getattr(model, 'ITEM_SEQ_LEN', 'item_length')
        if test_data is not None:
            try:
                for batch in test_data:
                    if uid_field not in batch.interaction or item_list_field not in batch.interaction:
                        break
                    batch_users = batch[uid_field].detach().cpu().tolist()
                    batch_seqs = batch[item_list_field].detach().cpu()
                    for u, seq in zip(batch_users, batch_seqs):
                        user_interactions[int(u)] = seq.long()
            except Exception:
                pass

    # ---- ID mapping helpers ----
    def ext_to_internal_user(user_ext):
        try:
            return dataset.token2id(uid_field, str(user_ext))
        except Exception:
            return None

    def ext_to_internal_item(item_ext):
        try:
            return dataset.token2id(iid_field, str(item_ext))
        except Exception:
            return None

    # ---- metric accumulators ----
    # Per-k accumulators
    precision_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    ndcg_scores = {k: [] for k in k_values}
    kendall_tau_scores = []

    # Handle test_r orientation
    if sparse.issparse(test_r):
        n_items_test, n_users_test = test_r.shape
        if n_items_test == n_users and n_users_test != n_users:
            test_r_items_users = test_r  # already (n_items, n_users)
        else:
            test_r_items_users = test_r  # assume (n_items, n_users)
    else:
        test_r_items_users = test_r

    users_processed = 0

    with torch.no_grad():
        for user_ext, item_ext_list in samples_products.items():
            user_internal = ext_to_internal_user(user_ext)
            if user_internal is None or not (0 <= user_internal < user_num):
                continue

            # Map items ext -> internal
            valid_items = []
            valid_items_ext = []
            for item_ext in item_ext_list:
                iid = ext_to_internal_item(item_ext)
                if iid is not None and 0 < iid < item_num:
                    valid_items.append(iid)
                    valid_items_ext.append(item_ext)

            if len(valid_items) < 2:
                continue

            # ---- get scores ----
            if is_sequential_model:
                item_seq = user_interactions.get(user_internal, None)
                if item_seq is None:
                    max_len = getattr(model, 'max_seq_length', 50)
                    item_seq = torch.zeros(max_len, dtype=torch.long)
                item_seq = item_seq.long().view(1, -1)
                seq_len = int((item_seq != 0).sum().item())
                if seq_len == 0:
                    continue
                inter = Interaction({
                    uid_field: torch.tensor([user_internal], dtype=torch.long),
                    item_list_field: item_seq,
                    item_len_field: torch.tensor([seq_len], dtype=torch.long),
                }).to(device)
                all_scores = model.full_sort_predict(inter)
                if all_scores.dim() == 2:
                    all_scores = all_scores[0]
                all_scores = all_scores.detach().cpu().numpy()
                scores = all_scores[valid_items]
            else:
                # Use full_sort_predict for general models (LightGCN, NGCF, etc.)
                # This computes scores for ALL items in one forward pass (embedding dot product)
                # Much faster than calling predict() with 20K user-item pairs per user
                inter = Interaction({
                    uid_field: torch.tensor([user_internal], dtype=torch.long),
                }).to(device)
                try:
                    all_scores = model.full_sort_predict(inter)
                    if all_scores.dim() == 2:
                        all_scores = all_scores[0]
                    all_scores = all_scores.detach().cpu().numpy()
                    scores = all_scores[valid_items]
                except (NotImplementedError, AttributeError, RuntimeError):
                    # Fallback to predict() for models that don't support full_sort_predict
                    user_tensor = torch.tensor([user_internal] * len(valid_items), dtype=torch.long)
                    item_tensor = torch.tensor(valid_items, dtype=torch.long)
                    inter = Interaction({
                        uid_field: user_tensor,
                        iid_field: item_tensor,
                    }).to(device)
                    scores = model.predict(inter).detach().cpu().numpy()

            items = np.array(valid_items_ext)

            # ---- compute ranking metrics inline ----
            # Get test ratings for this user
            if sparse.issparse(test_r_items_users):
                test_r_user = test_r_items_users[:, user_ext].toarray().flatten()
            else:
                test_r_user = test_r_items_users[:, user_ext].flatten()

            sampled_items_set = set(items.tolist())
            all_relevant_items = set(np.where(test_r_user > 0)[0])
            relevant_sample_items = sampled_items_set.intersection(all_relevant_items)

            if relevant_sample_items:
                sorted_indices = np.argsort(-scores)
                ranked_items = items[sorted_indices].tolist()

                for k in k_values:
                    ranked_k = ranked_items[:k]
                    hits_k = sum(1 for it in ranked_k if it in relevant_sample_items)
                    precision_scores[k].append(hits_k / k if k > 0 else 0.0)
                    recall_scores[k].append(hits_k / len(relevant_sample_items) if relevant_sample_items else 0.0)
                    # NDCG
                    rels = [test_r_user[it] if it in relevant_sample_items else 0.0 for it in ranked_k]
                    ideal_rels = sorted([test_r_user[it] for it in relevant_sample_items], reverse=True)[:len(ranked_k)]
                    if len(ideal_rels) < len(ranked_k):
                        ideal_rels.extend([0.0] * (len(ranked_k) - len(ideal_rels)))
                    dcg = sum((2**r - 1) / np.log2(2 + i) for i, r in enumerate(rels))
                    idcg = sum((2**r - 1) / np.log2(2 + i) for i, r in enumerate(ideal_rels))
                    ndcg_scores[k].append(dcg / idcg if idcg > 0 else 0.0)

            # ---- Kendall tau per user (only items with test ratings) ----
            # K = Q / (P + Q), where P = concordant, Q = discordant
            if relevant_sample_items and len(relevant_sample_items) >= 2:
                relevant_list = np.array(sorted(relevant_sample_items))
                if sparse.issparse(test_r_items_users):
                    kt_actual = test_r_items_users[relevant_list, user_ext].toarray().flatten()
                else:
                    kt_actual = test_r_items_users[relevant_list, user_ext].flatten()
                item_to_score = dict(zip(items.tolist(), scores))
                kt_preds = np.array([item_to_score.get(it, 0.0) for it in relevant_list])
                # Vectorized pairwise concordance/discordance
                n_kt = len(relevant_list)
                ii, jj = np.triu_indices(n_kt, k=1)
                actual_diff = np.sign(kt_actual[ii] - kt_actual[jj])
                pred_diff = np.sign(kt_preds[ii] - kt_preds[jj])
                non_tie = actual_diff != 0
                P = int(((actual_diff == pred_diff) & non_tie).sum())
                Q = int(((actual_diff != pred_diff) & non_tie & (pred_diff != 0)).sum())
                if P + Q > 0:
                    kendall_tau_scores.append(Q / (P + Q))

            users_processed += 1
            if users_processed % 2000 == 0 and logger:
                elapsed = time() - tic
                avg_kt = np.mean(kendall_tau_scores) if kendall_tau_scores else float('nan')
                avg_prec = np.mean(precision_scores[k_values[0]]) if precision_scores[k_values[0]] else float('nan')
                avg_recall = np.mean(recall_scores[k_values[0]]) if recall_scores[k_values[0]] else float('nan')
                avg_ndcg = np.mean(ndcg_scores[k_values[0]]) if ndcg_scores[k_values[0]] else float('nan')
                logger.log(
                    f"  [{algo_name}] {users_processed} users | "
                    f"{elapsed:.0f}s | "
                    f"KT={avg_kt:.4f} | "
                    f"P@{k_values[0]}={avg_prec:.4f} R@{k_values[0]}={avg_recall:.4f} N@{k_values[0]}={avg_ndcg:.4f}"
                )

    # ---- assemble ranking metrics ----
    ranking_metrics = {'precision': {}, 'recall': {}, 'ndcg': {}}
    for k in k_values:
        ranking_metrics['precision'][f'at_{k}'] = np.mean(precision_scores[k]) if precision_scores[k] else 0.0
        ranking_metrics['precision'][f'scores_{k}'] = precision_scores[k]
        ranking_metrics['recall'][f'at_{k}'] = np.mean(recall_scores[k]) if recall_scores[k] else 0.0
        ranking_metrics['recall'][f'scores_{k}'] = recall_scores[k]
        ranking_metrics['ndcg'][f'at_{k}'] = np.mean(ndcg_scores[k]) if ndcg_scores[k] else 0.0
        ranking_metrics['ndcg'][f'scores_{k}'] = ndcg_scores[k]

    elapsed = time() - tic
    if logger:
        logger.log(f"  [{algo_name}] Streaming complete: {users_processed} users in {elapsed:.2f}s")

    return ranking_metrics, kendall_tau_scores, elapsed


def run_algo_optimized(n_u, r_train, test_r, samples_products, device, logger,
                       algo_name='SASRec', temp_dir=None, config_dir=None, k_values=[5, 10, 20],
                       compute_uc_metrics=False):
    """
    Optimized RecBole algorithm runner for ranking evaluation.

    Key optimizations:
    1. Single data file creation (not 3 copies)
    2. Optimized config for speed (smaller model, fewer epochs)
    3. Predict ONLY on sampled items (NEVER creates full matrix)
    4. Early stopping
    5. Aggressive memory cleanup for large datasets

    Args:
        k_values (list): List of k values for computing Precision@k, Recall@k, NDCG@k
        compute_uc_metrics (bool): If True, store sparse predictions dict
                                   for CVR/Kendall-tau computation (memory-efficient)

    Returns:
        dict: Results including ranking_metrics, timing, kendall_tau_scores
              If compute_uc_metrics=True, also includes 'predictions' (dict format, sparse)
    """
    from utils.metric import calculate_ranking_metrics_from_predictions_dict

    tic_total = time()

    # Get matrix shape
    if sparse.issparse(r_train):
        n_items, n_users = r_train.shape
    elif isinstance(r_train, np.ndarray):
        n_items, n_users = r_train.shape
    else:
        raise ValueError("r_train must be scipy sparse or numpy array")

    # Handle temp directory
    if temp_dir is None:
        temp_dir_context = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_context.__enter__()
        should_cleanup = True
    else:
        temp_dir_context = None
        should_cleanup = False
        os.makedirs(temp_dir, exist_ok=True)

    model = None
    dataset = None
    train_data = None
    valid_data = None
    test_data = None

    try:
        # Prepare data directory
        data_path = os.path.join(temp_dir, "mydata")
        os.makedirs(data_path, exist_ok=True)

        # Convert matrix - NOTE: r_train is (n_items, n_users), need to transpose
        logger.log(f"[{algo_name}] Preparing data...")
        r_train_T = r_train.T if sparse.issparse(r_train) else r_train.T
        convert_matrix_to_recbole_format(r_train_T, data_path, filename='mydata.inter')

        # Clear intermediate data
        del r_train_T
        gc.collect()

        # Train model
        train_start = time()
        logger.log(f"[{algo_name}] Training started")

        config_file = None
        if config_dir:
            config_file = os.path.join(config_dir, f'{algo_name.lower()}_local.yaml')
            # Fall back to base configs/recbole/ if dataset-specific config doesn't exist
            if not os.path.exists(config_file):
                base_config_dir = os.path.dirname(config_dir)  # e.g. configs/recbole
                fallback = os.path.join(base_config_dir, f'{algo_name.lower()}_local.yaml')
                if os.path.exists(fallback):
                    config_file = fallback
        model, dataset, train_data, valid_data, test_data, config = train_recbole_model_optimized(
            algo_name, temp_dir, device, config_file
        )
        training_time = time() - train_start
        logger.log(f"[{algo_name}] Training completed in {training_time:.2f} seconds")

        # Aggressive memory cleanup after training
        # NOTE: Keep test_data for sequential models (needed for interaction history)
        # Set to None instead of del to avoid NameError in exception handler
        train_data = None
        valid_data = None
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        logger.log(f"[{algo_name}] Memory cleaned after training")

        # Get predictions and compute metrics in streaming fashion
        # NEVER accumulate all predictions in memory (would be ~32GB for ML-20M full_ranking)
        prediction_start = time()
        logger.log(f"[{algo_name}] Computing predictions for {len(samples_products)} users (streaming)...")

        ranking_metrics, kendall_tau_scores, prediction_time = compute_predictions_and_metrics_streaming(
            model=model,
            dataset=dataset,
            samples_products=samples_products,
            test_r=test_r,
            n_users=n_users,
            device=device,
            test_data=test_data,
            k_values=k_values,
            logger=logger,
            algo_name=algo_name,
        )
        logger.log(f"[{algo_name}] Predictions + metrics completed in {prediction_time:.2f} seconds")

        # Clean up test_data after predictions (no longer needed)
        test_data = None
        gc.collect()

        # Log ranking metrics
        for k in k_values:
            prec = ranking_metrics['precision'].get(f'at_{k}', 0.0)
            rec = ranking_metrics['recall'].get(f'at_{k}', 0.0)
            ndcg = ranking_metrics['ndcg'].get(f'at_{k}', 0.0)
            logger.log(f"[{algo_name}] k={k}: P@{k}={prec:.4f}, R@{k}={rec:.4f}, N@{k}={ndcg:.4f}")

        ranking_time = prediction_time  # Metrics computed inline with predictions

        # No predictions dict stored - everything computed inline
        predictions_for_uc = None

        # Final cleanup
        model = None
        dataset = None
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        logger.log(f"[{algo_name}] Final memory cleanup completed")

    except Exception as e:
        # Ensure cleanup even on error
        # Set to None to release memory
        if model is not None:
            model = None
        if dataset is not None:
            dataset = None
        if train_data is not None:
            train_data = None
        if valid_data is not None:
            valid_data = None
        if test_data is not None:
            test_data = None
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Extract file path from traceback
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        for frame in reversed(tb):
            if 'recbole' in frame.filename:
                logger.log(f"[{algo_name}] Error in: {frame.filename}:{frame.lineno}")
                break

        logger.log(f"[{algo_name}] Error: {str(e)}")
        raise e

    finally:
        if should_cleanup and temp_dir_context is not None:
            temp_dir_context.__exit__(None, None, None)

    total_time = time() - tic_total
    logger.log(f"[{algo_name}] Total time: {total_time:.2f} seconds")

    results = {
        'kendall_tau_scores': kendall_tau_scores,
        'kendall_tau_stats': {
            'normal_macro': calculate_macro_stats(kendall_tau_scores),
            'bootstrap_macro': calculate_bootstrap_stats(kendall_tau_scores)
        },
        'ranking_metrics': ranking_metrics,
        'timing': {
            'training_seconds': training_time,
            'prediction_seconds': prediction_time,
            'ranking_metrics_seconds': ranking_time,
            'total_seconds': total_time
        }
    }

    # Add predictions dict if requested for UC metrics
    if predictions_for_uc is not None:
        results['predictions'] = predictions_for_uc

    return results


# Aliases for compatibility with existing code
def run_algo_hard(n_u, r_train, test_r, samples_products, device, logger,
                  algo_name='LightGCN', temp_dir=None, config_dir=None, k_values=[5, 10, 20],
                  compute_uc_metrics=False):
    """Drop-in replacement for the original run_algo_hard."""
    return run_algo_optimized(
        n_u, r_train, test_r, samples_products, device, logger,
        algo_name=algo_name, temp_dir=temp_dir, config_dir=config_dir, k_values=k_values,
        compute_uc_metrics=compute_uc_metrics
    )


def run_algo_easy(n_u, r_train, test_r, samples_products, device, logger,
                  algo_name='LightGCN', temp_dir=None, config_dir=None, k_values=[5, 10, 20]):
    """Drop-in replacement for the original run_algo_easy."""
    # Convert dense to sparse for unified processing
    if isinstance(r_train, np.ndarray):
        r_train = sparse.csr_matrix(r_train)
    elif torch.is_tensor(r_train):
        r_train = sparse.csr_matrix(r_train.cpu().numpy())

    return run_algo_optimized(
        n_u, r_train, test_r, samples_products, device, logger,
        algo_name=algo_name, temp_dir=temp_dir, config_dir=config_dir, k_values=k_values
    )
