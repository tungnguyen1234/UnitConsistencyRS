# Evaluation code
import numpy as np
from time import time
from scipy import sparse
from scipy.stats import kendalltau


test_values = np.array([1.0, 1.0])

def kendall_tau_distance_scipy(values1, values2):
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    tau, p_value = kendalltau(values1, values2)
    normalized_distance = 1 - (tau + 1) / 2
    return normalized_distance


def kendall_tau_distance_two_elements(ground_truth, prediction):
    """
    Compute the normalized Kendall tau distance for two elements.
    
    Args:
        a (np.ndarray): Array of actual ratings (length 2).
        b (np.ndarray): Array of predicted ratings (length 2).
    
    Returns:
        float: Kendall tau distance (0.0 or 1.0).
    """
    if ground_truth[0] == ground_truth[1]:
        return 0.0  # No distance if actual ratings are tied
    
    if ground_truth[0] != ground_truth[1] and prediction[0] == prediction[1]:
        return 1.0  # No distance if actual ratings are tied
    
    # Check for discordant pairs
    if (ground_truth[0] < ground_truth[1] and prediction[0] > prediction[1]) or (ground_truth[0] > ground_truth[1] and prediction[0] < prediction[1]):
        return 1.0
    
    return 0.0


def calculate_scores_vectorized(n_u, test_r, samples_products, method=None, batch_size=1000, latent_1=None, latent_2=None, r_train=None, Q_side_result=None):
    """
    Calculates macro Kendall Tau scores for a given user-item matrix using batch processing.
    
    Args:
        n_u (int): Number of users.
        test_r (scipy.sparse.csr_matrix or np.ndarray): User-item rating matrix.
        latent_1 (np.ndarray): Latent factors for users.
        latent_2 (np.ndarray): Latent factors for items.
        samples_products (list of lists or np.ndarray): List of product samples for each user.
        method (str, optional): Method for calculating predictions ('TC' or 'UC'). Defaults to None.
        batch_size (int, optional): Number of users to process per batch. Defaults to 1000.
    
    Returns:
        tuple: (macro_scores, all_test_r, all_pre)
            - macro_scores (list of float): Kendall Tau distances per user.
    """
    macro_scores = []

    tic = time()

    for batch_start in range(0, len(samples_products), batch_size):
        batch_end = min(batch_start + batch_size, n_u)
        batch_users = range(batch_start, batch_end)
        current_batch_size = batch_end - batch_start

        # Initialize arrays to hold test and prediction values
        test_r_batch = np.empty((current_batch_size, 2), dtype=float)
        pre_N_batch = np.empty((current_batch_size, 2), dtype=float)

        for i, user in enumerate(batch_users):
            if user not in samples_products:
                continue
            samples = samples_products[user]
            
            if len(samples) < 2:
                continue

            samples_products_np = np.array(samples)
        
            # Extract actual ratings
            if sparse.issparse(test_r):
                test_r_N = test_r[user, samples_products_np].toarray().flatten()
            else:
                test_r_N = test_r[user, samples_products_np]

            # Compute predictions based on the selected method
            if method == "TC":
                # pre_N = latent_1[user] + latent_2[samples_products_np]
                pre_N = test_values
            elif method == "UC":
                # pre_N = latent_1[user] * latent_2[samples_products_np]
                pre_N = test_values
            elif method == "rankSVD":
                pre_N = (r_train[user] @ Q_side_result)
                pre_N = pre_N[samples_products_np].detach().cpu().to_dense()
            else:
                raise ValueError("Method must be 'TC' or 'UC'.")

            # Store in batch arrays
            test_r_batch[i] = test_r_N
            pre_N_batch[i] = pre_N

        # Vectorized Kendall Tau distance calculation for the batch
        a1 = test_r_batch[:, 0]
        a2 = test_r_batch[:, 1]
        b1 = pre_N_batch[:, 0]
        b2 = pre_N_batch[:, 1]

        # Compute discordant pairs
        discordant = ((a1 < a2) & (b1 > b2)) | ((a1 > a2) & (b1 < b2)) | ((a1 == a2) & (b1 != b2))
        scores = discordant.astype(float)  # 1.0 for discordant, 0.0 otherwise

        # Append results
        macro_scores.extend(scores.tolist())

        if batch_start % 10 == 0:
            print(f'Kendall tau calculation for batch {batch_start} in {time() - tic:.2f} seconds')

    return macro_scores


def get_mean(arr):
  arr1 = np.array([a[0] for a in arr])
  return np.mean(arr1), np.std(arr1)

def norm_kendall_tau(values1, values2):
    """Compute the normalized Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    
    # Convert inputs to numpy arrays if they aren't already
    values1 = np.asarray(values1)
    values2 = np.asarray(values2)
    
    if np.all(values1 == values1[0]) and np.all(values2 == values2[0]):
        return 0.0
    
    order = np.argsort(values1)
    a = values1[order]
    b = values2[order]
    
    result = kendall_tau_distance_two_elements(ground_truth=a, prediction=b)
    return result


def calculate_macro_stats(scores):
    """Calculate mean and standard error for macro Kendall's tau scores."""
    return {
        'mean': np.mean(scores),
        'std': np.std(scores) / np.sqrt(len(scores))
    }

def calculate_bootstrap_stats(scores, n_bootstrap=1000):
    """Calculate bootstrap statistics for macro Kendall's tau scores."""
    bootstrap_means = [np.mean(np.random.choice(scores, size=len(scores), replace=True)) 
                       for _ in range(n_bootstrap)]
    return {
        'mean': np.mean(bootstrap_means),
        'ci_lower': np.percentile(bootstrap_means, 2.5),
        'ci_upper': np.percentile(bootstrap_means, 97.5)
    }

def calculate_scores_UCTC(n_u, test_r, latent_1, latent_2, samples_products, method=None):
    macro_scores = []
    count = 0
    dis_count = 0
    tic = time()

    for user in list(samples_products.keys()):
        if len(samples_products[user]) <= 1:
            continue
        
        tic = time()
        samples_products_np = np.array(samples_products[user])
        if sparse.issparse(test_r):
            test_r_N = test_r[user, samples_products_np].toarray().flatten()
        else:  # dense numpy array
            test_r_N = test_r[user, samples_products_np]
        if method == "TC":
            # pre_N = (latent_1[user, None] + latent_2[None, samples_products_np]).flatten()
            pre_N = test_values
        elif method == "UC":
            # pre_N = (latent_1[user, None] * latent_2[None, samples_products_np]).flatten()
            pre_N = test_values

        if pre_N[0] == pre_N[1]:
            count += 1
        score = norm_kendall_tau(test_r_N, pre_N)
        dis_count += score
        macro_scores.append(score)
        
    print(f'Initiate kendal tau calculation in {time() - tic:.2f} seconds')
    print("The number of equal pairs is", count)
    print("The percentage of discordant pairs is", dis_count/len(samples_products.keys()))
    return macro_scores

def calculate_scores(n_u, test_r, samples_products, pre=None, r_train=None, Q=None):
    macro_scores = []
    count = 0
    dis_count = 0

    for user in list(samples_products.keys()):
        if len(samples_products[user]) <= 1:
            continue
        
        samples_products_np = np.array(samples_products[user])
        if sparse.issparse(test_r):
            test_r_N = test_r[user, samples_products_np].toarray().flatten()
        else:  # dense numpy array
            test_r_N = test_r[user, samples_products_np]
        if sparse.issparse(test_r):
            pre_N = r_train[user].to_dense() @ Q @ Q.t()
            pre_N = pre_N[samples_products_np]
        else:    
            pre_N = pre[user, samples_products_np]

        if pre_N[0] == pre_N[1]:
            count += 1
        score = norm_kendall_tau(test_r_N, pre_N)
        dis_count += score
        macro_scores.append(score)
    
    print("The number of equal pairs is", count)
    print("The percentage of discordant pairs is", dis_count/len(samples_products.keys()))
    return macro_scores


def calculate_ranking_metrics(n_u, test_r, predictions, samples_products, k_values=[10, 20]):
    """
    Calculate precision, recall, and NDCG metrics for ranking evaluation.
    
    Args:
        n_u (int): Number of users.
        test_r (scipy.sparse.csr_matrix or np.ndarray): Test matrix of shape (n_r, n_u) or (n_u, n_r).
        predictions (np.ndarray): Prediction matrix of shape (n_u, n_r).
        samples_products (dict): Dictionary mapping user_id to list of item indices to evaluate.
        k_values (list): List of k values for top-k evaluation (e.g., [10, 20]).
    
    Returns:
        dict: Dictionary containing precision, recall, and ndcg metrics at different k values.
            Format: {
                'precision': {'at_10': mean, 'at_20': mean, 'scores_10': list, 'scores_20': list},
                'recall': {'at_10': mean, 'at_20': mean, 'scores_10': list, 'scores_20': list},
                'ndcg': {'at_10': mean, 'at_20': mean, 'scores_10': list, 'scores_20': list}
            }
    """
    # Handle test_r format - if it's transposed (n_r, n_u), we need to transpose it
    if sparse.issparse(test_r):
        if test_r.shape[0] != n_u and test_r.shape[1] == n_u:
            test_r = test_r.T
    else:
        if test_r.shape[0] != n_u and test_r.shape[1] == n_u:
            test_r = test_r.T
    
    results = {}
    
    for k in k_values:
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        
        for user in list(samples_products.keys()):
            if len(samples_products[user]) <= 1:
                continue
            
            samples_products_np = np.array(samples_products[user])
            
            # Get full user ratings to determine relevant items across all products
            if sparse.issparse(test_r):
                test_r_user = test_r.getrow(user).toarray().flatten()
            else:  # dense numpy array
                test_r_user = np.asarray(test_r[user]).flatten()
            
            # Relevant items are those the user rated positively and that were sampled
            sampled_items_set = set(samples_products_np.tolist())
            all_relevant_items = set(np.where(test_r_user > 0)[0])
            relevant_sample_items = sampled_items_set.intersection(all_relevant_items)
            
            if not relevant_sample_items:
                continue
            
            # Get ranked items from sampled products (sorted by prediction score, descending order)
            user_predictions = predictions[user, samples_products_np]
            # Get indices that would sort predictions in descending order
            sorted_indices = np.argsort(-user_predictions)
            ranked_items = samples_products_np[sorted_indices].tolist()
            
            # Calculate metrics at k
            ranked_k = ranked_items[:k]
            hits_k = sum(1 for it in ranked_k if it in relevant_sample_items)
            
            # Precision@k
            precision = hits_k / k if k > 0 else 0.0
            precision_scores.append(precision)
            
            # Recall@k
            recall = hits_k / len(relevant_sample_items) if relevant_sample_items else 0.0
            recall_scores.append(recall)
            
            # NDCG@k
            rels = [test_r_user[it] if it in relevant_sample_items else 0.0 for it in ranked_k]
            ideal_rels = sorted([test_r_user[it] for it in relevant_sample_items], reverse=True)
            ideal_rels = ideal_rels[:len(ranked_k)]
            if len(ideal_rels) < len(ranked_k):
                ideal_rels.extend([0.0] * (len(ranked_k) - len(ideal_rels)))
            dcg = sum((2**r - 1) / np.log2(2 + i) for i, r in enumerate(rels))
            idcg = sum((2**r - 1) / np.log2(2 + i) for i, r in enumerate(ideal_rels))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        # Store results for this k value
        results[f'precision_at_{k}'] = {
            'mean': np.mean(precision_scores) if precision_scores else 0.0,
            'scores': precision_scores
        }
        results[f'recall_at_{k}'] = {
            'mean': np.mean(recall_scores) if recall_scores else 0.0,
            'scores': recall_scores
        }
        results[f'ndcg_at_{k}'] = {
            'mean': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'scores': ndcg_scores
        }
    
    # Format results in the expected format
    formatted_results = {
        'precision': {},
        'recall': {},
        'ndcg': {}
    }
    
    for k in k_values:
        formatted_results['precision'][f'at_{k}'] = results[f'precision_at_{k}']['mean']
        formatted_results['precision'][f'scores_{k}'] = results[f'precision_at_{k}']['scores']
        formatted_results['recall'][f'at_{k}'] = results[f'recall_at_{k}']['mean']
        formatted_results['recall'][f'scores_{k}'] = results[f'recall_at_{k}']['scores']
        formatted_results['ndcg'][f'at_{k}'] = results[f'ndcg_at_{k}']['mean']
        formatted_results['ndcg'][f'scores_{k}'] = results[f'ndcg_at_{k}']['scores']
    
    return formatted_results
