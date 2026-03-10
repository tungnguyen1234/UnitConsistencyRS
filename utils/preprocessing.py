
import numpy as np
from time import time
import random
from scipy import sparse
import pandas as pd
import sys

# Optional: psutil for memory tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_memory_usage():
    """Get current process memory usage in MB."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert to MB
    return None


##### Rating Threshold Filtering

def filter_by_rating_threshold(train_r, threshold=4.0):
    """
    Filter matrix to only include ratings >= threshold.
    Ratings below threshold are set to 0.
    
    :param train_r: 2D numpy array or sparse matrix (items x users)
    :param threshold: Minimum rating to keep (default 4.0)
    :return: Filtered matrix with same shape, ratings below threshold zeroed out
    """
    if sparse.issparse(train_r):
        train_r_filtered = train_r.copy()
        train_r_filtered.data[train_r_filtered.data < threshold] = 0
        train_r_filtered.eliminate_zeros()
        return train_r_filtered
    else:
        train_r_filtered = np.copy(train_r)
        train_r_filtered[train_r_filtered < threshold] = 0
        return train_r_filtered


def filter_and_clean_matrix(train_r, threshold=4.0, min_user_interactions=1, min_item_interactions=1):
    """
    Filter matrix by rating threshold and remove users/items with too few interactions.
    
    :param train_r: 2D numpy array or sparse matrix (items x users)
    :param threshold: Minimum rating to keep
    :param min_user_interactions: Minimum interactions per user after filtering
    :param min_item_interactions: Minimum interactions per item after filtering
    :return: Filtered matrix, valid_item_indices, valid_user_indices
    """
    # Apply rating threshold
    train_r_filtered = filter_by_rating_threshold(train_r, threshold)
    
    if sparse.issparse(train_r_filtered):
        # Count interactions per user (columns) and item (rows)
        user_counts = np.array(train_r_filtered.getnnz(axis=0)).flatten()
        item_counts = np.array(train_r_filtered.getnnz(axis=1)).flatten()
    else:
        user_counts = np.count_nonzero(train_r_filtered, axis=0)
        item_counts = np.count_nonzero(train_r_filtered, axis=1)
    
    # Find valid users and items
    valid_users = user_counts >= min_user_interactions
    valid_items = item_counts >= min_item_interactions
    
    # Filter matrix
    if sparse.issparse(train_r_filtered):
        train_r_clean = train_r_filtered[valid_items, :][:, valid_users]
    else:
        train_r_clean = train_r_filtered[np.ix_(valid_items, valid_users)]
    
    valid_item_indices = np.where(valid_items)[0]
    valid_user_indices = np.where(valid_users)[0]
    
    return train_r_clean, valid_item_indices, valid_user_indices

##### Standard Random Train/Test Split for Ranking Evaluation

def standard_train_test_split(ratings, test_ratio=0.2, min_ratings_per_user=5,
                               rating_threshold=4.0, random_state=42):
    """
    Create a standard random train/test split for ranking evaluation.

    This function addresses reviewer concerns by providing a conventional evaluation
    setup where ALL methods (UC, SASRec, LightGCN, etc.) are evaluated on
    Precision@k, Recall@k, and NDCG@k metrics.

    IMPORTANT: Only ratings >= rating_threshold are considered as positive interactions.
    This ensures consistency between what UC learns and what evaluation metrics expect.

    Args:
        ratings (pd.DataFrame): DataFrame with columns [UserID, MovieID, Rating, ...]
        test_ratio (float): Proportion of ratings to use for testing (default: 0.2)
        min_ratings_per_user (int): Minimum ratings per user to be included (default: 5)
        rating_threshold (float): Threshold for positive feedback (default: 4.0)
        random_state (int): Random seed for reproducibility

    Returns:
        dict: Contains train_matrix, test_matrix, train_df, test_df,
              user_mapping, item_mapping, samples_products
    """
    np.random.seed(random_state)
    tic = time()

    print('Creating standard train/test split for ranking evaluation...')
    print(f'Rating threshold: >= {rating_threshold} (only positive interactions)')

    # STEP 1: Filter to only positive ratings (>= threshold)
    ratings_positive = ratings[ratings['Rating'] >= rating_threshold].copy()
    print(f'Ratings after threshold filter: {len(ratings_positive)} / {len(ratings)} '
          f'({100*len(ratings_positive)/len(ratings):.1f}%)')

    # STEP 2: Filter users with at least min_ratings_per_user positive ratings
    user_counts = ratings_positive.groupby('UserID').size()
    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    ratings_filtered = ratings_positive[ratings_positive['UserID'].isin(valid_users)].copy()

    print(f'Users after filtering: {len(valid_users)} (min {min_ratings_per_user} positive ratings)')

    # STEP 3: Filter items that appear at least once
    item_counts = ratings_filtered.groupby('MovieID').size()
    valid_items = item_counts[item_counts >= 1].index
    ratings_filtered = ratings_filtered[ratings_filtered['MovieID'].isin(valid_items)].copy()

    print(f'Items after filtering: {len(valid_items)}')

    # Split per user to ensure all users appear in both train and test
    train_list = []
    test_list = []

    for user_id, user_ratings in ratings_filtered.groupby('UserID'):
        n_ratings = len(user_ratings)
        n_test = max(1, int(n_ratings * test_ratio))  # At least 1 test rating

        # Shuffle user ratings
        user_ratings_shuffled = user_ratings.sample(frac=1, random_state=random_state + user_id)

        test_list.append(user_ratings_shuffled.iloc[:n_test])
        train_list.append(user_ratings_shuffled.iloc[n_test:])

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    print(f'Train ratings: {len(train_df)}, Test ratings: {len(test_df)}')

    # Create user and item mappings
    all_users = sorted(ratings_filtered['UserID'].unique())
    all_items = sorted(ratings_filtered['MovieID'].unique())

    user_to_idx = {user_id: idx for idx, user_id in enumerate(all_users)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(all_items)}

    n_users = len(all_users)
    n_items = len(all_items)

    print(f'Total users: {n_users}, Total items: {n_items}')

    # Create sparse matrices (n_items, n_users) to match existing code convention
    train_df['User_Index'] = train_df['UserID'].map(user_to_idx)
    train_df['Item_Index'] = train_df['MovieID'].map(item_to_idx)
    test_df['User_Index'] = test_df['UserID'].map(user_to_idx)
    test_df['Item_Index'] = test_df['MovieID'].map(item_to_idx)

    # Build sparse matrices: (n_items, n_users)
    train_matrix = sparse.csr_matrix(
        (train_df['Rating'].values, (train_df['Item_Index'].values, train_df['User_Index'].values)),
        shape=(n_items, n_users),
        dtype=np.float32
    )

    test_matrix = sparse.csr_matrix(
        (test_df['Rating'].values, (test_df['Item_Index'].values, test_df['User_Index'].values)),
        shape=(n_items, n_users),
        dtype=np.float32
    )

    # Create samples_products for evaluation (test items per user)
    samples_products = {}
    for user_id in range(n_users):
        test_items = test_matrix[:, user_id].nonzero()[0]
        if len(test_items) > 0:
            samples_products[user_id] = test_items.tolist()

    print(f'Users with test items: {len(samples_products)}')
    print(f'Split created in {time() - tic:.2f} seconds\n')

    return {
        'train_matrix': train_matrix,
        'test_matrix': test_matrix,
        'train_df': train_df,
        'test_df': test_df,
        'user_mapping': user_to_idx,
        'item_mapping': item_to_idx,
        'samples_products': samples_products,
        'n_users': n_users,
        'n_items': n_items,
        'rating_threshold': rating_threshold
    }


def create_full_ranking_candidates(train_matrix, n_users, n_items, negative_samples=None,
                                    rating_threshold=4.0):
    """
    Create candidate pools for full ranking evaluation (standard RecSys protocol).
    
    For each user, the candidate pool consists of ALL items excluding their training items.
    This is the standard evaluation protocol used in RecSys papers (LightGCN, NCF, BPR-MF, etc.)
    
    IMPORTANT: Only items with rating >= threshold in training are excluded as "seen" items.
    
    Args:
        train_matrix (sparse matrix): Training data (n_items, n_users)
        n_users (int): Number of users
        n_items (int): Number of items
        negative_samples (int, optional): If specified, sample this many negative items
                                          instead of using all items (for efficiency)
        rating_threshold (float): Threshold for positive feedback (default: 4.0)
    
    Returns:
        dict: {user_id: [item_ids]} where item_ids are all items except positive training items
    """
    tic = time()
    mem_start = get_memory_usage()

    print(f'Creating full ranking candidates (negative_samples={negative_samples}, threshold={rating_threshold})...')
    print(f'Dataset: {n_users} users, {n_items} items')
    if mem_start:
        print(f'Memory usage at start: {mem_start:.1f} MB')

    # Apply rating threshold filter to training matrix
    print(f'Filtering training matrix by threshold={rating_threshold}...')
    train_filtered = filter_by_rating_threshold(train_matrix, threshold=rating_threshold)
    print(f'Training matrix filtered: {train_filtered.nnz} non-zero entries')

    full_ranking_candidates = {}

    # Convert to CSC for efficient column operations
    print('Converting to CSC format for efficient user slicing...')
    train_csc = train_filtered.tocsc()
    print('CSC conversion complete')

    # Create all_items set ONCE (not in the loop!)
    all_items_array = np.arange(n_items, dtype=np.int32)
    print(f'Created item array: {len(all_items_array)} items')

    print(f'Processing {n_users} users...')
    for user_id in range(n_users):
        if user_id > 0 and user_id % 10000 == 0:
            elapsed = time() - tic
            rate = user_id / elapsed
            remaining = (n_users - user_id) / rate
            mem_now = get_memory_usage()
            mem_msg = f', memory: {mem_now:.1f} MB' if mem_now else ''
            print(f'  Processed {user_id}/{n_users} users ({user_id/n_users*100:.1f}%) - {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining{mem_msg}')

        # Get positive training items for this user (items with rating >= threshold)
        training_items = train_csc[:, user_id].nonzero()[0]

        # Candidate pool = all items - positive training items
        if negative_samples is None:
            # Full ranking: use all non-training items (use numpy for memory efficiency)
            # Create boolean mask instead of set operations
            mask = np.ones(n_items, dtype=bool)
            mask[training_items] = False
            candidates = all_items_array[mask].tolist()
        else:
            # Negative sampling: sample N negative items
            mask = np.ones(n_items, dtype=bool)
            mask[training_items] = False
            non_training = all_items_array[mask]

            # Sample negative_samples items randomly
            if len(non_training) > negative_samples:
                candidates = np.random.choice(non_training, negative_samples, replace=False).tolist()
            else:
                candidates = non_training.tolist()

        if len(candidates) > 0:
            full_ranking_candidates[user_id] = candidates

    elapsed = time() - tic
    total_candidates = sum(len(v) for v in full_ranking_candidates.values())
    avg_candidates = total_candidates / len(full_ranking_candidates) if full_ranking_candidates else 0

    # Estimate memory usage (rough estimate: 8 bytes per int in Python list)
    memory_mb = (total_candidates * 8) / (1024 * 1024)
    mem_end = get_memory_usage()
    mem_increase = (mem_end - mem_start) if (mem_end and mem_start) else None

    print(f'✓ Full ranking candidates created in {elapsed:.2f} seconds')
    print(f'  Users with candidates: {len(full_ranking_candidates)}')
    print(f'  Total candidate entries: {total_candidates:,}')
    print(f'  Average candidates per user: {avg_candidates:.1f}')
    print(f'  Estimated dict memory: {memory_mb:.1f} MB')
    if mem_end:
        print(f'  Actual process memory: {mem_end:.1f} MB (increased by {mem_increase:.1f} MB)' if mem_increase else f'  Actual process memory: {mem_end:.1f} MB')

    return full_ranking_candidates


def create_negative_sampling_candidates(train_matrix, test_matrix, n_users, n_items, 
                                        num_negatives=100, rating_threshold=4.0, 
                                        random_state=42):
    """
    Create candidate pools using negative sampling (test items + random negatives).
    
    This is a common compromise between full ranking and re-ranking:
    - Candidate pool = test items + N randomly sampled negative items
    - More efficient than full ranking
    - Still challenging (must find positives among negatives)
    
    IMPORTANT: Only items with rating >= threshold are considered positive.
    Negative items are sampled from items NOT in positive training or test sets.
    
    Args:
        train_matrix (sparse matrix): Training data (n_items, n_users)
        test_matrix (sparse matrix): Test data (n_items, n_users)
        n_users (int): Number of users
        n_items (int): Number of items
        num_negatives (int): Number of negative items to sample per user
        rating_threshold (float): Threshold for positive feedback (default: 4.0)
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: {user_id: [item_ids]} where item_ids = test_items + sampled_negatives
    """
    random.seed(random_state)
    np.random.seed(random_state)
    
    tic = time()
    print(f'Creating negative sampling candidates ({num_negatives} negatives, threshold={rating_threshold})...')
    
    # Apply rating threshold filter to both matrices
    train_filtered = filter_by_rating_threshold(train_matrix, threshold=rating_threshold)
    test_filtered = filter_by_rating_threshold(test_matrix, threshold=rating_threshold)
    
    negative_sampling_candidates = {}
    
    # Convert to CSC for efficient column operations
    train_csc = train_filtered.tocsc()
    test_csc = test_filtered.tocsc()
    
    for user_id in range(n_users):
        # Get positive test items (items with rating >= threshold)
        test_items = test_csc[:, user_id].nonzero()[0].tolist()
        
        if len(test_items) == 0:
            continue
        
        # Get positive training items (to exclude from negatives)
        training_items = set(train_csc[:, user_id].nonzero()[0].tolist())
        
        # Sample negative items (items not in positive training or test)
        all_items = set(range(n_items))
        negative_pool = list(all_items - training_items - set(test_items))
        
        if len(negative_pool) >= num_negatives:
            sampled_negatives = random.sample(negative_pool, num_negatives)
        else:
            sampled_negatives = negative_pool
        
        # Combine positive test items and sampled negatives
        candidates = test_items + sampled_negatives
        negative_sampling_candidates[user_id] = candidates
    
    print(f'Negative sampling candidates created in {time() - tic:.2f} seconds')
    print(f'Average candidates per user: {sum(len(v) for v in negative_sampling_candidates.values()) / len(negative_sampling_candidates):.1f}')
    
    return negative_sampling_candidates
