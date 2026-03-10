"""
Lazy candidate generation for memory-efficient evaluation.

Instead of creating a dict with candidates for all users (which uses
~22 GB for ML-20M), generate candidates on-the-fly per user.
"""

import numpy as np
from scipy import sparse


class LazyCandidateGenerator:
    """
    Generate candidates on-demand per user, never storing all candidates in memory.

    For ML-20M:
    - Old approach: 136,677 users × 20,700 candidates = 2.8B integers = 22 GB
    - New approach: Generate 1 user's candidates at a time = ~165 KB peak memory
    """

    def __init__(self, train_matrix, test_matrix, n_users, n_items,
                 eval_mode='full_ranking', num_negatives=100,
                 rating_threshold=4.0, random_state=42):
        """
        Initialize lazy candidate generator.

        Args:
            train_matrix: Sparse training matrix (n_items, n_users)
            test_matrix: Sparse test matrix (n_items, n_users)
            n_users: Number of users
            n_items: Number of items
            eval_mode: 'full_ranking', 'negative_sampling', or 'rerank'
            num_negatives: Number of negatives for negative_sampling mode
            rating_threshold: Minimum rating to consider positive
            random_state: Random seed for negative sampling
        """
        self.train_matrix = train_matrix.tocsc()  # CSC for fast user slicing
        self.test_matrix = test_matrix.tocsc() if test_matrix is not None else None
        self.n_users = n_users
        self.n_items = n_items
        self.eval_mode = eval_mode
        self.num_negatives = num_negatives
        self.rating_threshold = rating_threshold
        self.random_state = random_state

        # Filter by rating threshold once
        if sparse.issparse(train_matrix):
            train_filtered = train_matrix.copy()
            train_filtered.data[train_filtered.data < rating_threshold] = 0
            train_filtered.eliminate_zeros()
            self.train_filtered = train_filtered.tocsc()
        else:
            train_filtered = train_matrix.copy()
            train_filtered[train_filtered < rating_threshold] = 0
            self.train_filtered = train_filtered

        # Create all_items array once
        self.all_items = np.arange(n_items, dtype=np.int32)

        # For negative sampling, set random seed
        if eval_mode == 'negative_sampling':
            np.random.seed(random_state)

    def get_candidates(self, user_id):
        """
        Generate candidate items for a single user on-demand.

        Args:
            user_id: User ID

        Returns:
            np.ndarray: Array of candidate item IDs
        """
        # Get positive training items for this user
        training_items = self.train_filtered[:, user_id].nonzero()[0]

        if self.eval_mode == 'rerank':
            # Re-ranking: only test items
            if self.test_matrix is not None:
                candidates = self.test_matrix[:, user_id].nonzero()[0]
            else:
                candidates = np.array([], dtype=np.int32)

        elif self.eval_mode == 'full_ranking':
            # Full ranking: all items - training items
            mask = np.ones(self.n_items, dtype=bool)
            mask[training_items] = False
            candidates = self.all_items[mask]

        else:  # negative_sampling
            # Test items + N random negatives
            if self.test_matrix is not None:
                test_items = self.test_matrix[:, user_id].nonzero()[0]
            else:
                test_items = np.array([], dtype=np.int32)

            # Get non-training items for negative sampling
            mask = np.ones(self.n_items, dtype=bool)
            mask[training_items] = False
            mask[test_items] = False  # Also exclude test items from negatives

            available_negatives = self.all_items[mask]

            if len(available_negatives) > self.num_negatives:
                negatives = np.random.choice(available_negatives, self.num_negatives, replace=False)
            else:
                negatives = available_negatives

            # Combine test items + negatives
            candidates = np.concatenate([test_items, negatives])

        return candidates

    def keys(self):
        """Return iterator over all user IDs (for compatibility with dict interface)."""
        return range(self.n_users)

    def values(self):
        """
        Generator that yields candidates for all users.

        WARNING: This will iterate through ALL users. Only use for computing
        statistics like average candidates. For evaluation, use get_candidates()
        or __getitem__() per user.
        """
        for user_id in range(self.n_users):
            yield self.get_candidates(user_id).tolist()

    def items(self):
        """
        Generator that yields (user_id, candidates) tuples.

        For compatibility with dict interface when code iterates like:
            for user_id, candidates in samples_products.items():
        """
        for user_id in range(self.n_users):
            yield (user_id, self.get_candidates(user_id).tolist())

    def __getitem__(self, user_id):
        """
        Dict-like access: candidates[user_id]

        For compatibility with existing metric code that expects dict.
        """
        return self.get_candidates(user_id).tolist()

    def __contains__(self, user_id):
        """Check if user_id is valid."""
        return 0 <= user_id < self.n_users

    def __len__(self):
        """Return number of users."""
        return self.n_users
