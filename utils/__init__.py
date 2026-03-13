"""
Utility modules for Unit Consistency (UC) recommender systems.

Submodules
----------
io              : Logger, save_data, load_json
preprocessing   : train/test splitting, rating filtering, exp-8 sampling helpers
metric          : ranking metrics (P@k, R@k, NDCG@k), Kendall-tau
lazy_candidates : memory-efficient candidate generation
dataloader      : dense matrix data loaders (exp-8 format)
experiment_utils: UC model runners + data loaders (exp-9 / standard ranking format)
ranking_eval    : UC-only standard ranking evaluation pipeline
"""

# I/O helpers
from .io import Logger, save_data, load_json

# Preprocessing
from .preprocessing import (
    standard_train_test_split,
    create_full_ranking_candidates,
    create_negative_sampling_candidates,
    get_memory_usage,
    filter_by_rating_threshold,
    filter_and_clean_matrix,
    # Experiment-8 helpers
    sample_products_all_users_by_indices,
    sample_products_all_users_by_indices_sparse,
    get_users_from_indices,
    get_users_from_indices_sparse,
)

# Lazy candidate generator
from .lazy_candidates import LazyCandidateGenerator

# Metrics
from .metric import (
    calculate_ranking_metrics,
    calculate_ranking_metrics_from_predictions_dict,
    calculate_global_kendall_tau,
    calculate_macro_stats,
    calculate_bootstrap_stats,
    norm_kendall_tau,
    calculate_scores,
    calculate_scores_UC,
)

# Experiment utilities — data loaders & UC runners
from .experiment_utils import (
    # Exp-9 / standard ranking data loaders
    load_data_ML100K_exp_9,
    load_data_ML1M_exp_9,
    load_data_ML20M_exp_9,
    load_data_monti_exp_9,
    load_data_Netflix_exp_9,
    # Experiment-9 preprocessing
    preprocess_data,
    train_test_split,
    # UC model runners
    run_UC_easy,
    run_UC_hard,
)

__all__ = [
    # I/O
    'Logger', 'save_data', 'load_json',
    # Preprocessing
    'standard_train_test_split',
    'create_full_ranking_candidates',
    'create_negative_sampling_candidates',
    'get_memory_usage',
    'filter_by_rating_threshold',
    'filter_and_clean_matrix',
    'sample_products_all_users_by_indices',
    'sample_products_all_users_by_indices_sparse',
    'get_users_from_indices',
    'get_users_from_indices_sparse',
    # Lazy structures
    'LazyCandidateGenerator',
    # Metrics
    'calculate_ranking_metrics',
    'calculate_ranking_metrics_from_predictions_dict',
    'calculate_global_kendall_tau',
    'calculate_macro_stats',
    'calculate_bootstrap_stats',
    'norm_kendall_tau',
    'calculate_scores',
    'calculate_scores_UC',
    # Data loaders
    'load_data_ML100K_exp_9',
    'load_data_ML1M_exp_9',
    'load_data_ML20M_exp_9',
    'load_data_monti_exp_9',
    'load_data_Netflix_exp_9',
    # Experiment-9 preprocessing
    'preprocess_data',
    'train_test_split',
    # UC runners
    'run_UC_easy',
    'run_UC_hard',
]
