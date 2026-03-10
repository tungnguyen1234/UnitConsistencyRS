"""
Utility modules for UC benchmarking.

This package contains data loading, preprocessing, metrics, and experiment utilities.

Note: For specific data loaders, import directly from submodules:
    from utils.experiment_utils import load_data_ML1M_exp_9
    from utils.dataloader import load_data_100k
"""

# Preprocessing (most commonly used)
from .preprocessing import (
    standard_train_test_split,
    create_full_ranking_candidates,
    create_negative_sampling_candidates,
    get_memory_usage,
    filter_by_rating_threshold,
    filter_and_clean_matrix
)

# Lazy data structures (for memory efficiency)
from .lazy_candidates import LazyCandidateGenerator

# Metrics (ranking and UC-specific)
from .metric import (
    calculate_ranking_metrics,
    calculate_ranking_metrics_from_predictions_dict,
    calculate_global_kendall_tau,
    calculate_macro_stats,
    calculate_bootstrap_stats,
    norm_kendall_tau,
    calculate_scores,
    calculate_scores_UCTC
)

# Experiment utilities - data loaders for standard evaluation
from .experiment_utils import (
    load_data_ML100K_exp_9,
    load_data_ML1M_exp_9,
    load_data_ML20M_exp_9,
    load_data_monti_exp_9
)

__all__ = [
    # Preprocessing
    'standard_train_test_split',
    'create_full_ranking_candidates',
    'create_negative_sampling_candidates',
    'get_memory_usage',
    'filter_by_rating_threshold',
    'filter_and_clean_matrix',

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
    'calculate_scores_UCTC',

    # Experiment utilities - data loaders
    'load_data_ML100K_exp_9',
    'load_data_ML1M_exp_9',
    'load_data_ML20M_exp_9',
    'load_data_monti_exp_9',
]
