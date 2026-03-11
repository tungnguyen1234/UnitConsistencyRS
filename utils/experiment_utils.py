"""
Experiment utilities: data loaders, preprocessing, and UC model runners.
"""
import numpy as np
import os
import sys
import gc
from time import time
from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix
import pandas as pd
import h5py

from .dataloader import *
from .metric import (
    calculate_macro_stats,
    calculate_bootstrap_stats,
    calculate_scores_UC,
    calculate_scores,
    calculate_scores_vectorized,
)
from .preprocessing import *

# ---------------------------------------------------------------------------
# Algorithm imports — UC_sparse.py lives at the repository root
# ---------------------------------------------------------------------------
_repo_root = os.path.dirname(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import torch as t
from UC_sparse import SparseUC
from rankingSVD_algo import rankingSVD


# ---------------------------------------------------------------------------
# UC Model Runners
# ---------------------------------------------------------------------------

def run_UC_hard(n_u, r_train, test_r, samples_products, device, epsilon=1e-9):
    if sparse.issparse(r_train):
        coo = r_train.tocoo().T
        i = torch.LongTensor(np.vstack((coo.row, coo.col)))
        v = torch.FloatTensor(coo.data)
        r_train = torch.sparse_coo_tensor(i, v, torch.Size(coo.shape)).to(device)
    elif isinstance(r_train, np.ndarray):
        coo = sparse.coo_matrix(r_train.T)
        i = torch.LongTensor(np.vstack((coo.row, coo.col)))
        v = torch.FloatTensor(coo.data)
        r_train = torch.sparse_coo_tensor(i, v, torch.Size(coo.shape)).to(device)

    UC = SparseUC(device, r_train, epsilon, mode_full=False)
    latent_1_UC, latent_2_UC = UC.UC()
    latent_1_UC = latent_1_UC.float().cpu().detach().numpy()
    latent_2_UC = latent_2_UC.float().cpu().detach().numpy()

    macro_scores = calculate_scores_vectorized(
        n_u, test_r.T, samples_products, method="UC",
        latent_1=latent_1_UC, latent_2=latent_2_UC
    )

    gc.collect()
    torch.cuda.empty_cache()
    return {
        'UC': {
            'normal_macro': calculate_macro_stats(macro_scores),
            'bootstrap_macro': calculate_bootstrap_stats(macro_scores)
        }
    }


def run_UC_easy(n_u, r_train, test_r, samples_products, device, epsilon=1e-9):
    tensor = t.t(r_train)
    UC = SparseUC(device, tensor, epsilon, mode_full=False)
    latent_1_UC, latent_2_UC = UC.UC()
    latent_1_UC = latent_1_UC.float().cpu().detach().numpy()
    latent_2_UC = latent_2_UC.float().cpu().detach().numpy()

    macro_scores = calculate_scores_UC(
        n_u, test_r.T, latent_1_UC, latent_2_UC, samples_products, method='UC'
    )
    return {
        'UC': {
            'normal_macro': calculate_macro_stats(macro_scores),
            'bootstrap_macro': calculate_bootstrap_stats(macro_scores),
        }
    }


def run_SVD_ranking_easy(n_u, n_m, r_train, test_r, percents, samples_products, n_bootstrap=1000):
    hash_SVD = {}
    svd = rankingSVD(n_u, n_m)
    tensor = t.t(r_train)

    for percent in percents:
        pre = svd.train(tensor, percent)
        pre = pre.float().cpu().detach().numpy()
        macro_scores = calculate_scores(n_u, test_r.T, samples_products, pre)
        hash_SVD[percent] = {
            'normal_macro': calculate_macro_stats(macro_scores),
            'bootstrap_macro': calculate_bootstrap_stats(macro_scores, n_bootstrap),
        }
    return hash_SVD


def run_SVD_ranking_hard(n_u, n_m, r_train, test_r, percents, samples_products, device, n_bootstrap=1000):
    if sparse.issparse(r_train):
        coo = r_train.tocoo().T
        i = torch.LongTensor(np.vstack((coo.row, coo.col)))
        v = torch.FloatTensor(coo.data)
        r_train = torch.sparse_coo_tensor(i, v, torch.Size(coo.shape)).to(device)
    elif isinstance(r_train, np.ndarray):
        coo = sparse.coo_matrix(r_train.T)
        i = torch.LongTensor(np.vstack((coo.row, coo.col)))
        v = torch.FloatTensor(coo.data)
        r_train = torch.sparse_coo_tensor(i, v, torch.Size(coo.shape)).to(device)

    svd = rankingSVD(n_u, n_m)
    hash_SVD = {}
    for percent in percents:
        tic = time()
        gc.collect()
        torch.cuda.empty_cache()
        Q_side_result = svd.train(r_train, percent)
        macro_scores = calculate_scores_vectorized(
            n_u, test_r.T, samples_products, method="rankSVD",
            r_train=r_train, Q_side_result=Q_side_result
        )
        hash_SVD[percent] = {
            'normal_macro': calculate_macro_stats(macro_scores),
            'bootstrap_macro': calculate_bootstrap_stats(macro_scores, n_bootstrap),
        }
        print(f'SVD for k={percent} done in {time() - tic:.2f}s')
    return hash_SVD


# ---------------------------------------------------------------------------
# Data Loaders (Experiment 9 / Standard Ranking format — returns DataFrame)
# ---------------------------------------------------------------------------

def load_data_ML1M_exp_9(path='./', delimiter='::', seed=1234):
    tic = time()
    print('reading data...')
    ratings = pd.read_csv(
        os.path.join(path, 'movielens_1m_dataset.dat'),
        sep=delimiter,
        engine='python',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp']
    )
    print(f'taken {time() - tic:.2f} seconds')
    return ratings


def load_data_monti_exp_9(path='./'):
    def _load_matlab_file(path_file, name_field):
        db = h5py.File(path_file, 'r')
        ds = db[name_field]
        try:
            if 'ir' in ds.keys():
                data = np.asarray(ds['data'])
                ir = np.asarray(ds['ir'])
                jc = np.asarray(ds['jc'])
                out = csc_matrix((data, ir, jc)).astype(np.float32)
        except AttributeError:
            out = np.asarray(ds).astype(np.float32).T
        db.close()
        return out

    tic = time()
    print('reading data...')
    M = _load_matlab_file(os.path.join(path, 'douban_monti_dataset.mat'), 'M')
    n_u, n_m = M.shape
    data_matrix = M.T
    users, movies = data_matrix.nonzero()
    rating_vals = data_matrix[users, movies]
    ratings = pd.DataFrame({'UserID': users, 'MovieID': movies, 'Rating': rating_vals})
    print(f'taken {time() - tic:.2f} seconds')
    return ratings


def load_data_ML100K_exp_9(path='./', delimiter='\t', seed=1234):
    tic = time()
    print('reading data...')
    train = pd.read_csv(
        os.path.join(path, 'movielens_100k_u1.base'),
        sep=delimiter, names=['UserID', 'MovieID', 'Rating', 'Timestamp']
    )
    test = pd.read_csv(
        os.path.join(path, 'movielens_100k_u1.test'),
        sep=delimiter, names=['UserID', 'MovieID', 'Rating', 'Timestamp']
    )
    ratings = pd.concat([train, test], axis=0)
    print(f'taken {time() - tic:.2f} seconds')
    return ratings


def load_data_Netflix_exp_9(path='./', files=[1, 2]):
    tic = time()
    print('reading data...')
    dfs = []
    for i in files:
        path_file = os.path.join(path, f'combined_data_{i}.txt')
        df = pd.read_csv(path_file, header=None, names=['UserID', 'Rating'], usecols=[0, 1])
        df['Rating'] = df['Rating'].astype(float)
        dfs.append(df)
    ratings = pd.concat(dfs, ignore_index=True)
    ratings.index = np.arange(0, len(ratings))
    del dfs

    ratings_nan = pd.DataFrame(pd.isnull(ratings.Rating))
    ratings_nan = ratings_nan[ratings_nan['Rating'] == True].reset_index()
    diff = np.diff(ratings_nan['index'])
    movie_ids = np.arange(1, len(diff) + 1)
    movie_np = np.repeat(movie_ids, diff - 1)
    last_record = np.full(len(ratings) - ratings_nan.iloc[-1, 0] - 1, len(diff) + 1)
    movie_np = np.concatenate([movie_np, last_record])
    ratings = ratings[pd.notnull(ratings['Rating'])]
    ratings['MovieID'] = movie_np.astype(int)
    ratings['UserID'] = ratings['UserID'].astype(int)
    print(f'taken {time() - tic:.2f} seconds')
    return ratings


def load_data_ML20M_exp_9(path='./'):
    tic = time()
    print('reading data...')
    ratings = pd.read_csv(
        os.path.join(path, 'ratings.csv'),
        usecols=['userId', 'movieId', 'rating', 'timestamp'], header=0
    )
    ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    print(f'taken {time() - tic:.2f} seconds')
    return ratings


# ---------------------------------------------------------------------------
# Experiment 9 Preprocessing
# ---------------------------------------------------------------------------

def preprocess_data(ratings):
    """Compute long-tail items and filter users with all 5 rating levels."""
    tic = time()
    print('preprocess data...')

    item_popularity = ratings.groupby('MovieID').size().reset_index(name='Count')
    item_popularity = item_popularity.sort_values(by='Count', ascending=False)
    item_popularity['Cumulative_Count'] = item_popularity['Count'].cumsum()
    total_ratings = item_popularity['Count'].sum()
    item_popularity['Cumulative_Percentage'] = (
        item_popularity['Cumulative_Count'] / total_ratings
    )
    long_tail_items = item_popularity[
        item_popularity['Cumulative_Percentage'] > 0.33
    ]['MovieID']

    user_rating_matrix = ratings.pivot_table(
        index='UserID', columns='Rating', aggfunc='size', fill_value=0
    )
    required_ratings = [1, 2, 3, 4, 5]
    qualified_users = user_rating_matrix.loc[
        (user_rating_matrix[required_ratings] > 0).all(axis=1)
    ].index

    ratings = ratings[ratings['UserID'].isin(qualified_users)]
    long_tail_ratings = ratings[ratings['MovieID'].isin(long_tail_items)]

    del item_popularity, qualified_users, ratings
    print(f'preprocess done in {time() - tic:.2f}s')
    return long_tail_ratings


def train_test_split(start_rating, long_tail_ratings, ratings, mode="easy"):
    """Create train/test split for experiment 9 (rank-preference consistency)."""
    user_rating_pivot = long_tail_ratings.pivot_table(
        index='UserID', columns='Rating', aggfunc='size', fill_value=0
    )
    qualified_users = user_rating_pivot[
        (user_rating_pivot.get(start_rating, 0) >= 1) &
        (user_rating_pivot.get(5, 0) >= 1)
    ].index

    test_candidates = long_tail_ratings[long_tail_ratings['UserID'].isin(qualified_users)]
    test_candidates_shuffled = test_candidates.sample(frac=1, random_state=42)

    ratings_i = test_candidates_shuffled[test_candidates_shuffled['Rating'] == start_rating]
    ratings_5 = test_candidates_shuffled[test_candidates_shuffled['Rating'] == 5]

    ratings_i_sampled = ratings_i.groupby('UserID').sample(n=1).reset_index(drop=True)
    ratings_5_sampled = ratings_5.groupby('UserID').sample(n=1).reset_index(drop=True)
    test_ratings = pd.concat([ratings_i_sampled, ratings_5_sampled])

    test_ratings['UserID_MovieID'] = (
        test_ratings['UserID'].astype(str) + '_' + test_ratings['MovieID'].astype(str)
    )
    ratings.loc[:, 'UserID_MovieID'] = (
        ratings['UserID'].astype(str) + '_' + ratings['MovieID'].astype(str)
    )
    training_ratings = ratings[~ratings['UserID_MovieID'].isin(test_ratings['UserID_MovieID'])]

    training_user_counts = training_ratings.groupby('UserID').size()
    users_with_ratings = training_user_counts[training_user_counts > 0].index
    training_ratings = training_ratings[training_ratings['UserID'].isin(users_with_ratings)]
    test_ratings = test_ratings[test_ratings['UserID'].isin(users_with_ratings)]

    user_indices = {
        uid: idx for idx, uid in enumerate(sorted(training_ratings['UserID'].unique()))
    }
    movie_indices = {
        mid: idx for idx, mid in enumerate(sorted(training_ratings['MovieID'].unique()))
    }

    training_ratings['User_Index'] = training_ratings['UserID'].map(user_indices)
    training_ratings['Movie_Index'] = training_ratings['MovieID'].map(movie_indices)

    test_ratings = test_ratings[test_ratings['MovieID'].isin(movie_indices.keys())]
    test_ratings['User_Index'] = test_ratings['UserID'].map(user_indices)
    test_ratings['Movie_Index'] = test_ratings['MovieID'].map(movie_indices)

    training_matrix = csr_matrix(
        (training_ratings['Rating'].astype(np.float32),
         (training_ratings['Movie_Index'], training_ratings['User_Index']))
    )
    test_matrix = csr_matrix(
        (test_ratings['Rating'].astype(np.float32),
         (test_ratings['Movie_Index'], test_ratings['User_Index'])),
        shape=training_matrix.shape
    )

    non_zero_per_col = np.diff(test_matrix.indptr)
    col_indices = np.repeat(np.arange(test_matrix.shape[0]), non_zero_per_col)
    row_indices = test_matrix.indices
    df = pd.DataFrame({'row_idx': row_indices, 'col_idx': col_indices})
    samples_products = df.groupby('row_idx')['col_idx'].apply(list).to_dict()
    del df

    print(f"Experiment for start_rating={start_rating}: "
          f"{training_matrix.shape[0]} items x {training_matrix.shape[1]} users, "
          f"train nnz={training_matrix.nnz}, test nnz={test_matrix.nnz}")

    del user_rating_pivot, qualified_users, test_candidates
    del ratings_i, ratings_5, training_user_counts, users_with_ratings
    del user_indices, movie_indices

    if mode == "easy":
        return (training_ratings, test_ratings,
                np.array(training_matrix.todense()),
                np.array(test_matrix.todense()),
                samples_products)
    else:  # "hard"
        return training_ratings, test_ratings, training_matrix, test_matrix, samples_products
