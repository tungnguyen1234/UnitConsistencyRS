
import numpy as np
from time import time
import random
import h5py
import os
from scipy.sparse import csc_matrix
from scipy import sparse
import pandas as pd

def load_big_data_Netflix(path='./', seed=1234, files=[2, 4]):
    tic = time()
    
    # Load and process data
    dfs = []
    movie_ids = []
    
    for i in files:
        path_file = os.path.join(path, f'combined_data_{i}.txt')
        df = pd.read_csv(path_file, header = None, names = ['userId', 'rating'], usecols = [0,1])
        df['rating'] = df['rating'].astype(float)
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    data.index = np.arange(0,len(data))

    del dfs  # Free up memory

    data_nan = pd.DataFrame(pd.isnull(data.rating))
    data_nan = data_nan[data_nan['rating'] == True]
    data_nan = data_nan.reset_index()

    movie_np = []

    # Calculate the differences between consecutive indices
    diff = np.diff(data_nan['index'])

    # Create an array of movie IDs
    movie_ids = np.arange(1, len(diff) + 1)

    # Create the repeated movie IDs
    movie_np = np.repeat(movie_ids, diff - 1)

    # Account for the last record
    last_record = np.full(len(data) - data_nan.iloc[-1, 0] - 1, len(diff) + 1)
    movie_np = np.concatenate([movie_np, last_record])

    data = data[pd.notnull(data['rating'])]

    data['movieId'] = movie_np.astype(int)
    data['userId'] = data['userId'].astype(int)
    
    print(f'Data loaded in {time() - tic:.2f} seconds')
    
    # Create ratings matrix
    n_u = data['userId'].nunique()
    n_m = data['movieId'].nunique()
    
    # Create dictionaries for user and movie IDs
    udict = {id: i for i, id in enumerate(data['userId'].unique())}
    mdict = {id: i for i, id in enumerate(data['movieId'].unique())}
    
    np.random.seed(seed)
    data = data.sample(frac=1).reset_index(drop=True)

    # Convert user and movie IDs to their corresponding indices
    user_indices = data['userId'].map(udict).values
    movie_indices = data['movieId'].map(mdict).values

    # Create the ratings sparse matrix
    train_r = sparse.csr_matrix((data['rating'].values, (movie_indices, user_indices)), 
                                shape=(n_m, n_u), dtype='float32')

    # Create mask for non-zero entries (this will also be a sparse matrix)
    train_m = train_r.copy()
    train_m.data = np.ones_like(train_m.data, dtype='float32')
    
    return train_r

def load_big_data_ML_20M(path='./', seed=1234):
    tic = time()
    
    # MovieLens 20M uses comma as delimiter and has a header
    data = pd.read_csv(os.path.join(path, 'ratings.csv'), usecols=['userId', 'movieId', 'rating'], header=0)
    
    print(f'Data loaded in {time() - tic:.2f} seconds')

    n_u = data['userId'].nunique()  # num of users
    n_m = data['movieId'].nunique()  # num of movies
    n_r = len(data)  # num of ratings

    # Create dictionaries for user and movie IDs
    udict = {id: i for i, id in enumerate(data['userId'].unique())}
    mdict = {id: i for i, id in enumerate(data['movieId'].unique())}

    # Shuffle the data
    np.random.seed(seed)
    data = data.sample(frac=1).reset_index(drop=True)

    # Convert user and movie IDs to their corresponding indices
    user_indices = data['userId'].map(udict).values
    movie_indices = data['movieId'].map(mdict).values

    # Create the ratings sparse matrix
    train_r = sparse.csr_matrix((data['rating'].values, (movie_indices, user_indices)), 
                                shape=(n_m, n_u), dtype='float32')

    # Create mask for non-zero entries (this will also be a sparse matrix)
    train_m = train_r.copy()
    train_m.data = np.ones_like(train_m.data, dtype='float32')

    print('Data matrix loaded')
    print(f'Number of users: {n_u}')
    print(f'Number of movies: {n_m}')
    print(f'Number of ratings: {n_r}')
    return train_r

def load_data_100k(path='./', delimiter='\t'):

    train = np.loadtxt(os.path.join(path, 'movielens_100k_u1.base'), skiprows=0, delimiter=delimiter).astype('int32')
    test = np.loadtxt(os.path.join(path, 'movielens_100k_u1.test'), skiprows=0, delimiter=delimiter).astype('int32')
    total = np.concatenate((train, test), axis=0)

    n_u = np.unique(total[:,0]).size  # num of users
    n_m = np.unique(total[:,1]).size  # num of movies
    n_train = train.shape[0]  # num of training ratings
    n_test = test.shape[0]  # num of test ratings

    train_r = np.zeros((n_m, n_u), dtype='float32')


    for i in range(n_train):
        train_r[train[i,1]-1, train[i,0]-1] = train[i,2]

    for i in range(n_test):
        train_r[test[i,1]-1, test[i,0]-1] = test[i,2]

    train_m = np.greater(train_r, 1e-12).astype('float32')  # masks indicating non-zero entries

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_train))
    return train_r


def load_data_100k_train_test(path='./', delimiter='\t'):

    train = np.loadtxt(os.path.join(path, 'movielens_100k_u1.base'), skiprows=0, delimiter=delimiter).astype('int32')
    test = np.loadtxt(os.path.join(path, 'movielens_100k_u1.test'), skiprows=0, delimiter=delimiter).astype('int32')
    total = np.concatenate((train, test), axis=0)

    n_u = np.unique(total[:,0]).size  # num of users
    n_m = np.unique(total[:,1]).size  # num of movies
    n_train = train.shape[0]  # num of training ratings
    n_test = test.shape[0]  # num of test ratings

    train_r = np.zeros((n_m, n_u), dtype='float32')
    test_r = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_train):
        train_r[train[i,1]-1, train[i,0]-1] = train[i,2]

    for i in range(n_test):
        test_r[test[i,1]-1, test[i,0]-1] = test[i,2]

    train_m = np.greater(train_r, 1e-12).astype('float32')  # masks indicating non-zero entries
    test_m = np.greater(test_r, 1e-12).astype('float32')

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_train))
    print('num of test ratings: {}'.format(n_test))

    return n_m, n_u, train_r, train_m, test_r, test_m



def load_data_1m(path='./', delimiter='::', seed=1234):
    tic = time()
    print('reading data...')
    data = np.genfromtxt(os.path.join(path, 'movielens_1m_dataset.dat'), delimiter=delimiter)
    print('taken', time() - tic, 'seconds')

    n_u = np.unique(data[:,0]).size  # num of users
    n_m = np.unique(data[:,1]).size  # num of movies
    n_r = data.shape[0]  # num of ratings

    udict = {}
    for i, u in enumerate(np.unique(data[:,0]).tolist()):
        udict[u] = i
    mdict = {}
    for i, m in enumerate(np.unique(data[:,1]).tolist()):
        mdict[m] = i

    np.random.seed(seed)
    idx = np.arange(n_r)
    np.random.shuffle(idx)

    train_r = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_r):
        u_id = data[idx[i], 0]
        m_id = data[idx[i], 1]
        r = data[idx[i], 2]
        train_r[mdict[m_id], udict[u_id]] = r

    train_m = np.greater(train_r, 1e-12).astype('float32')  # masks indicating non-zero entries

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))

    return train_r

# ------------------------------
def load_matlab_file(path_file, name_field):
    
    db = h5py.File(path_file, 'r')
    ds = db[name_field]

    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir   = np.asarray(ds['ir'])
            jc   = np.asarray(ds['jc'])
            out  = csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out

def load_data_monti(path='./'):
    # Load the 'M' matrix and 'Otraining' matrix from the MATLAB file
    M = load_matlab_file(os.path.join(path, 'douban_monti_dataset.mat'), 'M')

    # Calculate the number of users and movies from the 'M' matrix
    n_u = M.shape[0]  # num of users
    n_m = M.shape[1]  # num of movies

    # Count the number of training ratings
    n_train = M[np.where(M)].size

    # Transpose the training data to get the desired format (movies x users)
    train_r = M.T

    # Create a mask for non-zero entries in the training ratings matrix
    train_m = np.greater(train_r, 1e-12).astype('float32')

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_train))

    return train_r