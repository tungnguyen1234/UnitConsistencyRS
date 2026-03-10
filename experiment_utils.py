# main_execution.py
import numpy as np
from dataloader import *
from metric import *
from scipy import sparse
import importlib.util
from preprocessing import *
from scipy.sparse import csc_matrix, csr_matrix

def print_memory_summary():
    """Print GPU memory summary."""
    print(torch.cuda.memory_summary(device=torch.device('cuda'), abbreviated=False))

def is_package_installed(package_name):
    spec = importlib.util.find_spec(package_name)
    return spec is not None

if is_package_installed("tensorflow"):
    print("TensorFlow is installed. Importing TensorFlow and modules from glocalk_new_test.")
    import tensorflow as tf
    from GLocalK import *
else:
    print("TensorFlow is not installed in this environment. Skipping TensorFlow-related imports.")

if is_package_installed("torch"):
    print("Torch is installed. Importing PyTorch and modules from UCTC_sparse and rankingSVD_algo.")
    import torch
    from UCTC_sparse import *
    from rankingSVD_algo import *
else:
    print("PyTorch is not installed in this environment. Skipping PyTorch-related imports.")

def run_UC_hard(n_u, r_train, test_r, samples_products, device, epsilon=1e-9):
    print_memory_summary()

    UC = SparseUC(device, r_train, epsilon, mode_full=False)
    latent_1_UC, latent_2_UC = UC.UC() 
    latent_1_UC, latent_2_UC = latent_1_UC.float().cpu().detach().numpy(), latent_2_UC.float().cpu().detach().numpy()

    print_memory_summary()
    
    macro_scores = calculate_scores_vectorized(n_u, test_r.T, samples_products, method="UC", latent_1=latent_1_UC, latent_2=latent_2_UC)

    # Parallelize the processing
    return {
        'normal_macro': calculate_macro_stats(macro_scores),
        'bootstrap_macro': calculate_bootstrap_stats(macro_scores)
    }

def run_TC_hard(n_u, r_train, test_r, samples_products, device, epsilon=1e-9):
    TC = SparseTC(device, r_train, epsilon, mode_full=False)
    latent_1_TC, latent_2_TC = TC.TC() 
    latent_1_TC, latent_2_TC = latent_1_TC.float().cpu().detach().numpy(), latent_2_TC.float().cpu().detach().numpy()
    
    macro_scores = calculate_scores_UCTC(n_u, test_r.T, latent_1_TC, latent_2_TC, samples_products, method='TC')

    return {
        'normal_macro': calculate_macro_stats(macro_scores),
        'bootstrap_macro': calculate_bootstrap_stats(macro_scores)
    }


def run_UC_TC_hard(n_u, r_train, test_r, samples_products, device, epsilon=1e-9):
    if sparse.issparse(r_train):
        # Convert scipy sparse matrix to torch sparse tensor
        coo = r_train.tocoo().T  # Transpose here if needed
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        r_train = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)
    elif isinstance(r_train, np.ndarray):
        # Convert numpy array to torch sparse tensor
        coo = sparse.coo_matrix(r_train.T)
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        r_train = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)

    elif not torch.is_tensor(r_train) or not r_train.is_sparse:
        raise ValueError("r_train must be a scipy sparse matrix, numpy array, or torch sparse tensor")
    
        
    # Call this function after specific parts of your code
    
    UC_hard = run_UC_hard(n_u, r_train, test_r, samples_products, device, epsilon)
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'UC': UC_hard
    }


def run_UC_easy(n_u, r_train, test_r, samples_products, device, epsilon=1e-9):
    tensor = t.t(r_train)
    UC = SparseUC(device, tensor, epsilon, mode_full=False)
    latent_1_UC, latent_2_UC = UC.UC() 
    latent_1_UC, latent_2_UC = latent_1_UC.float().cpu().detach().numpy(), latent_2_UC.float().cpu().detach().numpy()
    
    macro_scores = calculate_scores_UCTC(n_u, test_r.T, latent_1_UC, latent_2_UC, samples_products, method='UC')

    return {
        'normal_macro': calculate_macro_stats(macro_scores),
        'bootstrap_macro': calculate_bootstrap_stats(macro_scores),
    }

def run_TC_easy(n_u, r_train, test_r, samples_products, device, epsilon=1e-9):
    tensor = t.t(r_train)
    TC = SparseTC(device, tensor, epsilon, mode_full=False)
    latent_1_TC, latent_2_TC = TC.TC() 
    latent_1_TC, latent_2_TC = latent_1_TC.float().cpu().detach().numpy(), latent_2_TC.float().cpu().detach().numpy()
    
    macro_scores = calculate_scores_UCTC(n_u, test_r.T, latent_1_TC, latent_2_TC, samples_products, method='TC')

    return {
        'normal_macro': calculate_macro_stats(macro_scores),
        'bootstrap_macro': calculate_bootstrap_stats(macro_scores),
    }


def run_UC_TC_easy(n_u, r_train, test_r, samples_products, device, epsilon=1e-9):
    return {
        'UC': run_UC_easy(n_u, r_train, test_r, samples_products, device, epsilon),
        'TC': run_TC_easy(n_u, r_train, test_r, samples_products, device, epsilon)
    }


def run_SVD_ranking_easy(n_u, n_m, r_train, test_r, percents, samples_products, n_bootstrap=1000):
    hash_SVD = {}
    rankSVD = rankingSVD(n_u, n_m)
    tensor = t.t(r_train)

    for percent in percents:
        pre = rankSVD.train(tensor, percent)
        pre = pre.float().cpu().detach().numpy()
        macro_scores = calculate_scores(n_u, test_r.T, samples_products, pre)

        hash_SVD[percent] = {
            'normal_macro': calculate_macro_stats(macro_scores),
            'bootstrap_macro': calculate_bootstrap_stats(macro_scores, n_bootstrap),
        }
    return hash_SVD

def run_SVD_ranking_hard(n_u, n_m, r_train, test_r, percents, samples_products, n_bootstrap=1000):
    if sparse.issparse(r_train):
        # Convert scipy sparse matrix to torch sparse tensor
        coo = r_train.tocoo().T  # Transpose here if needed
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        r_train = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)
    elif isinstance(r_train, np.ndarray):
        # Convert numpy array to torch sparse tensor
        coo = sparse.coo_matrix(r_train.T)
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        r_train = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)

    rankSVD = rankingSVD(n_u, n_m)
    hash_SVD = {}

    for percent in percents:
        tic = time()
        print(f'SVD for k={percent} starts at time {time() - tic:.2f} seconds')
        gc.collect()
        torch.cuda.empty_cache()

        Q_side_result = rankSVD.train(r_train, percent)
        macro_scores = calculate_scores_vectorized(n_u, test_r.T, samples_products, method="rankSVD", r_train=r_train, Q_side_result=Q_side_result)

        hash_SVD[percent] = {
            'normal_macro': calculate_macro_stats(macro_scores),
            'bootstrap_macro': calculate_bootstrap_stats(macro_scores, n_bootstrap),
        }
        print(f'SVD for k={percent} end at time {time() - tic:.2f} seconds')

    return hash_SVD


def run_GlocalK(dataset, n_u, n_m, train_m, train_r, test_r, test_m, samples_products, logger, n_bootstrap=1000):
    # Common hyperparameter settings
    n_hid = 500
    n_dim = 5
    n_layers = 2
    gk_size = 3

    # Different hyperparameter settings for each dataset
    if dataset == 'ML-100K':
        lambda_2 = 20.  # l2 regularisation
        lambda_s = 0.006
        iter_p = 5  # optimisation
        iter_f = 5
        epoch_p = 100 # training epoch
        epoch_f = 100
        dot_scale = 1  # scaled dot product

    elif dataset == 'ML-1M':
        lambda_2 = 70.0
        lambda_s = 0.018
        iter_p = 50
        iter_f = 10
        epoch_p = 100
        epoch_f = 100
        dot_scale = 0.5

    elif dataset == 'Douban_monti':
        lambda_2 = 10.0
        lambda_s = 0.022
        iter_p = 5
        iter_f = 5
        epoch_p = 20
        epoch_f = 60
        dot_scale = 2

    """# Network Instantiation"""

    # Usage
    print("dot_scale", dot_scale)
    R = train_r.copy()
    model = RecommenderModel(n_m, n_u, n_layers, n_hid, n_dim, gk_size, dot_scale, lambda_2, lambda_s, R)
    optimizer_label = "RMSprop"  # or "ADAM"

    if optimizer_label == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.3, epsilon=1e-07)
    elif optimizer_label == "Adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01, initial_accumulator_value=0.1, epsilon=1e-07)
    elif optimizer_label == "Nadam":
        optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    elif optimizer_label == "Adadelta":
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-07)
    else:
        optimizer = tf.keras.optimizers.Adam()

    normal_macro, bootstrap_macro, micro = None, None, None

    # Pre-training step
    @tf.function
    def train_step_pre(inputs, train_r, train_m):
        with tf.GradientTape() as tape:
            pred_p, reg_losses = model(inputs, training=True)
            diff = train_m * (train_r - pred_p)
            sqE = tf.nn.l2_loss(diff)
            loss = sqE + reg_losses

        # Filter out None gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, pred_p

    # Fine-tuning step
    @tf.function
    def train_step_fine(inputs, train_r, train_m):
        with tf.GradientTape() as tape:
            pred_f, reg_losses = model(inputs, training=False)
            diff = train_m * (train_r - pred_f)
            sqE = tf.nn.l2_loss(diff)
            loss = sqE + reg_losses
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, pred_f

    best_rmse = float("inf")
    time_cumulative = 0

    # Pre-training
    for i in range(epoch_p):
        tic = time()
        loss, pre = train_step_pre(R, train_r, train_m)
        t = time() - tic
        time_cumulative += t

        # Calculate errors
        error = (test_m * (np.clip(pre, 1., 5.) - test_r) ** 2).sum() / test_m.sum()  # test error
        test_rmse = np.sqrt(error)

        error_train = (train_m * (np.clip(pre, 1., 5.) - train_r) ** 2).sum() / train_m.sum()  # train error
        train_rmse = np.sqrt(error_train)

        # Logging
        if (i + 1) % 19 == 0:
            print('.-^-._' * 12)
            print('PRE-TRAINING')
            print(f'Epoch: {i + 1}, test rmse: {test_rmse}, train rmse: {train_rmse}')
            print(f'Time: {round(t, 2)} seconds')
            print(f'Time cumulative: {round(time_cumulative, 2)} seconds')
            print('.-^-._' * 12)

        # Delete variables no longer needed
        del loss, pre, error, error_train
        gc.collect()

    # Fine-tuning
    for i in range(epoch_f):
        tic = time()
        loss, pre = train_step_fine(R, train_r, train_m)
        t = time() - tic
        time_cumulative += t

        # Calculate errors
        error = (test_m * (tf.clip_by_value(pre, 1.0, 5.0) - test_r) ** 2).numpy().sum() / test_m.sum()  # test error
        test_rmse = np.sqrt(error)

        error_train = (train_m * (tf.clip_by_value(pre, 1.0, 5.0) - train_r) ** 2).numpy().sum() / train_m.sum()  # train error
        train_rmse = np.sqrt(error_train)

        # Clip predictions
        pre = tf.clip_by_value(pre, 1.0, 5.0).numpy()

        # Check for improvement
        if test_rmse <= best_rmse:
            macro_scores = calculate_scores(n_u, test_r.T, samples_products, pre.T)
            normal_macro = calculate_macro_stats(macro_scores)
            bootstrap_macro = calculate_bootstrap_stats(macro_scores, n_bootstrap)
            best_rmse = test_rmse


            # Logging
            logger.log('.-^-._' * 12)
            logger.log('FINE-TUNING')
            logger.log(f'Epoch: {i}, discordant pairs: {normal_macro}, {bootstrap_macro}, {micro}')
            logger.log(f'Epoch: {i}, train rmse: {train_rmse}')
            logger.log(f'Epoch: {i}, test rmse: {test_rmse} best rmse: {best_rmse}')
            logger.log(f'Time: {t} seconds')
            logger.log(f'Time cumulative: {time_cumulative} seconds')
            logger.log('.-^-._' * 12)

        # Delete variables no longer needed
        del loss, pre
        gc.collect()
        
    if abs(normal_macro['mean'] - 1.0) < 1e-4:
        breakpoint()

    # Clean up TensorFlow session
    del model, optimizer
    tf.keras.backend.clear_session()
    gc.collect()

    return {
        'normal_macro': normal_macro,
        'bootstrap_macro': bootstrap_macro,
        'micro': micro
    }


def preprocess_data(ratings):
    tic = time()
    print('preprocess data...')
    # Step 1: Compute item popularity and define long-tail items
    item_popularity = ratings.groupby('MovieID').size().reset_index(name='Count')
    item_popularity = item_popularity.sort_values(by='Count', ascending=False)

    # Calculate cumulative percentage of ratings
    item_popularity['Cumulative_Count'] = item_popularity['Count'].cumsum()
    total_ratings = item_popularity['Count'].sum()
    item_popularity['Cumulative_Percentage'] = item_popularity['Cumulative_Count'] / total_ratings

    # Define short-head (top 33% of ratings)
    short_head_threshold = 0.33
    # Define long-tail items (remaining 67% of ratings)
    long_tail_items = item_popularity[item_popularity['Cumulative_Percentage'] > short_head_threshold]['MovieID']

    # Step 2: Filter users who rated all ratings from 1 to 5
    # user_ratings = ratings.groupby('UserID')['Rating'].apply(set).reset_index()
    # user_ratings['Has_All_Ratings'] = user_ratings['Rating'].apply(lambda x: set(range(1,6)).issubset(x))
    # qualified_users = user_ratings[user_ratings['Has_All_Ratings'] == True]['UserID']

    user_rating_matrix = ratings.pivot_table(index='UserID', columns='Rating', aggfunc='size', fill_value=0)
    
    # Identify users who have at least one rating for each rating from 1 to 5
    required_ratings = [1, 2, 3, 4, 5]
    qualified_users = user_rating_matrix.loc[
        (user_rating_matrix[required_ratings] > 0).all(axis=1)
    ].index

    # Filter ratings to only include qualified users
    ratings = ratings[ratings['UserID'].isin(qualified_users)]

    # Step 3: Prepare long-tail ratings
    long_tail_ratings = ratings[ratings['MovieID'].isin(long_tail_items)]

    del item_popularity, qualified_users, ratings

    return long_tail_ratings


def train_test_split(start_rating, long_tail_ratings, ratings, mode="easy"):
    # Step 4: Identify users who have at least one rating of i and 5 in the long-tail items
    user_rating_pivot = long_tail_ratings.pivot_table(index='UserID', columns='Rating', aggfunc='size', fill_value=0)
    qualified_users = user_rating_pivot[
        (user_rating_pivot.get(start_rating, 0) >= 1) & 
        (user_rating_pivot.get(5, 0) >= 1)
    ].index

    # Filter long-tail ratings for these users
    test_candidates = long_tail_ratings[long_tail_ratings['UserID'].isin(qualified_users)]
    test_candidates_shuffled = test_candidates.sample(frac=1, random_state=42)

    # Select one sample per user for start_rating
    ratings_i = test_candidates_shuffled[test_candidates_shuffled['Rating'] == start_rating]
    # Select one sample per user for rating 5
    ratings_5 = test_candidates_shuffled[test_candidates_shuffled['Rating'] == 5]

    # Sample one rating of i and one rating of 5 per user
    ratings_i_sampled = ratings_i.groupby('UserID').apply(lambda x: x.sample(1)).reset_index(drop=True)
    ratings_5_sampled = ratings_5.groupby('UserID').apply(lambda x: x.sample(1)).reset_index(drop=True)

    # Combine sampled ratings to form the test set
    test_ratings = pd.concat([ratings_i_sampled, ratings_5_sampled])

    # Step 6: Remove test set ratings from the training set
    test_ratings['UserID_MovieID'] = test_ratings['UserID'].astype(str) + '_' + test_ratings['MovieID'].astype(str)
    ratings.loc[:, 'UserID_MovieID'] = ratings['UserID'].astype(str) + '_' + ratings['MovieID'].astype(str)
    training_ratings = ratings[~ratings['UserID_MovieID'].isin(test_ratings['UserID_MovieID'])]

    # Step 7: Ensure no users have zero ratings in the training set
    training_user_counts = training_ratings.groupby('UserID').size()
    users_with_ratings = training_user_counts[training_user_counts > 0].index
    training_ratings = training_ratings[training_ratings['UserID'].isin(users_with_ratings)]
    test_ratings = test_ratings[test_ratings['UserID'].isin(users_with_ratings)]

    # Step 8: Output matrices (if needed)
    # For example, create user-item matrices
    # Using sparse matrices for efficiency

    # Map UserID and MovieID to indices
    user_indices = {user_id: index for index, user_id in enumerate(sorted(training_ratings['UserID'].unique()))}
    movie_indices = {movie_id: index for index, movie_id in enumerate(sorted(training_ratings['MovieID'].unique()))}

    # Create training matrix
    training_ratings['User_Index'] = training_ratings['UserID'].map(user_indices)
    training_ratings['Movie_Index'] = training_ratings['MovieID'].map(movie_indices)

    # Similarly, create test matrix
    test_ratings = test_ratings[test_ratings['MovieID'].isin(movie_indices.keys())]
    test_ratings['User_Index'] = test_ratings['UserID'].map(user_indices)
    test_ratings['Movie_Index'] = test_ratings['MovieID'].map(movie_indices)

    # Convert into matrix modes
    training_matrix = csr_matrix(
        (training_ratings['Rating'].astype(np.float32), (training_ratings['Movie_Index'], training_ratings['User_Index']))
    )

    test_matrix = csr_matrix(
        (test_ratings['Rating'].astype(np.float32), (test_ratings['Movie_Index'], test_ratings['User_Index'])),
        shape=training_matrix.shape
    )

    # Step 1: Compute the number of non-zero entries per row
    non_zero_per_col = np.diff(test_matrix.indptr)

    # Step 2: Generate row indices for each non-zero entry
    col_indices = np.repeat(np.arange(test_matrix.shape[0]), non_zero_per_col)

    # Step 3: Get the column indices of non-zero entries
    row_indices = test_matrix.indices

    # Step 4: Create a DataFrame to hold the row and column indices
    df = pd.DataFrame({'row_idx': row_indices, 'col_idx': col_indices})

    # Step 5: Group by row indices and collect column indices into lists
    samples_products = df.groupby('row_idx')['col_idx'].apply(list).to_dict()

    del df

    # The training_matrix and test_matrix are ready for model training and evaluation
    # You can now proceed to train your recommendation model and evaluate it using Kendall tau metric

    # Example output (number of users and items)
    print(f"Experiment for start rating = {start_rating}")
    print(f"Number of users: {training_matrix.shape[0]}")
    print(f"Number of items: {training_matrix.shape[1]}")
    print(f"Training ratings: {training_matrix.nnz}")
    print(f"Test ratings: {test_matrix.nnz}")
    print("\n")

    # Optional: Free memory by deleting intermediate DataFrames
    del user_rating_pivot, qualified_users, test_candidates
    del ratings_i, ratings_5
    del training_user_counts, users_with_ratings
    del user_indices, movie_indices
    
    if mode == "easy":
        return training_ratings, test_ratings, np.array(training_matrix.todense()), np.array(test_matrix.todense()), samples_products
    elif mode == "hard":
        return training_ratings, test_ratings, training_matrix, test_matrix, samples_products


def load_data_ML1M_exp_9(path='./', delimiter='::', seed=1234):
    tic = time()
    print('reading data...')
    # Load the data
    ratings = pd.read_csv(
        os.path.join(path,'movielens_1m_dataset.dat'), 
        sep=delimiter, 
        engine='python', 
        names=['UserID', 'MovieID', 'Rating', 'Timestamp']
    )
    print('taken', time() - tic, 'seconds')
    return ratings

def load_data_monti_exp_9(path='./'):
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
    tic = time()
    print('reading data...')
    # Load the 'M' matrix (which contains the ratings) from the MATLAB file
    M = load_matlab_file(os.path.join(path, 'douban_monti_dataset.mat'), 'M')

    # Get the number of users and movies from the matrix dimensions
    n_u = M.shape[0]  # num of users
    n_m = M.shape[1]  # num of movies

    # Transpose the matrix to get movies as columns and users as rows (if needed)
    data_matrix = M.T

    # Find the non-zero entries in the data matrix to convert it into a DataFrame
    users, movies = data_matrix.nonzero()
    ratings = data_matrix[users, movies]

    # Create a Pandas DataFrame with columns: UserID, MovieID, Rating
    ratings = pd.DataFrame({
        'UserID': users,
        'MovieID': movies,
        'Rating': ratings
    })

    print('Data matrix loaded and converted to DataFrame.')
    print('Number of users: {}'.format(n_u))
    print('Number of movies: {}'.format(n_m))
    print('Number of ratings: {}'.format(ratings.size))
    print('taken', time() - tic, 'seconds')
    return ratings


def load_data_ML100K_exp_9(path='./', delimiter='\t', seed=1234):
    tic = time()
    print('reading data...')
    # Load the data
    # Reading the u1.base and u1.test files into pandas DataFrame
    train = pd.read_csv(os.path.join(path, 'movielens_100k_u1.base'), sep=delimiter, names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    test = pd.read_csv(os.path.join(path, 'movielens_100k_u1.test'), sep=delimiter, names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    
    # Combining the train and test sets
    ratings = pd.concat([train, test], axis=0)
    
    print('taken', time() - tic, 'seconds')
    return ratings

def load_data_Netflix_exp_9(path='./', files=[1, 2]):
    tic = time()
    print('reading data...')
    
    # Load and process data
    dfs = []
    movie_ids = []
    
    for i in files:
        path_file = os.path.join(path, f'combined_data_{i}.txt')
        df = pd.read_csv(path_file, header = None, names = ['UserID', 'Rating'], usecols = [0,1])
        df['Rating'] = df['Rating'].astype(float)
        dfs.append(df)
    
    ratings = pd.concat(dfs, ignore_index=True)
    ratings.index = np.arange(0,len(ratings))

    del dfs  # Free up memory

    ratings_nan = pd.DataFrame(pd.isnull(ratings.Rating))
    ratings_nan = ratings_nan[ratings_nan['Rating'] == True]
    ratings_nan = ratings_nan.reset_index()

    movie_np = []

    # Calculate the differences between consecutive indices
    diff = np.diff(ratings_nan['index'])

    # Create an array of movie IDs
    movie_ids = np.arange(1, len(diff) + 1)

    # Create the repeated movie IDs
    movie_np = np.repeat(movie_ids, diff - 1)

    # Account for the last record
    last_record = np.full(len(ratings) - ratings_nan.iloc[-1, 0] - 1, len(diff) + 1)
    movie_np = np.concatenate([movie_np, last_record])

    ratings = ratings[pd.notnull(ratings['Rating'])]

    ratings['MovieID'] = movie_np.astype(int)
    ratings['UserID'] = ratings['UserID'].astype(int)
    
    print(f'Data loaded in {time() - tic:.2f} seconds')
    print('taken', time() - tic, 'seconds')
    return ratings

def load_data_ML20M_exp_9(path='./'):
    tic = time()
    print('reading data...')
    # Load the data
    ratings = pd.read_csv(os.path.join(path, 'ratings.csv'), usecols=['userId', 'movieId', 'rating', "timestamp"], header=0)
    ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    print('taken', time() - tic, 'seconds')
    return ratings


def save_ratings_to_inter_recbole(
    training_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    out_dir: str = "data/splits",
    prefix: str = "mydata"
):
    """
    Save training and test ratings DataFrames in RecBole atomic .inter format.

    Args:
        training_ratings: pd.DataFrame with columns [UserID, MovieID, Rating, Timestamp]
        test_ratings: pd.DataFrame with the same columns
        out_dir: directory path to save files
        prefix: prefix for the output file names (e.g., 'mydata')
    """
    os.makedirs(out_dir, exist_ok=True)

    # Select only required columns
    cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    training_ratings = training_ratings[cols].copy()
    test_ratings = test_ratings[cols].copy()

    # Define atomic .inter file paths
    train_path = os.path.join(out_dir, f"{prefix}.train.inter")
    test_path  = os.path.join(out_dir, f"{prefix}.test.inter")

    # Save as tab-separated .inter files
    training_ratings.to_csv(train_path, sep="\t", index=False)
    test_ratings.to_csv(test_path, sep="\t", index=False)

    print(f"✅ Saved training data to: {train_path}")
    print(f"✅ Saved test data to:     {test_path}")

    return train_path, test_path


def save_ratings_for_msft_recommenders(
    training_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    out_dir: str = "data/splits",
    prefix: str = "mydata",
    task: str = "implicit",  # or "explicit"
):
    """
    Save training and test ratings in a format suitable for Microsoft Recommenders.

    Expected columns after normalization:
        userID, itemID, rating, timestamp (timestamp optional)

    Args:
        training_ratings: DataFrame with at least user/item columns
        test_ratings: DataFrame with the same schema
        out_dir: where to write CSVs
        prefix: filename prefix
        task: "implicit" -> fill rating=1.0 if missing
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) normalize column names from your RecBole-style data
    col_map = {
        "UserID": "userID",
        "user_id": "userID",
        "MovieID": "itemID",
        "ItemID": "itemID",
        "Rating": "rating",
        "Timestamp": "timestamp",
    }

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})
        # make sure essential cols exist
        if "userID" not in df.columns or "itemID" not in df.columns:
            raise ValueError("DataFrame must contain user and item columns.")
        # implicit case: fill rating
        if task == "implicit":
            if "rating" not in df.columns:
                df["rating"] = 1.0
        return df

    train_norm = _normalize(training_ratings).copy()
    test_norm = _normalize(test_ratings).copy()

    # 2) select only what MS Recommenders usually needs
    keep_cols = [c for c in ["userID", "itemID", "rating", "timestamp"] if c in train_norm.columns]
    train_norm = train_norm[keep_cols]
    test_norm = test_norm[keep_cols]

    # 3) write CSVs
    train_path = os.path.join(out_dir, f"{prefix}_train_msft.csv")
    test_path = os.path.join(out_dir, f"{prefix}_test_msft.csv")

    train_norm.to_csv(train_path, index=False)
    test_norm.to_csv(test_path, index=False)

    print(f"✅ Saved training data to: {train_path}")
    print(f"✅ Saved test data to:     {test_path}")

    return train_path, test_path