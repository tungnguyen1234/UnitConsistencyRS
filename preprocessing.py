
import numpy as np
from time import time
import random
from scipy import sparse
import pandas as pd


##### Experiment 3
def get_subarray_with_min_ratings(array, min_ratings=30):
    """
    Filter the array to include only columns where the number of non-zero
    entries is at least `min_ratings`.

    :param array: 2D numpy array with shape (n_products, n_users).
    :param min_ratings: The minimum number of non-zero entries required per column.
    :return: Filtered 2D numpy array.
    """
    # Count the non-zero entries in each column
    user_ratings_count = np.count_nonzero(array > 0, axis=0)
    
    # Find the indices of users with at least `min_ratings` ratings
    valid_user_indices = np.where(user_ratings_count >= min_ratings)[0]
    
    # Select columns with valid users
    subarray = array[:, valid_user_indices]

    rows_with_nonzero = ~(np.all(subarray == 0, axis=1))  # Rows where not all elements are zero
    subarray = subarray[rows_with_nonzero] 
    
    return subarray


def sample_products_all_users(train_r):
    """
    Sample 10% of nonzero indices values in arr[:, idx], set those values to 0 to create new arr,
    and check whether the new arr has any rows that only have 0.

    :param arr: 2D numpy array.
    :param idx: Column index to modify.
    :return: Modified array and a boolean indicating if any row is entirely 0.
    """
    # Copy the original array to avoid modifying it directly
    

    n_m, n_u = train_r.shape
    


    while True:
        train_r_N = np.copy(train_r)  # Reset train_r_N to arr at each iteration if required
        modification_made = False  # Flag to check if any modification was made
        rows = []
        
        for user in range(n_u):
            # Get indices of non-zero elements in the specified column
            non_zero_indices = np.nonzero(train_r_N[:, user])[0]
            
            # Sample 10% of these indices, ensuring at least one is selected if any are non-zero
            sample_size = min(int(len(non_zero_indices) * 0.1), 10)
            sampled_indices = np.random.choice(non_zero_indices, size=sample_size, replace=False)
            # Set the sampled values to 0
            train_r_N.T[user, sampled_indices] = 0
            modification_made = True
            # get all indices for each row
            rows.append(sampled_indices)

        # Check if the condition to break the loop is met
        # print(np.all(np.sum(train_r_N, axis=1) > 0).any(), np.all(np.sum(train_r_N, axis=0) > 0).any(), np.where(np.all(train_r_N == 0, axis=1))[0])
        if not np.all(train_r_N == 0, axis=1).any() and modification_made:
            break

    train_m_N = np.greater(train_r_N, 1e-12).astype('float32')  # masks indicating non-zero entries
    return train_r_N, train_m_N, rows

##### Experiment 4 + 5

def get_list_by_indices(user, train_r, ratings = [1, 5]):
    # Initialize an empty list to store the indices
    indices = []
    
    # Loop through the numbers 1 to 5 to find matching column indices
    for j in ratings:  
        # Find indices in row 'idx' where the value equals 'j'
        matched_indices = np.where(train_r.T[user] == j)[0]
        
        # If there are matching indices, randomly choose one and add to the list
        if matched_indices.size > 0:
            chosen_index = np.random.choice(matched_indices)
            indices.append(chosen_index)

    # Ensure we have found exactly 5 distinct indices
    if len(set(indices)) == len(ratings):
        return indices
    else:
        return []


def sample_products_all_users_by_indices(train_r, ratings=[1, 5], users=None):
    """
    Sample indices for all users based on the ratings, modify the original matrix,
    and eliminate rows that have all zero values. Also, update rows[user] accordingly.

    :param train_r: 2D NumPy matrix
    :param ratings: List of ratings to consider
    :param users: List of users to process (optional)
    :return: Modified matrix, mask matrix, and dictionary of sampled indices per user
    """
    n_m, n_u = train_r.shape  # n_m: number of products, n_u: number of users

    # Initialize users if not provided
    if users is None:
        users = range(n_u)

    # Copy the original matrix to avoid modifying it directly
    train_r_N = np.copy(train_r)
    rows = {}

    for user in users:
        # Get sampled indices for the current user
        sampled_indices = get_list_by_indices(user, train_r, ratings)
        sampled_indices = np.array(sampled_indices)

        # If sampled indices are valid, proceed
        if sampled_indices.size > 0:
            # Set the sampled values to 0 for the user
            train_r_N.T[user, sampled_indices] = 0

        # Store the sampled indices for the current user
        rows[user] = sampled_indices

    # Eliminate rows with all zeros
    non_zero_row_mask = ~np.all(train_r_N == 0, axis=1)
    train_r_N_filtered = train_r_N[non_zero_row_mask]
    train_r_filtered = train_r[non_zero_row_mask]

    # Update the sampled indices to reflect the filtered rows
    filtered_rows = {}
    row_mapping = np.where(non_zero_row_mask)[0]  # Get the indices of the non-zero rows
    for user, indices in rows.items():
        filtered_indices = [row_mapping.tolist().index(idx) for idx in indices if idx in row_mapping]
        filtered_rows[user] = filtered_indices

    # Create a binary mask indicating non-zero entries
    train_m_N = np.greater(train_r_N_filtered, 1e-12).astype('float32')

    return train_r_filtered, train_r_N_filtered, train_m_N, filtered_rows


def get_users_from_indices(arr, ratings):
   # Target set of integers
    target_set = set([0] + ratings)
                                                                                                                                 
    # Check for columns that match the target set exactly
    valid_columns_indices = np.array([col for col in range(arr.shape[1]) if target_set.issubset(set(arr[:, col]))])

    print(valid_columns_indices)
    # Extract the relevant columns based on valid_columns_indices
    relevant_arr = arr[:, valid_columns_indices]

    # Find indices of rows and columns where all elements are nonzero
    rows_with_nonzeros = np.any(relevant_arr != 0, axis=1)
    cols_with_nonzeros = np.any(relevant_arr != 0, axis=0)

    # Filter the array to keep only those rows and columns
    filtered_arr = relevant_arr[np.ix_(rows_with_nonzeros, cols_with_nonzeros)]

    return filtered_arr

##### Experiment 5
# Sample 100 numbers between 1 and n
def sampled_users(n_u, N=100):
    sampled_numbers = random.sample(range(1, n_u+1), N)
    return sampled_numbers


##### Experiment 7


def sample_products_all_users_by_indices_sparse(train_r, ratings, users=None):
    """
    Sample all users between ratings of nonzero indices values in train_r, set those values to 0 to create new train_r_N,
    and check whether the new train_r_N has any rows that only have 0. Filter out rows that are completely zero.

    :param train_r: Sparse matrix (CSR format).
    :param ratings: List of ratings to consider.
    :param users: List of users to process (optional).
    :return: Modified sparse matrix, sparse mask matrix, and dictionary of sampled indices per user.
    """
    n_m, n_u = train_r.shape
    tic = time()
    
    # If no users provided, process all users
    if users is None:
        users = np.arange(n_u)

    # Convert the matrix to COO format to efficiently access data
    train_r_coo = train_r.tocoo()

    # Create a mask for the ratings we're interested in
    ratings_mask = np.isin(train_r_coo.data, ratings)

    # Get the indices of the rows (items) and columns (users) for the ratings we want to modify
    row_indices = train_r_coo.row[ratings_mask]
    col_indices = train_r_coo.col[ratings_mask]
    data_values = train_r_coo.data[ratings_mask]

    # Dictionary to store sampled indices for each user
    sampled_indices_dict = {}

    # Vectorized user processing
    unique_users, inverse_indices = np.unique(col_indices, return_inverse=True)
    print(f'vectorize data loaded in {time() - tic:.2f} seconds')

    # Method to quickly retrieve two data points for each user such that their values match with distinct values in ratings.

    data = pd.DataFrame({
        'user': col_indices,
        'item': row_indices,
        'rating': data_values
    })

    # Group by 'user' and 'rating', and collect 'item' indices into lists
    grouped = data.groupby(['user', 'rating'])['item'].apply(list).reset_index()

    # Randomly sample one 'item' per group (user-rating combination)
    grouped['sampled_item'] = grouped['item'].apply(np.random.choice)

    # Pivot the DataFrame to have users as index and ratings as columns
    sampled_pivot = grouped.pivot(index='user', columns='rating', values='sampled_item')

    # Function to collect sampled items for each user
    def get_sampled_items(row):
        return [row.get(rating) for rating in ratings if not pd.isnull(row.get(rating))]

    # Apply the function to each row to create the dictionary
    sampled_indices_dict = sampled_pivot.apply(get_sampled_items, axis=1).to_dict()

    # Remove any None values from the lists (in case there are missing ratings)
    sampled_indices_dict = {
        user: [item for item in items if item is not None]
        for user, items in sampled_indices_dict.items()
    }

    print(f'sampled_indices_dict at time {time() - tic:.2f} seconds')

    # Modify the matrix: Set the sampled values to 0 in the original matrix
    train_r_N = train_r.tolil()  # Convert to LIL format for efficient row-based modification
    for user, sampled_indices in sampled_indices_dict.items():
        train_r_N[sampled_indices, user] = 0

    # Convert back to CSR format and eliminate zeros
    train_r_N = train_r_N.tocsr()
    train_r_N.eliminate_zeros()

    # Now we need to filter out rows that are entirely zero
    non_zero_row_mask = np.diff(train_r_N.indptr) > 0  # Rows with at least one non-zero value
    train_r_N_filtered = train_r_N[non_zero_row_mask]
    train_r_filtered = train_r[non_zero_row_mask]

    print(f'filter out nonzero rows at time {time() - tic:.2f} seconds')

    # Assuming 'non_zero_row_mask' is a boolean numpy array
    row_mapping = np.where(non_zero_row_mask)[0]  # Indices of non-zero rows
    # Create a pandas Series mapping from original indices to positions in 'row_mapping'
    row_mapping_series = pd.Series(data=np.arange(len(row_mapping)), index=row_mapping)
    filtered_rows = {}
    for user, indices in sampled_indices_dict.items():
        if user % 10000 == 2:
            print(unique_users, user)
        # Convert indices to a pandas Index for efficient operations
        indices_series = pd.Index(indices)
        # Use pandas indexing to get positions in 'row_mapping' for the indices
        # The '.reindex' method aligns the indices and fills missing values with NaN
        filtered_indices = row_mapping_series.reindex(indices_series).dropna().astype(int).tolist()
        filtered_rows[user] = filtered_indices

    print(f'sampled indices to reflect the filtered rows in {time() - tic:.2f} seconds')


    # Create a binary mask indicating non-zero entries in the filtered matrix
    train_m_N = sparse.csr_matrix(train_r_N_filtered)
    train_m_N.data = np.ones_like(train_m_N.data)  # Set all non-zero elements to 1 for the mask

    print(f'Final output in {time() - tic:.2f} seconds')

    return train_r_filtered, train_r_N_filtered, train_m_N, filtered_rows

def get_users_from_indices_sparse(arr, ratings):
    """
    Get users (columns) from the sparse matrix that contain all specified ratings.

    :param arr: Sparse matrix (CSR format).
    :param ratings: List of ratings to consider.
    :return: Filtered sparse matrix.
    """
    target_set = set(ratings)
    n_ratings = len(target_set)
    tic = time()
    
    # Convert to CSC format for efficient column operations
    arr_csc = arr.tocsc()
    
    # Initialize an array to count ratings for each column
    valid_columns = np.ones(arr_csc.shape[1], dtype=bool)
    for rating in ratings:
        valid_columns &= (arr_csc == rating).sum(axis=0).A1 > 0
    
    # Find columns that have at least all the target ratings
    valid_columns = np.where(valid_columns)[0]
    
    if len(valid_columns) == 0:
        return sparse.csr_matrix((0, 0))
    
    print(f'get_users_from_indices_sparse loaded in {time() - tic:.2f} seconds')
        
    # Extract the relevant columns based on valid_columns
    relevant_arr = arr[:, np.array(valid_columns)]

    print(f'get_users_from_indices_sparse loaded in {time() - tic:.2f} seconds')

    # Find indices of rows and columns where not all elements are zero
    rows_with_nonzeros = np.diff(relevant_arr.indptr) > 0
    cols_with_nonzeros = relevant_arr.getnnz(axis=0) > 0

    print(f'Non-zero elements identified in {time() - tic:.2f} seconds')

    # Filter the array to keep only those rows and columns
    filtered_arr = relevant_arr[rows_with_nonzeros, :][:, cols_with_nonzeros]

    print(f'Final filtering completed in {time() - tic:.2f} seconds')
    return filtered_arr


