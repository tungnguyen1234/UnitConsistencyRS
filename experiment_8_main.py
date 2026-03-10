# main_execution.py

import numpy as np
from datetime import datetime
import json 
from scipy import sparse
# Import utility functions and algorithms
from experiment_utils import *
from utils import Logger

percents = [0.15]


def save_data(json_file_path, data):
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

def run_hard(dataset, n_m, n_u, train_r, train_m, experiment=8):
    dt_string = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    logger = Logger(f'GlocalK_{dataset}_{dt_string}.log')

    torch.manual_seed(1284)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.log("Start the process")
    ratings_arr = [[i, 5] for i in range(1, 5)]
    train_r_sparse = sparse.csr_matrix(train_r)
    iterations = 5
    methods = ["rankSVD", "UCTC"]
    train_r_sparse = get_users_from_indices_sparse(train_r_sparse, [1, 2, 3, 4, 5])

    for ratings in ratings_arr:  
        for i in range(iterations):
            train_r_ratings_curr, train_r_N, train_m_N, samples_products = sample_products_all_users_by_indices_sparse(train_r_sparse, ratings)
            n_m, n_u = train_r_ratings_curr.shape
            logger.log(f"Start the process for experiment {experiment} at step {i}")
            test_r = train_r_ratings_curr.copy()
            test_r = test_r - train_r_N  # Use sparse matrix subtraction
            if "UCTC" in methods:
                hash_UCTC = run_UC_TC_hard(n_u, train_r_N, test_r, samples_products, device)
                json_file_path = f'{dataset}/data_user_experiment_{experiment}_{ratings}_{dataset}_{i}_UCTC.json'
                save_data(json_file_path, hash_UCTC)
            if "rankSVD" in methods:
                hash_SVD = run_SVD_ranking_hard(n_u, n_m, train_r_N, test_r, percents = percents, samples_products=samples_products)
                json_file_path = f'{dataset}/data_user_experiment_{experiment}_{ratings}_{dataset}_{i}_RankSVD.json'
                save_data(json_file_path, hash_SVD)
            
            del train_r_N, train_m_N, test_r, train_r_ratings_curr
            gc.collect()
            torch.cuda.empty_cache()


def run_easy(dataset, n_m, n_u, train_r, train_m, experiment=8):
    dt_string = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    logger = Logger(f'GlocalK_{dataset}_{dt_string}.log')

    torch.manual_seed(1284)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.log("Start the GlocalK process")
    ratings_arr = [[i, 5] for i in range(1, 5)]
    iterations = 10

    train_r = get_users_from_indices(train_r, [1, 2, 3, 4, 5])
    # methods = ["UCTC", "GLocalK", "rankSVD"]
    methods = ["GLocalK"]

    for ratings in ratings_arr:  
        for i in range(iterations):
            train_r_ratings_curr, train_r_N, train_m_N, samples_products = sample_products_all_users_by_indices(train_r, ratings)
            train_m = np.greater(train_r_ratings_curr, 1e-12).astype('float32')
            n_m, n_u = train_r_ratings_curr.shape

            logger.log(f"Start the GlocalK process for experiment {experiment} at step {i}")
            test_r = train_r_ratings_curr.copy()
            test_m = train_m.copy()

            test_r = test_r - train_r_N
            test_m = test_m - train_m_N

            r_train = torch.tensor(train_r_N).to(device)

            if "GLocalK" in methods:
                hash_result = {}
                result_glocalk = run_GlocalK(dataset, n_u, n_m, train_m, train_r_N, test_r, test_m, samples_products, logger)
                hash_result["GLocalK"] = result_glocalk
                json_file_path = f'{dataset}/data_user_experiment_{experiment}_{ratings}_{dataset}_{i}_GLocalK.json'
                save_data(json_file_path, hash_result)
            if "UCTC" in methods:
                hash_UCTC = run_UC_TC_easy(n_u, r_train, test_r, samples_products, device)
                json_file_path = f'{dataset}/data_user_experiment_{experiment}_{ratings}_{dataset}_{i}_UCTC.json'
                save_data(json_file_path, hash_UCTC)
            if "rankSVD" in methods:
                hash_SVD = run_SVD_ranking_easy(n_u, n_m, r_train, test_r, percents, samples_products)
                json_file_path = f'{dataset}/data_user_experiment_{experiment}_{ratings}_{dataset}_{i}_RankSVD.json'
                save_data(json_file_path, hash_SVD)

            del train_r_N, train_m_N, train_r_ratings_curr

if __name__ == "__main__":
    # Add any main execution code here if needed
    pass