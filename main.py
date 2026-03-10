import experiment_8_main, experiment_9_main_optimized
from dataloader import *
from experiment_utils import *
import os
from utils import load_json
import numpy as np

# Temporary compatibility patch for NumPy 2.0+
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "int"):
    np.int = np.int64
if not hasattr(np, "bool"):
    np.bool = np.bool_

if __name__ == "__main__":
    # Insert the path of a data directory by yourself (e.g., 'C:\\Users\\...\\data')
    # .-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._
    # Modify this for your task: 'ML-100K', 'Douban_monti'
    data_mode = False
    easy_set = ['ML-100K', 'ML-1M', 'Douban_monti']
    hard_set = ['ML-20M', "Netflix"]
    datasets = ['ML-1M']
    experiment = 9
    PATHS = load_json(os.path.join(os.path.dirname(__file__), 'paths.json'))

    for dataset in datasets:
        if experiment == 8:
            experiment_main = experiment_8_main
        elif experiment == 9:
            experiment_main = experiment_9_main_optimized

        os.makedirs(f'{dataset}', exist_ok=True)
        # Data Load
        try:            
            data_path = PATHS['data_path']
            if experiment == 8:
                # .-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._
                if dataset in easy_set:
                    path = os.path.join(data_path, dataset)
                    if dataset == 'ML-100K':
                        n_m, n_u, train_r, train_m = load_data_100k(path=path)
                    elif dataset == 'ML-1M':
                        n_m, n_u, train_r, train_m = load_data_1m(path=path, delimiter='::', seed=1234)
                    elif dataset == 'Douban_monti':  # Assuming 'Monti' is the identifier for the Monti dataset
                        n_m, n_u, train_r, train_m = load_data_monti(path=path)
                    experiment_main.run_easy(dataset, n_m, n_u, train_r, train_m, data_mode)
                elif dataset in hard_set:
                    path = os.path.join(data_path, dataset)
                    if dataset == 'ML-20M':
                        n_m, n_u, train_r, train_m = load_big_data_ML_20M(path=path, seed=1234)
                    else:
                        n_m, n_u, train_r, train_m = load_big_data_Netflix(path=path, seed=1234)
                    experiment_main.run_hard(dataset, n_m, n_u, train_r, train_m, data_mode)
            elif experiment == 9:
                if dataset in easy_set:
                    path = os.path.join(data_path, dataset)
                    if dataset == 'ML-100K':
                        ratings = load_data_ML100K_exp_9(path=path)
                    elif dataset == 'ML-1M':
                        ratings = load_data_ML1M_exp_9(path=path, delimiter='::', seed=1234)
                    elif dataset == 'Douban_monti':  # Assuming 'Monti' is the identifier for the Monti dataset
                        ratings = load_data_monti_exp_9(path=path)
                    experiment_main.run_easy(ratings, dataset, mode="easy", data_mode=data_mode)
                elif dataset in hard_set:
                    path = os.path.join(data_path, dataset)
                    if dataset == 'ML-20M':
                        ratings = load_data_ML20M_exp_9(path=path)
                    else:
                        ratings = load_data_Netflix_exp_9(path=path)
                    experiment_main.run_hard(ratings, dataset, mode="hard", data_mode=data_mode)

        except Exception as e:
            raise e