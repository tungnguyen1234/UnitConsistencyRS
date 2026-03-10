"""
Optimized experiment_9_main.py

Key changes:
1. Support for fast ranking models (BPR-MF, LightGCN-Simple)
2. Optimized RecBole integration
3. Better memory management
"""
from datetime import datetime
from experiment_utils import *
from utils import Logger, load_json, save_data

# Import optimized modules
try:
    from fast_ranking_models import run_bpr_mf, run_lightgcn_simple
    FAST_MODELS_AVAILABLE = True
except ImportError:
    FAST_MODELS_AVAILABLE = False
    print("Warning: fast_ranking_models not found. Using RecBole models only.")

try:
    from algos_optimized import run_algo_hard, run_algo_easy
    OPTIMIZED_RECBOLE = True
except ImportError:
    from algos_optimized import run_algo_hard, run_algo_easy
    OPTIMIZED_RECBOLE = False
    print("Warning: algos_optimized not found. Using original RecBole implementation.")

percents = [0.15]
PATHS = load_json(os.path.join(os.path.dirname(__file__), 'paths.json'))
temp_dir = PATHS['temp_dir']
config_dir = PATHS['config_dir']
methods = PATHS['methods']

# Model speed categories
FAST_MODELS = ['BPR-MF', 'LightGCN-Simple', 'BPR', 'MF']
RECBOLE_MODELS = ['SASRec', 'BERT4Rec', 'LightGCN', 'NGCF', 'NCF']


def run_fast_model(method, n_u, n_m, train_r, test_r, samples_products, device, logger):
    """Run fast ranking models (no RecBole dependency)."""
    if method in ['BPR-MF', 'BPR', 'MF']:
        return run_bpr_mf(
            n_u, train_r, test_r, samples_products, device, logger,
            embedding_dim=64, epochs=30, batch_size=2048
        )
    elif method == 'LightGCN-Simple':
        return run_lightgcn_simple(
            n_u, train_r, test_r, samples_products, device, logger,
            embedding_dim=64, n_layers=2, epochs=30, batch_size=2048
        )
    else:
        raise ValueError(f"Unknown fast model: {method}")


def run_hard(ratings, dataset, experiment=9, mode="hard", data_mode=False):
    dt_string = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    logger = Logger(f'Experiment_{dataset}_{dt_string}.log')

    torch.manual_seed(1284)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.log(f"Start experiment on {dataset} (mode={mode})")
    logger.log(f"Device: {device}")
    logger.log(f"Methods: {methods}")
    if OPTIMIZED_RECBOLE:
        logger.log("Using optimized RecBole implementation")
    if FAST_MODELS_AVAILABLE:
        logger.log("Fast ranking models available")
    
    iterations = 5
    long_tail_ratings = preprocess_data(ratings)
    
    for start_rating in [1, 4]:
        pairs = [start_rating, 5]
        logger.log(f"\n{'='*50}")
        logger.log(f"Processing rating pair: {pairs}")
        
        for i in range(iterations):
            logger.log(f"\n--- Iteration {i+1}/{iterations} ---")
            
            training_ratings, test_ratings, train_r, test_r, samples_products = train_test_split(
                start_rating, long_tail_ratings, ratings, mode=mode
            )
            
            n_m, n_u = train_r.shape
            logger.log(f"Matrix shape: {n_m} items x {n_u} users")
            logger.log(f"Test samples: {len(samples_products)} users")
            
            # Process each method
            if "UCTC" in methods:
                hash_UCTC = run_UC_TC_hard(n_u, train_r, test_r, samples_products, device)
                json_file_path = f'{dataset}/data_user_experiment_{experiment}_{pairs}_{dataset}_{i}_UCTC.json'
                save_data(json_file_path, hash_UCTC)
                
            elif "rankSVD" in methods:
                hash_SVD = run_SVD_ranking_hard(n_u, n_m, train_r, test_r, percents=percents, 
                                                samples_products=samples_products)
                json_file_path = f'{dataset}/data_user_experiment_{experiment}_{pairs}_{dataset}_{i}_RankSVD.json'
                save_data(json_file_path, hash_SVD)
                
            else:
                for method in methods:
                    logger.log(f'\n{"="*60}')
                    logger.log(f'Running {method}')
                    
                    try:
                        # Check if it's a fast model
                        if FAST_MODELS_AVAILABLE and method in FAST_MODELS:
                            results = run_fast_model(
                                method, n_u, n_m, train_r, test_r, 
                                samples_products, device, logger
                            )
                        else:
                            # Use RecBole (optimized or original)
                            results = run_algo_hard(
                                n_u, train_r, test_r, samples_products, device, logger,
                                epsilon=1e-9, algo_name=method, 
                                temp_dir=temp_dir, config_dir=config_dir
                            )
                        
                        logger.log(f'{method} Results: {results["kendall_tau_stats"]}')
                        
                    except Exception as e:
                        logger.log(f'{method} FAILED: {str(e)}')
                        import traceback
                        logger.log(traceback.format_exc())
                        results = {'error': str(e)}
                    
                    json_file_path = f'{dataset}/data_user_experiment_{experiment}_{pairs}_{dataset}_{i}_{method}.json'
                    save_data(json_file_path, results)
                    logger.log(f'{"="*60}\n')
            
            # Memory cleanup
            del train_r, test_r
            gc.collect()
            torch.cuda.empty_cache()
    
    logger.close()


def run_easy(ratings, dataset, experiment=9, mode="easy", data_mode=False):
    dt_string = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    logger = Logger(f'Experiment_{dataset}_{dt_string}.log')

    torch.manual_seed(1284)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.log(f"Start experiment on {dataset} (mode={mode})")
    logger.log(f"Device: {device}")
    logger.log(f"Methods: {methods}")
    
    iterations = 10
    long_tail_ratings = preprocess_data(ratings)
    
    for start_rating in [1, 4]:
        pairs = [start_rating, 5]
        logger.log(f"\n{'='*50}")
        logger.log(f"Processing rating pair: {pairs}")
        
        for i in range(iterations):
            logger.log(f"\n--- Iteration {i+1}/{iterations} ---")
            
            training_ratings, test_ratings, train_r, test_r, samples_products = train_test_split(
                start_rating, long_tail_ratings, ratings, mode=mode
            )
            
            n_m, n_u = train_r.shape
            train_m = np.greater(train_r, 1e-12).astype('float32')
            test_m = np.greater(test_r, 1e-12).astype('float32')
            r_train = torch.tensor(train_r).to(device)
            
            logger.log(f"Matrix shape: {n_m} items x {n_u} users")
            
            if "GLocalK" in methods:
                hash_result = {}
                result_glocalk = run_GlocalK(dataset, n_u, n_m, train_m, train_r, test_r, 
                                             test_m, samples_products, logger)
                hash_result["GLocalK"] = result_glocalk
                json_file_path = f'{dataset}/data_user_experiment_{experiment}_{pairs}_{dataset}_{i}_GLocalK.json'
                save_data(json_file_path, hash_result)
                
            elif "UCTC" in methods:
                hash_UCTC = run_UC_TC_easy(n_u, r_train, test_r, samples_products, device)
                json_file_path = f'{dataset}/data_user_experiment_{experiment}_{pairs}_{dataset}_{i}_UCTC.json'
                save_data(json_file_path, hash_UCTC)
                
            elif "rankSVD" in methods:
                hash_SVD = run_SVD_ranking_easy(n_u, n_m, r_train, test_r, percents, samples_products)
                json_file_path = f'{dataset}/data_user_experiment_{experiment}_{pairs}_{dataset}_{i}_RankSVD.json'
                save_data(json_file_path, hash_SVD)
                
            else:
                for method in methods:
                    logger.log(f'\n{"="*60}')
                    logger.log(f'Running {method}')
                    
                    try:
                        # For easy mode, convert to sparse for fast models
                        train_r_sparse = sparse.csr_matrix(train_r) if isinstance(train_r, np.ndarray) else train_r
                        
                        if FAST_MODELS_AVAILABLE and method in FAST_MODELS:
                            results = run_fast_model(
                                method, n_u, n_m, train_r_sparse, test_r,
                                samples_products, device, logger
                            )
                        else:
                            results = run_algo_easy(
                                n_u, train_r, test_r, samples_products, device, logger,
                                epsilon=1e-9, algo_name=method,
                                temp_dir=temp_dir, config_dir=config_dir
                            )
                        
                        logger.log(f'{method} Results: {results["kendall_tau_stats"]}')
                        
                    except Exception as e:
                        logger.log(f'{method} FAILED: {str(e)}')
                        import traceback
                        logger.log(traceback.format_exc())
                        results = {'error': str(e)}
                    
                    json_file_path = f'{dataset}/data_user_experiment_{experiment}_{pairs}_{dataset}_{i}_{method}.json'
                    save_data(json_file_path, results)
                    logger.log(f'{"="*60}\n')
            
            # Memory cleanup
            del r_train, train_m, test_m
            gc.collect()
            torch.cuda.empty_cache()
    
    logger.close()
