# UC Benchmarking

Evaluation suite for recommender systems comparing UC (Unit Consistency) against standard baselines on both **Kendall-tau rank consistency** and **standard ranking metrics** (Precision@k, Recall@k, NDCG@k).

## Download Dataset

https://drive.google.com/drive/folders/116WrCmaHDyLox9KmL3-YL1GbGWmK-r8S?usp=sharing

Place downloaded data under `data/` so the structure is:
```
data/
  ML-100K/
  ML-1M/
  ML-20M/
  Douban_monti/
  Netflix/
```

## Supported Datasets

| Dataset | Users | Items | Notes |
|---------|-------|-------|-------|
| ML-100K | ~1K | ~1.7K | Small, fast |
| ML-1M | ~6K | ~3.7K | Medium |
| Douban_monti | ~3K | ~3.7K | Medium |
| ML-20M | ~138K | ~27K | Large |
| Netflix | ~440K | ~9K | Large, GPU memory constrained |

## Supported Methods

- **UC** — Unit Consistency (deterministic)
- **TC** — Tensor Completion (deterministic)
- **BPR-MF** — Bayesian Personalized Ranking with Matrix Factorization
- **LightGCN** — Light Graph Convolutional Network (RecBole-GNN)
- **SimGCL** — Simple Graph Contrastive Learning (RecBole-GNN)
- **NGCF** — Neural Graph Collaborative Filtering (RecBole-GNN)
- **NCF** — Neural Collaborative Filtering (RecBole)
- **SGL** — Self-supervised Graph Learning (RecBole-GNN)
- **NCL** — Neighborhood-enriched Contrastive Learning (RecBole-GNN)
- **SimpleX** — Simple Contrastive Learning (RecBole)
- **MCCLK** — Multi-Context Contrastive Learning for Knowledge Graphs (RecBole-GNN)
- **fairGAN_tf** — FairGAN (TensorFlow implementation)

## Per-Dataset Configs

Model hyperparameters are stored as YAML files under `configs/recbole/{Dataset}/`:

```
configs/recbole/
  ML-100K/
    lightgcn_local.yaml
    simgcl_local.yaml
    bprmf_local.yaml
    fairgan_local.yaml
  ML-1M/
    ...
  Douban_monti/
    ...
  ML-20M/
    ...
  Netflix/
    ...
```

Key differences by dataset size:
- **ML-100K / ML-1M / Douban_monti**: `embedding_size=64`, `n_layers=3`, `batch_size=4096`
- **ML-20M**: `embedding_size=32`, `n_layers=2`, `batch_size=8192`
- **Netflix**: `embedding_size=16`, `n_layers=1`, `batch_size=24576` (fits 12GB GPU)

---

## 1. Standard Ranking Evaluation

`run_standard_ranking_eval.py` — Main experiment script. Trains all specified methods and evaluates on Precision@k, Recall@k, NDCG@k, with optional Kendall-tau and CVR metrics.

```bash
# Basic: evaluate UC, BPR-MF, LightGCN on ML-1M
python run_standard_ranking_eval.py --dataset ML-1M --methods UC BPR-MF LightGCN

# Multiple seeds for mean/std
python run_standard_ranking_eval.py --dataset ML-1M --methods UC BPR-MF LightGCN --random_state 0
python run_standard_ranking_eval.py --dataset ML-1M --methods UC BPR-MF LightGCN --random_state 42

# All methods on Netflix with UC metrics
python run_standard_ranking_eval.py --dataset Netflix \
  --methods UC BPR-MF LightGCN SimGCL fairGAN_tf \
  --compute_uc_metrics --random_state 0

# Custom k values and evaluation mode
python run_standard_ranking_eval.py --dataset ML-1M \
  --methods UC BPR-MF LightGCN \
  --k_values 5 10 20 \
  --eval_mode full_ranking

# UC as a re-ranking module on top of base models
python run_standard_ranking_eval.py --dataset ML-1M \
  --methods UC BPR-MF LightGCN \
  --use_uc_reranking --rerank_top_n 100
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | ML-1M | ML-100K, ML-1M, ML-20M, Douban_monti, Netflix |
| `--methods` | UC BPR-MF LightGCN | Methods to evaluate |
| `--k_values` | 5 10 20 | K values for top-k metrics |
| `--random_state` | 42 | Seed for reproducibility |
| `--eval_mode` | rerank | rerank, full_ranking, negative_sampling |
| `--rating_threshold` | 4.0 | Minimum rating as positive feedback |
| `--compute_uc_metrics` | off | Compute CVR and Kendall-tau |
| `--use_uc_reranking` | off | Evaluate UC as re-ranking module |
| `--config_dir` | configs/recbole | Config directory for RecBole models |

**Output:** Results are saved in `results/{Dataset}_standard_ranking_results/summary_seed_{seed}_{methods}.csv`.

---

## 2. Aggregate Results

`graph_results.py` — Aggregates CSV results across seeds, produces mean/std summary tables and training time plots.

```bash
# Single dataset
python graph_results.py --dataset ML-1M

# Filter to specific seeds
python graph_results.py --dataset ML-1M --seeds 0 42

# Multiple datasets (produces combined plots)
python graph_results.py --dataset ML-1M ML-100K Douban_monti

# Custom baseline for speedup plot
python graph_results.py --dataset ML-1M --baseline LightGCN
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | ML-1M | One or more dataset names |
| `--seeds` | all found | Filter to specific seeds |
| `--results_dir` | results | Root results directory |
| `--baseline` | UC | Baseline method for speedup normalization |

**Output:**
- `results/{Dataset}_standard_ranking_results/mean_std_summary.csv` — Mean +/- std table
- `results/{Dataset}_standard_ranking_results/training_time_log.png` — Log-scale training time plot
- `results/{Dataset}_standard_ranking_results/training_time_speedup.png` — Relative speedup plot
- If multiple datasets: `results/training_time_log_combined.png` and `results/training_time_speedup_combined.png`

---

## 3. Convert to LaTeX

`convert_latex.py` — Converts the `mean_std_summary.csv` from step 2 into publication-ready LaTeX tables with bold best values.

```bash
# Single dataset
python convert_latex.py --dataset ML-1M

# Multiple datasets
python convert_latex.py --dataset ML-1M ML-100K Douban_monti

# Save all tables to one file
python convert_latex.py --dataset ML-1M ML-100K --output tables.tex
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | ML-1M | One or more dataset names |
| `--results_dir` | results | Root results directory |
| `--output` | stdout + per-dataset | Optional single output .tex file |

**Output:** `results/{Dataset}_standard_ranking_results/results_table.tex`

---

## 4. Hyperparameter Tuning

`run_hyperparam_tuning.py` — Grid search over hyperparameters for LightGCN, SimGCL, and FairGAN with reduced epochs. Uses constrained grids for large datasets (ML-20M, Netflix).

```bash
# Tune LightGCN on Netflix
python run_hyperparam_tuning.py --dataset Netflix --methods LightGCN

# Tune all methods on ML-1M
python run_hyperparam_tuning.py --dataset ML-1M --methods LightGCN SimGCL fairGAN_tf --tune_epochs 15

# Custom epoch budget
python run_hyperparam_tuning.py --dataset Netflix --methods SimGCL --tune_epochs 15
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | Dataset name |
| `--methods` | required | LightGCN, SimGCL, fairGAN_tf |
| `--tune_epochs` | 10 | Max epochs per trial |
| `--data_path` | data | Path to data directory |
| `--batch_size` | auto | Override batch size for RecBole models |
| `--output_dir` | tuning_results | Directory to save result CSVs |

**Output:** `tuning_results/{Method}_{Dataset}_tuning.csv` — Sorted by best metric (Recall@10 for RecBole models, Coverage for FairGAN). Update the YAML configs in `configs/recbole/{Dataset}/` with the best values.

---

## Typical Workflow

```bash
# 1. Run experiments with multiple seeds
for seed in 0 42 123; do
  python run_standard_ranking_eval.py --dataset ML-1M \
    --methods UC BPR-MF LightGCN SimGCL fairGAN_tf \
    --compute_uc_metrics --random_state $seed
done

# 2. Aggregate results
python graph_results.py --dataset ML-1M

# 3. Generate LaTeX table
python convert_latex.py --dataset ML-1M
```

## Requirements

- Python 3.8+
- PyTorch
- RecBole + RecBole-GNN (for LightGCN, SimGCL, NGCF, etc.)
- TensorFlow (for FairGAN)
- NumPy, SciPy, Pandas, Matplotlib

```bash
pip install recbole recbole-gnn torch tensorflow numpy scipy pandas matplotlib pyyaml
```

## Troubleshooting

**Out of Memory (GPU):** Reduce `embedding_size`, `n_layers`, or increase `batch_size` in the dataset-specific YAML config. Netflix configs are already tuned for 12GB GPUs.

**FairGAN crash at epoch 70-80 (Netflix):** TensorFlow accumulates system RAM over many epochs. The Netflix FairGAN config uses 40 epochs to avoid this. If it still crashes, reduce `epochs` further in `configs/recbole/Netflix/fairgan_local.yaml`.

**RecBole import errors:**
```bash
pip install recbole recbole-gnn
```
