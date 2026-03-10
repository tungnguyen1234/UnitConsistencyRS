# Unit Consistency (UC) for Recommender Systems

Code for the paper **"Unit Consistency: A Paradigm Shift for Recommender Systems —
Rank-Preference Consistency as the Appropriate Metric for Recommender Systems"**.

This repository contains the UC implementation and evaluation scripts for three
experiments described in the paper.  Baseline methods were run using RecBole
[cite]; hyperparameter configurations are reported in Tables 15–18 of the paper.

---

## Download Data

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

---

## Supported Datasets

| Dataset | Users | Items | Notes |
|---------|-------|-------|-------|
| ML-100K | ~1 K | ~1.7 K | Small, fast |
| ML-1M | ~6 K | ~3.7 K | Medium |
| Douban_monti | ~3 K | ~3.7 K | Medium |
| ML-20M | ~138 K | ~27 K | Large |
| Netflix | ~440 K | ~9 K | Large, GPU memory constrained |

---

## Repository Layout

```
main.py                            # Unified experiment runner (all experiments)
plot_ranksvd_metric_divergence.py  # Reproduce RankSVD RMSE vs ranking figure
UCTC_algo.py                       # UC / TC dense algorithm
UCTC_sparse.py                     # UC / TC sparse algorithm (large datasets)
rankingSVD_algo.py                 # RankSVD baseline
utils/
  io.py                            # Logger, save_data, load_json
  dataloader.py                    # Dense matrix data loaders (Exp 8)
  experiment_utils.py              # UC runners + DataFrame data loaders (Exp 9)
  preprocessing.py                 # Train/test splitting + Exp-8 sampling helpers
  metric.py                        # Ranking metrics (P@k, R@k, NDCG@k), Kendall-tau
  lazy_candidates.py               # Memory-efficient candidate generation
  ranking_eval.py                  # Standard ranking evaluation pipeline (UC-only)
```

---

## Running UC Experiments

All experiments are launched through `main.py`:

```bash
python main.py --dataset DATASET --seed SEED --experiment EXP [--output_dir DIR] [--data_path PATH]
```

| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `--dataset` | `ML-1M` | ML-100K, ML-1M, Douban_monti, ML-20M, Netflix | Dataset |
| `--seed` | `42` | any int | Random seed |
| `--experiment` | `long_tail` | `strong_and_subtle`, `long_tail`, `ranking` | Experiment type |
| `--output_dir` | `results` | any path | Output directory |
| `--data_path` | *(paths.json or `data/`)* | any path | Root data directory |

### `strong_and_subtle` — All-Items Rank-Preference Consistency

UC evaluated on all items for strong (1 vs 5) and subtle (4 vs 5) preference pairs.

```bash
# Multiple seeds (S=10 small datasets, S=5 large datasets)
for seed in 0 42 123 456 789 1000 2000 3000 4000 5000; do
  python main.py --dataset ML-1M --seed $seed --experiment strong_and_subtle
done
```

### `long_tail` — Long-Tail Rank-Preference Consistency

UC evaluated on the least-frequently rated 67% of items, same preference pairs.

```bash
for seed in 0 42 123 456 789; do
  python main.py --dataset Netflix --seed $seed --experiment long_tail
done
```

### `ranking` — Standard Ranking Evaluation (P@k, R@k, NDCG@k)

Ratings ≥ 4.0 as positive, 80/20 per-user random split, full-ranking protocol.

```bash
python main.py --dataset ML-1M  --seed 42 --experiment ranking
python main.py --dataset ML-20M --seed 0  --experiment ranking
```

Results → `results/{dataset}_ranking_results/summary_seed_{seed}.csv`.

---

## Reproducing the RankSVD Figure

The figure showing RMSE, NDCG@10, and NDCG@20 for RankSVD across values of *k*
on ML-1M can be reproduced with:

```bash
python plot_ranksvd_metric_divergence.py --dataset ML-1M
```

Optional arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `ML-1M` | ML-100K or ML-1M |
| `--k_min` | `1` | Minimum k |
| `--k_max` | `100` | Maximum k |
| `--k_step` | `2` | Step between k values |
| `--data_path` | `data` | Root data directory |
| `--output_dir` | `results/experiment_1` | Output directory |

Output plots are saved to `results/experiment_1/`:
- `experiment_1_{dataset}_combined.png` — dual-axis RMSE vs NDCG (paper figure)
- `experiment_1_{dataset}_rmse_vs_k.png`
- `experiment_1_{dataset}_ndcg_vs_k.png`
- `experiment_1_{dataset}_sidebyside.png`

---

## Requirements

- Python 3.8+
- PyTorch (with optional CUDA)
- NumPy, SciPy, Pandas, Matplotlib, scikit-learn, h5py

```bash
pip install torch numpy scipy pandas matplotlib scikit-learn h5py
```

---

## Troubleshooting

**Out of GPU memory:** For ML-20M or Netflix, reduce batch sizes or switch to a
CPU run by setting `CUDA_VISIBLE_DEVICES=""`.

**`paths.json` not found:** Either create a `paths.json` file:
```json
{ "data_path": "/path/to/your/data" }
```
or pass `--data_path /path/to/your/data` directly on the command line.
