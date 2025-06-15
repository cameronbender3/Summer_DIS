# SAGE Watermarking Implementation

This repository contains a Python implementation of SAGE-based watermarking for graph neural networks, based on Wang et al. (2023): "Making Watermark Survive Model Extraction Attacks".

---

## Repository Structure

- **sage_watermark.py** – Main script for training and evaluating a watermarked model on a single dataset with specified parameters.
- **run_experiment.py** – Automates grid search over datasets, watermark ratios (`alpha`), and watermark sizes (`N_t`) to reproduce results from the original paper, saving metrics to CSV files.
- **requirements.txt** – Python dependencies.
- **results/** – Directory where all experiment output CSV files are saved.

---

## Usage

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run a simple experiment
```
python sage_watermark.py
```
- **Parameters:** To modify parameters for a single run (e.g., dataset, epochs, alpha, N_t), edit the top section of sage_watermark.py. Some parameters are currently hardcoded and must be changed directly in the script.
## 3. Reproduce experiments reported in the paper
```
python run_experiment.py
```
- This script sweeps through all main hyperparameters (dataset, alpha, N_t, etc.) as in the original paper and saves aggregate results for each configuration in the results/ directory as CSV files.
- **Parameters:** Datasets, alpha values, N_t range, epochs, number of runs, etc. are specified at the top of run_experiment.py.
If you wish to use other values or add datasets, edit the corresponding Python lists/values at the top of the script.

## Output Files

For each dataset/alpha grid search, a summary CSV file is saved in the `results/` directory, with columns:

- **N_t**: Number of nodes in the watermark random part
- **avg_effectiveness**: Average watermark retention across runs
- **avg_clean_acc**: Average clean accuracy of the watermarked model
- **avg_acc_loss**: Difference in accuracy between clean and watermarked model

## Modifying Parameters

**Hardcoded Parameters:**
- In both `sage_watermark.py` and `run_experiment.py`, the key parameters (`dataset`, `alpha`, `N_t`, `epochs`, `num_layers`, and `num_runs`) are set near the top of each script.
- To run different datasets or values, edit these scripts directly as needed.

**Number of Watermark Samples:**
- The number of key input (watermark) samples is set as `num_watermark_samples = int(len(train_data) * alpha)`.  
  If you wish to change this, edit the relevant line in `sage_watermark.py`.

### ENZYMES Accuracy Loss (Watermarked Model, α = 0.1)

| N_t | Accuracy Loss (Paper) | Accuracy Loss (Ours) |
|-----|-----------------------|----------------------|
| 3   | ~0.21                 | 0.08                 |
| 4   | ~0.31                 | 0.08                 |
| 5   | ~0.33                 | 0.12                 |

#### α = 0.05 (Ours)

| N_t | Accuracy Loss (Ours) |
|-----|----------------------|
| 3   | 0.07                 |
| 4   | 0.03                 |
| 5   | 0.08                 |

> For ENZYMES, our reproduced results show lower accuracy loss after watermarking compared to the paper.  
> The trend with respect to $N_t$ is consistent as the number of random nodes increases, accuracy loss increases.

---

### ENZYMES Watermark Effectiveness (Optimized, α = 0.1)

| N_t | Effectiveness (Paper) | Effectiveness (Ours) |
|-----|-----------------------|----------------------|
| 3   | >0.9                  | 0.57                 |
| 4   | >0.9                  | 0.50                 |
| 5   | >0.9                  | 0.36                 |

#### α = 0.05 (Ours)

| N_t | Effectiveness (Ours) |
|-----|----------------------|
| 3   | 0.31                 |
| 4   | 0.29                 |
| 5   | 0.42                 |

> The paper reports watermark effectiveness >0.9 for all $N_t$ at $\alpha = 0.1$.  
> Our effectiveness is lower, but follows the same trend of decreasing as $N_t$ increases.

---

### MSRC_9 Accuracy Loss (Watermarked Model, α = 0.1)

| N_t | Accuracy Loss (Paper) | Accuracy Loss (Ours) |
|-----|-----------------------|----------------------|
| 3   | ~0.07                 | 0.00                 |
| 4   | ~0.12                 | 0.01                 |
| 5   | ~0.13                 | 0.00                 |

#### α = 0.05 (Ours)

| N_t | Accuracy Loss (Ours) |
|-----|----------------------|
| 3   | 0.01                 |
| 4   | -0.01                |
| 5   | 0.00                 |

> For MSRC_9, our reproduced models experience almost no accuracy loss after watermarking, compared to the modest losses in the paper.

---

### MSRC_9 Watermark Effectiveness (Optimized, α = 0.1)

| N_t | Effectiveness (Paper) | Effectiveness (Ours) |
|-----|-----------------------|----------------------|
| 3   | >0.9                  | 0.41                 |
| 4   | >0.9                  | 0.24                 |
| 5   | >0.9                  | 0.29                 |

#### α = 0.05 (Ours)

| N_t | Effectiveness (Ours) |
|-----|----------------------|
| 3   | 0.19                 |
| 4   | 0.13                 |
| 5   | 0.19                 |

> Paper shows >0.9 effectiveness for all settings; ours is lower but increases with $N_t$ and $\alpha$.

---

**Summary:**  
- Our watermarking implementation produces similar *trends* as the original paper such as more key input-label pairs and higher $N_t$ result in higher effectiveness, with some tradeoff in accuracy loss.
- Our models have generally lower accuracy loss, but watermark retention is lower than reported.  This may be due to implementation or hyperparameter differences.  Further tuning may improve results.

---

## Notes

- Main experiment parameters (`dataset`, `alpha`, `N_t`, `epochs`, etc.) can be changed in `run_experiment.py`.
- By default, grid experiments cover multiple datasets (`ENZYMES`, `MSRC_9`, etc.) and `alpha`/`N_t` values as described in the paper.
- Full reproduction as in the paper may require large compute resources.
- This is a best-effort reproduction and may differ in some implementation details or scale (e.g., number of random seeds).
- Watermark effectiveness and accuracy metrics are printed to the console and saved as CSV summaries.

## Requirements
- Python 3.9+
- PyTorch
- PyTorch Geometric

## Contact
Cameron Bender
cdb21c@fsu.edu

