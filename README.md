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

