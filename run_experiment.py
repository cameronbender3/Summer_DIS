import torch
import numpy as np
import csv
import os
from sage_watermark import run_simple_experiment

def run_grid_experiments(
    results_dir='results',
    datasets=None,
    alphas=None,
    N_ts=None,
    clean_epochs=200,
    wm_epochs=200,
    num_layers=3,
    num_runs=5
):
    """
    Runs a grid of watermarking experiments over datasets, alphas, and N_t values.
    Aggregates and saves results as CSV summaries in the specified directory.

    Args:
        results_dir (str): Output directory for results.
        datasets (list): List of dataset names to evaluate.
        alphas (list): List of alpha values (fraction of watermarked samples).
        N_ts (list): List of N_t values (watermark trigger size).
        clean_epochs (int): Training epochs for the clean model.
        wm_epochs (int): Training epochs for the watermarked model.
        num_layers (int): Number of layers in the model.
        num_runs (int): Number of random seeds to average results.
    """
    if datasets is None:
        datasets = ['ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'PROTEINS', 'MSRC_9']
    if alphas is None:
        alphas = [0.05, 0.10, 0.15, 0.20]
    if N_ts is None:
        N_ts = [3, 4, 5, 6, 7]

    os.makedirs(results_dir, exist_ok=True)
    for dataset in datasets:
        for alpha in alphas:
            for N_t in N_ts:
                results = []
                for run in range(num_runs):
                    torch.manual_seed(run)
                    np.random.seed(run)
                    result = run_simple_experiment(
                        dataset_name=dataset,
                        N_t=N_t,
                        alpha=alpha,
                        num_layers=num_layers,
                        clean_epochs=clean_epochs,
                        wm_epochs=wm_epochs,
                    )
                    results.append(result)
                effs = [r['effectiveness'] for r in results if r is not None]
                clean_accs = [r['clean_accuracy'] for r in results if r is not None]
                baseline_accs = [r['baseline_accuracy'] for r in results if r is not None]
                avg_eff = np.mean(effs)
                avg_clean_acc = np.mean(clean_accs)
                avg_acc_loss = np.mean([base - clean for base, clean in zip(baseline_accs, clean_accs)])
                csv_path = os.path.join(results_dir, f"{dataset}_alpha{alpha}_summary.csv")
                write_header = not os.path.exists(csv_path)
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(['N_t', 'avg_effectiveness', 'avg_clean_acc', 'avg_acc_loss'])
                    writer.writerow([N_t, avg_eff, avg_clean_acc, avg_acc_loss])
    print(f"All experiment results saved to {results_dir}/")

if __name__ == '__main__':
    run_grid_experiments()
