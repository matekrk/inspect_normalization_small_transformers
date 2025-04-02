import os
import matplotlib.pyplot as plt
import torch
from normalization import *

rez_file_paths = [
    "normalization_results_with_dyt.pt",
    "normalization_results_with_dyt_0.pt",
    "normalization_results.pt",
]
rez_file_paths_adamw = [
    "normalization_results_with_dyt_8_adamw.pt",
    "normalization_results_with_dyt_0_adamw.pt",
    "normalization_results_with_dyt_42_adamw.pt",
]
"""
{
    'model': model,
    'train_losses': train_losses,
    'train_accs': train_accs,
    'val_losses': val_losses,
    'val_accs': val_accs
}
"""

def aggregate(rez_file_paths):
    """
    Aggregate the results from the given file paths.
    """
    aggregated_results = {}
    relevant_models = ['dyt', 'layer', 'batch', 'rms', 'none']
    relevant_metrics = ['train_losses', 'train_accs', 'val_losses', 'val_accs']
    
    for file_path in rez_file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                results = torch.load(f, map_location=torch.device('cpu'))
                print(results.keys())
                for model in relevant_models:
                    if model not in results.keys():
                        continue
                    if model not in aggregated_results:
                        aggregated_results[model] = {}
                    for metric in relevant_metrics:
                        if metric not in aggregated_results[model]:
                            aggregated_results[model][metric] = []
                        aggregated_results[model][metric].append(results[model][metric])

    # Average the metrics across all runs
    averaged_results = {}
    for model in relevant_models:
        if model not in averaged_results:
            averaged_results[model] = {}
        for metric in relevant_metrics:
            if metric not in averaged_results[model]:
                averaged_results[model][metric] = {}
            data = torch.stack([torch.tensor(x) for x in aggregated_results[model][metric]])
            averaged_results[model][metric]["mean"] = torch.mean(data, dim=0)
            averaged_results[model][metric]["std"] = torch.std(data, dim=0)

    print(averaged_results)

    # Set up the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(averaged_results[relevant_models[-1]]["train_losses"]["mean"]) + 1)
    
    # Training loss
    ax = axes[0, 0]
    for norm_type in relevant_models:
        ax.plot(epochs, averaged_results[norm_type]['train_losses']["mean"], label=norm_type)
        ax.fill_between(epochs, 
                        averaged_results[norm_type]['train_losses']["mean"] - averaged_results[norm_type]['train_losses']["std"],
                        averaged_results[norm_type]['train_losses']["mean"] + averaged_results[norm_type]['train_losses']["std"], alpha=0.2)
        # ax.errorbar(epochs, averaged_results[norm_type]['train_losses']["mean"],
        #             yerr=averaged_results[norm_type]['train_losses']["std"], fmt='o', markersize=2, capsize=3)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Validation loss
    ax = axes[0, 1]
    for norm_type in relevant_models:
        ax.plot(epochs, averaged_results[norm_type]['val_losses']["mean"], label=norm_type)
        ax.fill_between(epochs, 
                        averaged_results[norm_type]['val_losses']["mean"] - averaged_results[norm_type]['val_losses']["std"],
                        averaged_results[norm_type]['val_losses']["mean"] + averaged_results[norm_type]['val_losses']["std"], alpha=0.2)
        # ax.errorbar(epochs, averaged_results[norm_type]['val_losses']["mean"],
        #             yerr=averaged_results[norm_type]['val_losses']["std"], fmt='o', markersize=2, capsize=3)
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Training accuracy
    ax = axes[1, 0]
    for norm_type in relevant_models:
        ax.plot(epochs, averaged_results[norm_type]['train_accs']["mean"], label=norm_type)
        ax.fill_between(epochs, 
                        averaged_results[norm_type]['train_accs']["mean"] - averaged_results[norm_type]['train_accs']["std"],
                        averaged_results[norm_type]['train_accs']["mean"] + averaged_results[norm_type]['train_accs']["std"], alpha=0.2)
        # ax.errorbar(epochs, averaged_results[norm_type]['train_accs']["mean"],
        #             yerr=averaged_results[norm_type]['train_accs']["std"], fmt='o', markersize=2, capsize=3)
    ax.set_title('Training Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # Validation accuracy
    ax = axes[1, 1]
    for norm_type in relevant_models:
        ax.plot(epochs, averaged_results[norm_type]['val_accs']["mean"], label=norm_type)
        ax.fill_between(epochs, 
                        averaged_results[norm_type]['val_accs']["mean"] - averaged_results[norm_type]['val_accs']["std"],
                        averaged_results[norm_type]['val_accs']["mean"] + averaged_results[norm_type]['val_accs']["std"], alpha=0.2)
        # ax.errorbar(epochs, averaged_results[norm_type]['val_accs']["mean"],
        #             yerr=averaged_results[norm_type]['val_accs']["std"], fmt='o', markersize=2, capsize=3)
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('normalization_comparison_agg_adamw.png')
    plt.show()

if __name__ == "__main__":
    aggregate(rez_file_paths_adamw)
    print("Aggregated results have been plotted and saved as 'normalization_comparison_agg_SUFFIX.png'.")
