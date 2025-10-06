import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# File paths - 3.a is Altered (Surprise GRU), 3.b is Normal (Regular GRU)
alter_gru_files = [
    r'C:\Users\billy\Desktop\Paper1_Final\analyses\3\3.a\PPOSurpriseGRUPolicy_popgym-RepeatFirstEasy-v0_trial1_2025-09-13_09-57-23.csv'
]

normal_gru_files = [
    r'C:\Users\billy\Desktop\Paper1_Final\analyses\3\3.b\PPOGRUPolicy_popgym-RepeatFirstEasy-v0_trial1_2025-09-13_09-57-41.csv'
]

def load_and_aggregate(file_list):
    """Load CSV files and compute mean and 95% CI, handling different lengths"""
    dfs = [pd.read_csv(f) for f in file_list]

    # Find maximum length to align all data
    max_length = max(len(df) for df in dfs)

    # Get all unique epochs across all files
    all_epochs = sorted(set().union(*[set(df['epoch'].values) for df in dfs]))

    # Create aligned dataframes with NaN for missing epochs
    aligned_data = []
    for df in dfs:
        # Create a series indexed by epoch
        series = df.set_index('epoch')['best_mmer']
        # Reindex to include all epochs, filling missing with NaN
        aligned = series.reindex(all_epochs)
        aligned_data.append(aligned.values)

    # Stack data (now all same length, with NaNs where data is missing)
    mmer_values = np.array(aligned_data)

    # Calculate mean and 95% CI, ignoring NaNs
    mean_mmer = np.nanmean(mmer_values, axis=0)

    # Count non-NaN values at each position
    n_samples = np.sum(~np.isnan(mmer_values), axis=0)

    # Only compute CI where we have all 3 seeds
    ci_95 = np.full_like(mean_mmer, np.nan)
    mask_all_seeds = (n_samples == len(file_list))

    if np.any(mask_all_seeds):
        sem = stats.sem(mmer_values[:, mask_all_seeds], axis=0)
        ci_95[mask_all_seeds] = sem * stats.t.ppf(0.975, len(file_list) - 1)

    return np.array(all_epochs), mean_mmer, ci_95, mask_all_seeds

# Load data for both models
episodes_normal, mean_normal, ci_normal, mask_normal = load_and_aggregate(normal_gru_files)
episodes_alter, mean_alter, ci_alter, mask_alter = load_and_aggregate(alter_gru_files)

# Create plot
plt.figure(figsize=(10, 6))

# Plot Normal GRU
plt.plot(episodes_normal, mean_normal, color='blue', label='Normal GRU', linewidth=2)
plt.fill_between(episodes_normal[mask_normal],
                 mean_normal[mask_normal] - ci_normal[mask_normal],
                 mean_normal[mask_normal] + ci_normal[mask_normal],
                 color='blue', alpha=0.2)

# Plot Alter GRU
plt.plot(episodes_alter, mean_alter, color='orange', label='Alter GRU', linewidth=2)
plt.fill_between(episodes_alter[mask_alter],
                 mean_alter[mask_alter] - ci_alter[mask_alter],
                 mean_alter[mask_alter] + ci_alter[mask_alter],
                 color='orange', alpha=0.2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Best MMER', fontsize=12)
plt.title('MMER Comparison: Normal GRU vs Alter GRU', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure
plt.savefig('mmer_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'mmer_comparison.png'")

plt.show()
