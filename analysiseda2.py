import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Base directory for the dataset files
base_dir = "eda2/"  # adjust as needed

# Define groups and their CSV files
groups = {
    "PPOGRUPolicy": [
        base_dir + "PPOGRUPolicy_popgym-CountRecallEasy-v0_trial4_2025-06-26_12-34-05.csv",
        base_dir + "PPOGRUPolicy_popgym-CountRecallEasy-v0_trial5_2025-06-26_18-49-20.csv",
    ],
    "PPOSurpriseGRUPolicy": [
        base_dir + "PPOSurpriseGRUPolicy_popgym-CountRecallEasy-v0_trial1_2025-06-24_22-11-39.csv",
        base_dir + "PPOSurpriseGRUPolicy_popgym-CountRecallEasy-v0_trial2_2025-06-26_00-28-12.csv",
    ],
}

plt.figure(figsize=(14, 8))

group_colors = {'PPOGRUPolicy': 'tab:green', 'PPOSurpriseGRUPolicy': 'tab:red'}

for group_name, files in groups.items():
    num_trials = len(files)
    print(f"\nProcessing group: {group_name} over {num_trials} trial{'s' if num_trials > 1 else ''}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, skiprows=1)  # skip metadata line
        df.columns = df.columns.str.strip().str.lower()
        dfs.append(df)

    # Assign trial numbers (not used in plotting individual trials anymore)
    for i, df in enumerate(dfs):
        df['trial'] = i + 1

    # Concatenate all trials for group statistics
    combined_df = pd.concat(dfs, ignore_index=True)

    stats = combined_df.groupby('epoch')['best_mmer'].agg(['mean', 'std', 'count']).reset_index()

    max_best_mmer = stats['mean'].max()
    std_at_max = stats.loc[stats['mean'].idxmax(), 'std']
    threshold = 0.95 * max_best_mmer

    qualifying_epochs = stats[stats['mean'] >= threshold]
    first_epoch_95 = qualifying_epochs['epoch'].min() if not qualifying_epochs.empty else None

    print(f"Best (maximum) mean best_mmer: {max_best_mmer:.6f} ± {std_at_max:.6f}")
    print(f"First epoch reaching at least 95% of best (>= {threshold:.6f}): {first_epoch_95}")

    # Calculate 95% confidence interval
    stats['ci95'] = 1.96 * stats['std'] / np.sqrt(stats['count'])

    # Clip confidence intervals to metric bounds [-1, 1]
    lower_bound = np.maximum(stats['mean'] - stats['ci95'], -1.0)
    upper_bound = np.minimum(stats['mean'] + stats['ci95'], 1.0)

    # Plot mean with clipped 95% confidence interval shading only
    plt.plot(stats['epoch'], stats['mean'], color=group_colors[group_name], linewidth=2, label=f"{group_name} Mean")
    plt.fill_between(stats['epoch'],
                     lower_bound,
                     upper_bound,
                     color=group_colors[group_name], alpha=0.2)

plt.xlabel("Epoch")
plt.ylabel("Best MMER")
plt.title("Best MMER over Epochs with 95% Confidence Interval")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
