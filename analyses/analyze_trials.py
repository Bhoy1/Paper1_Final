import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# Load all trial files for Alter GRU
alter_files = sorted(glob(r'C:\Users\billy\Desktop\Paper1_Final\analyses\7\7.a\*.csv'))

print("Analyzing Alter GRU (Surprise GRU) trials:")
print("=" * 60)

# Plot individual trials
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

colors = ['red', 'green', 'purple']
for i, file in enumerate(alter_files):
    df = pd.read_csv(file)
    trial_name = file.split('\\')[-1].split('_')[2]  # Extract trial number

    # Plot on first subplot
    ax1.plot(df['epoch'], df['best_mmer'], label=f'Trial {i+1}', color=colors[i], alpha=0.7)

    # Print statistics
    print(f"\nTrial {i+1} ({trial_name}):")
    print(f"  File: {file.split('\\')[-1]}")
    print(f"  Final best_mmer: {df['best_mmer'].iloc[-1]:.4f}")
    print(f"  Max best_mmer: {df['best_mmer'].max():.4f}")
    print(f"  Min best_mmer: {df['best_mmer'].min():.4f}")
    print(f"  Mean best_mmer: {df['best_mmer'].mean():.4f}")
    print(f"  Std best_mmer: {df['best_mmer'].std():.4f}")

    # Check specific problematic region (epochs 500-3500)
    problematic_region = df[(df['epoch'] >= 500) & (df['epoch'] <= 3500)]
    if len(problematic_region) > 0:
        print(f"  Epochs 500-3500 mean: {problematic_region['best_mmer'].mean():.4f}")
        print(f"  Epochs 500-3500 std: {problematic_region['best_mmer'].std():.4f}")

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Best MMER', fontsize=12)
ax1.set_title('Individual Alter GRU Trials', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Zoom into problematic region
for i, file in enumerate(alter_files):
    df = pd.read_csv(file)
    df_zoom = df[(df['epoch'] >= 500) & (df['epoch'] <= 3500)]
    ax2.plot(df_zoom['epoch'], df_zoom['best_mmer'], label=f'Trial {i+1}', color=colors[i], alpha=0.7)

ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Best MMER', fontsize=12)
ax2.set_title('Zoomed: Epochs 500-3500 (High Variance Region)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('trial_analysis.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 60)
print("Plot saved as 'trial_analysis.png'")

# Compare variance across trials at specific epochs
print("\n" + "=" * 60)
print("Variance analysis at key epochs:")
print("=" * 60)

test_epochs = [500, 1000, 2000, 3000, 4000, 5000, 7500]
for epoch in test_epochs:
    values = []
    for file in alter_files:
        df = pd.read_csv(file)
        closest_epoch = df.iloc[(df['epoch'] - epoch).abs().argsort()[:1]]
        values.append(closest_epoch['best_mmer'].values[0])

    print(f"\nEpoch ~{epoch}:")
    print(f"  Values: {[f'{v:.4f}' for v in values]}")
    print(f"  Mean: {sum(values)/len(values):.4f}")
    print(f"  Std: {pd.Series(values).std():.4f}")
    print(f"  Range: {max(values) - min(values):.4f}")

plt.show()
