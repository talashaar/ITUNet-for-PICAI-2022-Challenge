import matplotlib.pyplot as plt
import numpy as np

# Data for 4-folds (with missing values marked as None)
sequences = ['Full\nResampled', 'ADC\nZero', 'HBV\nZero', 'T2W\nZero', 'ADC\nAvg', 'HBV\nAvg', 'T2W\nAvg']
auroc_values = [78.15, 81.12, 80.72, None, 50.36, 64.79, 54.99]  # None for missing values
ap_values = [34.89, 46.32, 29.34, None, 8.64, 9.36, 6.37]        # None for missing values

fig, ax = plt.subplots(figsize=(14, 8))

# X positions for the sequences
x = np.arange(len(sequences))

# Plot AUROC as circles with connecting line (only for available data)
available_auroc_x = [i for i, val in enumerate(auroc_values) if val is not None]
available_auroc_y = [val for val in auroc_values if val is not None]

# Plot AP as squares with connecting line (only for available data)
available_ap_x = [i for i, val in enumerate(ap_values) if val is not None]
available_ap_y = [val for val in ap_values if val is not None]

# Plot AUROC as circles with connecting line
auroc_line = ax.plot(available_auroc_x, available_auroc_y, 'o-', color='#2E86AB', linewidth=2.5, 
                     markersize=10, label='AUROC', markerfacecolor='#2E86AB', 
                     markeredgecolor='white', markeredgewidth=1.5)

# Plot AP as squares with connecting line
ap_line = ax.plot(available_ap_x, available_ap_y, 's-', color='#A23B72', linewidth=2.5, 
                  markersize=9, label='AP', markerfacecolor='#A23B72', 
                  markeredgecolor='white', markeredgewidth=1.5)

# Customize the plot
ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
ax.set_title('4-Fold Cross Validation: AUROC and Average Precision', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sequences)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Set y-axis from 0% to 100% to show full scale
ax.set_ylim(0, 100)

# Add value labels on the points (only for available data)
for i, (auroc_val, ap_val) in enumerate(zip(auroc_values, ap_values)):
    if auroc_val is not None:
        # AUROC values (circles)
        ax.annotate(f'{auroc_val}%', (x[i], auroc_val), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color='#2E86AB')
    
    if ap_val is not None:
        # AP values (squares) 
        ax.annotate(f'{ap_val}%', (x[i], ap_val), 
                    textcoords="offset points", xytext=(0,-15), 
                    ha='center', va='top', fontsize=9, fontweight='bold',
                    color='#A23B72')

# Add text to indicate missing data
missing_text = "Missing: HBV Zero, T2W Zero results"
ax.text(0.02, 0.02, missing_text, transform=ax.transAxes, fontsize=10, 
        style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()

# Save the plot to files with different name
plt.savefig('4folds_results.png', dpi=300, bbox_inches='tight')
plt.savefig('4folds_results.pdf', bbox_inches='tight')

print("4-Folds plots saved successfully:")
print("- 4folds_results.png")
print("- 4folds_results.pdf")
print(f"Missing results: T2W Zero")