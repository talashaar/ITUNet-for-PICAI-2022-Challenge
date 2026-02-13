import matplotlib.pyplot as plt
import numpy as np

# Data for 4-folds 
sequences = ['Full\nResampled', 'ADC\nZero', 'HBV\nZero', 'T2W\nZero', 'ADC\nAvg', 'HBV\nAvg', 'T2W\nAvg']
auroc_values = [78.15, 81.12, 80.72, 45.42, 50.36, 64.79, 54.99]  
ap_values = [34.89, 46.32, 29.34, 22.1, 8.64, 9.36, 6.37]       

# Random guess results (from your experiment)
random_auroc = 44.01  # 44.01%
random_ap = 0.00      # 0.00% (or -0.00% as shown in your output)

fig, ax = plt.subplots(figsize=(14, 8))

# X positions for the sequences
x = np.arange(len(sequences))
bar_width = 0.35  # Width of each bar

# Plot AUROC as bars
auroc_bars = ax.bar(x - bar_width/2, auroc_values, bar_width, 
                    color='#2E86AB', edgecolor='white', linewidth=1.5,
                    label='AUROC', alpha=0.9)

# Plot AP as bars
ap_bars = ax.bar(x + bar_width/2, ap_values, bar_width,
                 color='#A23B72', edgecolor='white', linewidth=1.5,
                 label='AP', alpha=0.9)

# Add horizontal lines for random guess
# AUROC random guess line (using a dashed line in orange/red)
ax.axhline(y=random_auroc, color='#FF6B35', linestyle='--', linewidth=2.5, 
           label=f'Random Guess AUROC ({random_auroc}%)', alpha=0.8)

# AP random guess line (using a dotted line in dark green)
ax.axhline(y=random_ap, color='#2A9D8F', linestyle=':', linewidth=2.5,
           label=f'Random Guess AP ({random_ap}%)', alpha=0.8)

# Customize the plot
ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
ax.set_title('4-Fold Cross Validation: AUROC and Average Precision', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sequences)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')  # Only show horizontal grid lines

# Set y-axis from 0% to 100% to show full scale
ax.set_ylim(0, 100)

# Add value labels on top of the bars
for i, (auroc_val, ap_val) in enumerate(zip(auroc_values, ap_values)):
    # AUROC values
    ax.annotate(f'{auroc_val}%', 
                xy=(x[i] - bar_width/2, auroc_val),
                xytext=(0, 3),  # Offset from bar top
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#2E86AB')
    
    # AP values
    ax.annotate(f'{ap_val}%', 
                xy=(x[i] + bar_width/2, ap_val),
                xytext=(0, 3),  # Offset from bar top
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#A23B72')

# Add text annotation for random guess lines (optional)
ax.annotate(f'Random AUROC: {random_auroc}%', 
            xy=(len(sequences)-0.5, random_auroc),
            xytext=(10, 0),  # Move text to the right of the line end
            textcoords="offset points",
            ha='left', va='center', fontsize=10, fontweight='bold',
            color='#FF6B35', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.annotate(f'Random AP: {random_ap}%', 
            xy=(len(sequences)-0.5, random_ap),
            xytext=(10, 0),  # Move text to the right of the line end
            textcoords="offset points",
            ha='left', va='center', fontsize=10, fontweight='bold',
            color='#2A9D8F', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save the plot to files with different name
plt.savefig('4folds_results_bar_with_random.png', dpi=300, bbox_inches='tight')
plt.savefig('4folds_results_bar_with_random.pdf', bbox_inches='tight')

print("4-Folds plots with random guess lines saved successfully:")
print("- 4folds_results_bar_with_random.png")
print("- 4folds_results_bar_with_random.pdf")
plt.show()