import matplotlib.pyplot as plt
import numpy as np

# Data with all sequences
sequences = ['Resampled\n(Full)', 'ADC\nZero', 'HBV\nZero', 'T2W\nZero', 'ADC\nAvg', 'T2W\nAvg', 'HBV\nAvg']
auroc_values = [87.1, 67.8, 76.2, 50.3, 67.0, 66.6, 54.9]  # AUROC values in order
ap_values = [63.2, 35.2, 41.9, 14.6, 21.5, 15.3, 6.3]      # AP values in order

fig, ax = plt.subplots(figsize=(14, 8))

# X positions for the sequences
x = np.arange(len(sequences))

# Plot AUROC as circles with connecting line
auroc_line = ax.plot(x, auroc_values, 'o-', color='#2E86AB', linewidth=2.5, 
                     markersize=10, label='AUROC', markerfacecolor='#2E86AB', 
                     markeredgecolor='white', markeredgewidth=1.5)

# Plot AP as squares with connecting line
ap_line = ax.plot(x, ap_values, 's-', color='#A23B72', linewidth=2.5, 
                  markersize=9, label='AP', markerfacecolor='#A23B72', 
                  markeredgecolor='white', markeredgewidth=1.5)

# Customize the plot
ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
ax.set_title('Model Performance: AUROC and Average Precision Comparison', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sequences)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Set y-axis from 0% to 100% to show full scale
ax.set_ylim(0, 100)

# Add value labels on the points
for i, (auroc_val, ap_val) in enumerate(zip(auroc_values, ap_values)):
    # AUROC values (circles)
    ax.annotate(f'{auroc_val}%', (x[i], auroc_val), 
                textcoords="offset points", xytext=(0,10), 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#2E86AB')
    
    # AP values (squares) 
    ax.annotate(f'{ap_val}%', (x[i], ap_val), 
                textcoords="offset points", xytext=(0,-15), 
                ha='center', va='top', fontsize=9, fontweight='bold',
                color='#A23B72')

plt.tight_layout()

# Save the plot to files
plt.savefig('thesis_results.png', dpi=300, bbox_inches='tight')
plt.savefig('thesis_results.pdf', bbox_inches='tight')

print("Plots saved successfully:")
print("- thesis_results.png")
print("- thesis_results.pdf")