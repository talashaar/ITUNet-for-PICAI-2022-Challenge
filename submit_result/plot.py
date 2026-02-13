import matplotlib.pyplot as plt
import numpy as np

# Data with all sequences
sequences = ['Resampled\n(Full)', 'ADC\nZero', 'HBV\nZero', 'T2W\nZero', 'ADC\nAvg', 'HBV\nAvg' , 'T2W\nAvg']
auroc_values = [87.1, 67.8, 76.2, 50.3, 67.0,54.9, 66.6]  # AUROC values in order
ap_values = [63.2, 35.2, 41.9, 14.6, 21.5,6.3, 15.3]      # AP values in order

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

# Customize the plot
ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
ax.set_title('Model Performance: AUROC and Average Precision Comparison', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sequences)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')  # Only show horizontal grid lines

# Set y-axis from 0% to 100% to show full scale
ax.set_ylim(0, 100)

# Add value labels on top of the bars
for i, (auroc_bar, ap_bar) in enumerate(zip(auroc_bars, ap_bars)):
    # AUROC values
    height_auroc = auroc_bar.get_height()
    ax.annotate(f'{auroc_values[i]}%', 
                xy=(auroc_bar.get_x() + auroc_bar.get_width()/2, height_auroc),
                xytext=(0, 3),  # Offset from bar top
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#2E86AB')
    
    # AP values
    height_ap = ap_bar.get_height()
    ax.annotate(f'{ap_values[i]}%', 
                xy=(ap_bar.get_x() + ap_bar.get_width()/2, height_ap),
                xytext=(0, 3),  # Offset from bar top
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#A23B72')

plt.tight_layout()

# Save the plot to files
plt.savefig('thesis_results_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('thesis_results_plot.pdf', bbox_inches='tight')

print("Plots saved successfully:")
print("- thesis_results_bar.png")
print("- thesis_results_bar.pdf")