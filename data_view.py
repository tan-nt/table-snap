import matplotlib.pyplot as plt
import numpy as np

# Dataset info
datasets = ['SciTSR', 'PubTables-1M']
width_avg = [859, 1032]
width_max = [1283, 4485]
height_avg = [363, 875]
height_max = [2052, 4561]
total_images = [15000, 575305]  # Updated for PubTables-1M

x = np.arange(len(datasets))
bar_width = 0.35

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. Image Widths
axs[0].bar(x - bar_width/2, width_avg, bar_width, label='Avg Width', color='orange')
axs[0].bar(x + bar_width/2, width_max, bar_width, label='Max Width', color='orangered')
axs[0].set_title('Image Widths')
axs[0].set_xticks(x)
axs[0].set_xticklabels(datasets)
axs[0].set_ylabel('Pixels')
axs[0].legend()

# 2. Image Heights
axs[1].bar(x - bar_width/2, height_avg, bar_width, label='Avg Height', color='gold')
axs[1].bar(x + bar_width/2, height_max, bar_width, label='Max Height', color='darkorange')
axs[1].set_title('Image Heights')
axs[1].set_xticks(x)
axs[1].set_xticklabels(datasets)
axs[1].set_ylabel('Pixels')
axs[1].legend()

# 3. Total Images (log scale)
axs[2].bar(x, total_images, color='skyblue')
axs[2].set_title('Total Images')
axs[2].set_xticks(x)
axs[2].set_xticklabels(datasets)
axs[2].set_ylabel('Count')
axs[2].set_yscale('log')  # log scale for better visualization
axs[2].bar_label(axs[2].containers[0], fmt='%d')

# Super title and layout
plt.suptitle('Dataset Statistics: SciTSR vs PubTables-1M', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Show plot
plt.show()
