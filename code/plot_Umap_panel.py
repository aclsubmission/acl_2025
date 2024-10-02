import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from LIWC import liwc
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

liwcDict = liwc().getLIWCCount('')
liwcHeaders = list(liwcDict.keys())

liwc_sorted = sorted(liwcHeaders)
sorted_liwc_idx = {cat: i for i, cat in enumerate(liwc_sorted)}

shuffle_order = {i: sorted_liwc_idx[cat] for i, cat in enumerate(liwcHeaders)}

ROOT = './style_rank/'
genres = ['articles_wikipedia', 'descriptions_airbnb', 'gutenberg_domesticfiction', 'review_yelp']
title_map = {'articles_wikipedia':'Articles', 'descriptions_airbnb':'Advertisements', 'gutenberg_domesticfiction':'Novels', 'review_yelp':'Business Review'}
cmap = LinearSegmentedColormap.from_list("custom_wb", ["white", "black"])

fig, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2x2 array to make indexing easier

for idx, genre in enumerate(genres):
    for data in os.listdir(ROOT):
        if data.startswith(genre) and data.endswith('.pickle_U_RANK.pickle'):
            with open(os.path.join(ROOT, data), 'rb') as handle:
                data_ = pickle.load(handle)
            
            len_map = {len(d[0]): i for i, d in enumerate(data_)}
            U_vec = data_[len_map[24]]
            
            if U_vec.ndim == 1:
                size = int(np.sqrt(U_vec.shape[0]))
                U_vec = U_vec.reshape(size, size)
            
            U_vec = np.abs(U_vec[1:])
            
            reorder_indices = [shuffle_order[i] for i in range(U_vec.shape[0])]
            U_vec_reordered = U_vec[reorder_indices]
            
            U_vec_normalized = (U_vec_reordered - U_vec_reordered.min()) / (U_vec_reordered.max() - U_vec_reordered.min())
            
            sns.heatmap(U_vec_normalized, cmap=cmap, cbar=False, ax=axes[idx],
                        yticklabels=liwc_sorted[:U_vec.shape[0]] if idx % 2 == 0 else False,
                        xticklabels=range(1, 25) if idx >= 2 else False)
            
            axes[idx].set_title(title_map[genre].title(), fontsize=18)
            
            if idx >= 2:  # Bottom row
                axes[idx].set_xlabel('Column Index', fontsize=16)
            
            if idx % 2 == 0:  # Left column
                axes[idx].set_ylabel('LIWC Categories', fontsize=16)
            
            # Set x-ticks for every tick and label them from 1 to 24
            axes[idx].set_xticks(np.arange(0.5, 24.5, 1))
            if idx >= 2:
                axes[idx].set_xticklabels(range(1, 25), rotation=90, ha='center')

plt.tight_layout(rect=[0, 0.03, 0.97, 0.95])

# Add colorbar to the right of the subplots
cbar_ax = fig.add_axes([0.98, 0.15, 0.02, 0.7])
cbar = fig.colorbar(axes[0].collections[0], cax=cbar_ax)
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('Normalized Absolute Value', fontsize=16)

plt.savefig('images/U_vec_heatmaps/latent_group_plots_2x2.pdf', dpi=300, bbox_inches='tight')
plt.close()