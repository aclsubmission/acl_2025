import matplotlib.gridspec as gridspec
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from LIWC import liwc
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

liwcDict = liwc().getLIWCCount('')
liwcHeaders = list(liwcDict.keys())

liwc_sorted = sorted(liwcHeaders)
sorted_liwc_idx = {cat: i for i, cat in enumerate(liwc_sorted)}

shuffle_order = {i: sorted_liwc_idx[cat] for i, cat in enumerate(liwcHeaders)}

ROOT = './style_rank/'
genres = ['articles_wikipedia', 'descriptions_airbnb', 'gutenberg_domesticfiction', 'review_yelp', 'lyrics_hot100']

cmap = LinearSegmentedColormap.from_list("custom_wb", ["white", "black"])

for data in os.listdir(ROOT):
    if data.endswith('.pickle_U_RANK.pickle') and data.split('.')[0] in genres:
        name = data.split('.')[0]
        with open(os.path.join(ROOT, data), 'rb') as handle:
            data_ = pickle.load(handle)
        
        len_map = {len(data_[i][0]): i for i in range(len(data_))}

        U_vec = data_[len_map[24]]
        if U_vec.ndim == 1:
            size = int(np.sqrt(U_vec.shape[0]))
            U_vec = U_vec.reshape(size, size)
        
        U_vec = np.abs(U_vec[1:])
        U_vec = np.power(U_vec, 1.0)
        reorder_indices = [shuffle_order[i] for i in range(U_vec.shape[0])]
        U_vec_reordered = U_vec[reorder_indices]
        
        U_vec_normalized = (U_vec_reordered - U_vec_reordered.min()) / (U_vec_reordered.max() - U_vec_reordered.min())

        # Create a figure and a GridSpec layout
        fig = plt.figure(figsize=(14, 14))  # Increased figure width
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 15, 0.5])  # Added a column for labels

        # Create the label subplot
        ax_labels = plt.subplot(gs[0])
        ax_labels.set_axis_off()
        for i, category in enumerate(liwcHeaders[:U_vec.shape[0]]):
            ax_labels.text(1, 1 - (i + 0.5) / len(liwcHeaders), category, 
                           ha='right', va='center', fontsize=14)

        # Create the heatmap in the middle subplot
        ax_heatmap = plt.subplot(gs[1])
        cbar_ax = plt.subplot(gs[2])

        # Create heatmap without y-axis labels
        sns.heatmap(U_vec_normalized, cmap=cmap, ax=ax_heatmap, cbar_ax=cbar_ax, 
                    cbar_kws={'label': 'Group Contribution'},
                    yticklabels=False)
        cbar_ax.set_ylabel('Group Contribution', fontsize=25)

        ax_heatmap.set_xlabel('Dimensions', fontsize=25)
        ax_heatmap.set_ylabel('')  # Remove y-axis label

        # Set font size for x-axis ticks
        ax_heatmap.tick_params(axis='x', labelsize=22)
        
        # Set font size for colorbar ticks
        cbar_ax.tick_params(labelsize=16)

        plt.tight_layout()
        plt.savefig(f'images/U_vec_heatmaps/{name}_heatmap.pdf', dpi=300, bbox_inches='tight')
        plt.close()