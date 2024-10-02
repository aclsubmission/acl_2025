import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from LIWC import liwc
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({'font.size': 14})  # Set default font size to 14

liwcDict = liwc().getLIWCCount('')
liwcHeaders = list(liwcDict.keys())

liwc_sorted = sorted(liwcHeaders)
sorted_liwc_idx = {cat: i for i, cat in enumerate(liwc_sorted)}

shuffle_order = {i: sorted_liwc_idx[cat] for i, cat in enumerate(liwcHeaders)}

ROOT = './style_rank/'
genres = ['descriptions_airbnb', 'gutenberg', 'review_yelp']
title_map = {'descriptions_airbnb':'Advertisements', 'gutenberg':'Novels', 'review_yelp':'Business Review'}
columns = [2, 10, 1]

cmap = LinearSegmentedColormap.from_list("custom_wb", ["white", "black"])

combined_matrix = []

for genre, column in zip(genres, columns):
    for data in os.listdir(ROOT):
        if data.startswith(genre) and data.endswith('.pickle_U_RANK.pickle'):
            with open(os.path.join(ROOT, data), 'rb') as handle:
                data_ = pickle.load(handle)
            
            len_map = {len(data_[i][0]): i for i in range(len(data_))}
            
            U_vec = data_[len_map[24]]
            if U_vec.ndim == 1:
                size = int(np.sqrt(U_vec.shape[0]))
                U_vec = U_vec.reshape(size, size)
            
            U_vec = np.abs(U_vec[1:])
            
            reorder_indices = [shuffle_order[i] for i in range(U_vec.shape[0])]
            U_vec_reordered = U_vec[reorder_indices]
            
            U_vec_normalized = (U_vec_reordered - U_vec_reordered.min()) / (U_vec_reordered.max() - U_vec_reordered.min())
            
            combined_matrix.append(U_vec_normalized[:, column])

combined_matrix = np.column_stack(combined_matrix)

# Rotate the matrix 90 degrees counterclockwise
combined_matrix = np.rot90(combined_matrix)

fig, ax = plt.subplots(figsize=(15, 3.5))  # Increased height to accommodate labels
genres = [ 'review_yelp', 'gutenberg','descriptions_airbnb']
y_labels = [title_map[genre] for genre in genres]

sns.heatmap(combined_matrix, cmap=cmap, cbar_kws={'label': 'Group Contribution', 'orientation': 'horizontal'},
            xticklabels=liwcHeaders[:combined_matrix.shape[1]],
            yticklabels=y_labels, ax=ax)



ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

plt.xlabel('LIWC Categories', fontsize=18)  # Increased font size to 18
ax.xaxis.label.set_size(18)  # Ensure xlabel font size is 18

# Rotate x-axis labels by 90 degrees and set font size to 18
plt.xticks(rotation=90, fontsize=12)

# Add 'genre' label on top of y-axis labels with font size 18
ax.text(-0.05, 1.02, 'Genres', transform=ax.transAxes, ha='right', va='bottom', fontsize=18)

# Set y-axis (genre) label font size to 14
ax.tick_params(axis='y', labelsize=14)

# Add 'columns' label on top of column numbers
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(ax.get_yticks())
ax2.set_yticklabels(columns, fontsize=18)  # Set font size to 18
ax2.set_ylabel('Dimension', rotation=270, labelpad=15, fontsize=18)  # Set font size to 18
ax2.yaxis.set_label_position("right")

plt.tight_layout()
plt.savefig('images/combined_U_vec_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.close()