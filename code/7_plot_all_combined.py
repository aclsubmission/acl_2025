import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from LIWC import liwc
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Initialize LIWC
liwcDict = liwc().getLIWCCount('')
liwcHeaders = list(liwcDict.keys())

liwc_sorted = sorted(liwcHeaders)
sorted_liwc_idx = {cat: i for i, cat in enumerate(liwc_sorted)}

shuffle_order = {i: sorted_liwc_idx[cat] for i, cat in enumerate(liwcHeaders)}

genres = ['descriptions_airbnb', 'gutenberg', 'review_yelp']
columns_to_keep = [1, 2, 10]
cmap = LinearSegmentedColormap.from_list("custom_wb", ["white", "black"])

ROOT = './style_rank/'

combined_matrix = []

for genre in genres:
    for data in os.listdir(ROOT):
        if data.startswith(genre) and data.endswith('.pickle_U_RANK.pickle'):
            name = data.split('.')[0]
            with open(os.path.join(ROOT, data), 'rb') as handle:
                data_ = pickle.load(handle)
            
            len_map = {len(d[0]): i for i, d in enumerate(data_)}
            
            U_vec = data_[len_map[24]]
            if U_vec.ndim == 1:
                size = int(np.sqrt(U_vec.shape[0]))
                U_vec = U_vec.reshape(size, size)
            
            U_vec = np.abs(U_vec[1:])
            
            # Normalize U_vec for each genre
            U_vec_normalized = (U_vec - U_vec.min()) / (U_vec.max() - U_vec.min())
            
            # Filter out columns 1, 2, 10
            U_vec_filtered = U_vec_normalized[:, columns_to_keep]
            
            combined_matrix.append(U_vec_filtered)

# Combine matrices from all genres
combined_matrix = np.hstack(combined_matrix)

# Reorder rows based on shuffle_order
reorder_indices = [shuffle_order[i] for i in range(combined_matrix.shape[0])]
combined_matrix_reordered = combined_matrix[reorder_indices]

# Rotate the matrix 90 degrees clockwise
combined_matrix_rotated = np.rot90(combined_matrix_reordered, k=1)

# Plot the heatmap
plt.figure(figsize=(15, 4))

ax = sns.heatmap(combined_matrix_rotated, cmap=cmap, cbar_kws={'label': 'Group Contribution', 'orientation': 'horizontal', 'shrink': 0.2},
            xticklabels=liwcHeaders, yticklabels=False)

cbar = ax.collections[0].colorbar
cbar.set_label('Group Contribution', fontsize=14)  # Set font size for the colorbar label

plt.title('')
plt.xlabel('LIWC Categories',fontsize=18)

# Move x-axis ticks and labels to the top
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')  # Move the x-axis label as well


# Rotate x-axis tick labels by 90 degrees
plt.xticks(rotation=90,fontsize=13)



# Set custom y-axis ticks and labels (on the right)
y_ticks = np.arange(0.5, 9, 1)
y_labels = ['10', '2', '1'] * 3  # Reversed order due to rotation
plt.yticks(y_ticks, y_labels, va='center', rotation=0,fontsize=12)
plt.gca().yaxis.tick_right()  # Move y-axis ticks and labels to the right

# Add genre labels on the left side
genre_positions = [1, 4, 7]
genre_labels = ['Business Reviews', 'Novels', 'Advertisements']  # Reversed order due to rotation
for pos, label in zip(genre_positions, genre_labels):
    plt.text(-0.5, pos, label, 
             horizontalalignment='right', verticalalignment='center', rotation=0, fontsize=14)

# Add "Dimensions" label to the right side
plt.text(combined_matrix_rotated.shape[1] + 2.5, 4.5, 'Dimensions', 
         rotation=-90, va='center', ha='center', fontsize=14)
for genre_sep in [3, 6]:  # Assuming each genre has 3 rows
    plt.axhline(y=genre_sep, color='black', linestyle='-', linewidth=1.5)

plt.tight_layout()
plt.savefig('combined_U_vec_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.close()
