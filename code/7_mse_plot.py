import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

ROOT = './style_rank/'
genres = ['articles_wikipedia', 'descriptions_airbnb', 'gutenberg_domesticfiction', 'review_yelp', 'lyrics_hot100']
title_map = {'articles_wikipedia':'Articles', 'descriptions_airbnb':'Advertisements', 'gutenberg_domesticfiction':'Novels', 'review_yelp':'Business Review', 'lyrics_hot100':'Music Lyrics'}

# Define line styles and markers for each genre
styles = [
    {'linestyle': '-', 'marker': 'o', 'markersize': 4},
    {'linestyle': '--', 'marker': 's', 'markersize': 4},
    {'linestyle': '-.', 'marker': '^', 'markersize': 4},
    {'linestyle': ':', 'marker': 'D', 'markersize': 4},
    {'linestyle': (0, (3, 1, 1, 1)), 'marker': 'x', 'markersize': 4}
]

# Create a figure with 5 subplots in a row and an extra subplot for the y-label on the left
fig, axs = plt.subplots(1, 6, figsize=(18, 5.5), gridspec_kw={'width_ratios': [0.1, 1, 1, 1, 1, 1]})
fig.suptitle('', fontsize=24)

for idx, genre in enumerate(genres):
    data_file = f"{genre}.csv.pickle_test_MSE.pickle"
    if os.path.exists(os.path.join(ROOT, data_file)):
        with open(os.path.join(ROOT, data_file), 'rb') as handle:
            data_ = pickle.load(handle)
        
        keys = list(data_.keys())[:-1]
        values = list(data_.values())[:-1]
        
        axs[idx+1].plot(keys, values, color='black', **styles[idx], label=title_map[genre])
        #axs[idx+1].set_title(title_map[genre].title())
        axs[idx+1].set_title(title_map[genre].title(), fontsize=18)  # Increased title font size to 16

        axs[idx+1].set_xlabel('')
        axs[idx+1].grid(True, linestyle=':', alpha=0.5)
        
        # Set x-ticks to every 10 units
        axs[idx+1].set_xticks(range(0, max(keys)+1, 10))
        
        # Rotate x-axis labels for better readability
        axs[idx+1].tick_params(axis='x', rotation=45, labelsize=14)
        
        # Add dotted line at x = 24
        axs[idx+1].axvline(x=24, color='gray', linestyle='--', linewidth=2)
        
        # Ensure x-axis includes the dotted line
        x_min, x_max = axs[idx+1].get_xlim()
        axs[idx+1].set_xlim([min(x_min, 24), max(x_max, 24)])
        
        # Remove y-label from individual subplots
        axs[idx+1].set_ylabel('')

# Set the MSE label on the left side
axs[0].set_visible(False)
fig.text(0.05, 0.5, 'Mean Squared Error', va='center', rotation=90, fontsize=18)

# Set the Ranks label at the bottom center
fig.text(0.5, 0.05, 'Ranks', ha='center', fontsize=18)

# Add a legend
handles, labels = axs[1].get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))

# Adjust the layout and save the figure
plt.tight_layout(rect=[0.03, 0.05, 1, 0.92])  # Adjusted to accommodate the legend
plt.savefig('images/MSE/all_genres_MSE.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Plot saved as 'images/MSE/all_genres_MSE.png'")

