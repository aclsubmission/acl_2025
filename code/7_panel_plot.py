import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ROOT = './style_rank/'
genres = ['articles_wikipedia', 'descriptions_airbnb', 'gutenberg_domesticfiction', 'review_yelp', 'lyrics_hot100']
title_map = {'articles_wikipedia':'Articles', 'descriptions_airbnb':'Advertisements', 'gutenberg_domesticfiction':'Novels', 'review_yelp':'Business Review', 'lyrics_hot100':'Music Lyrics'}

# Create a figure with 5 subplots in a row and an extra subplot for the y-label on the left
fig, axs = plt.subplots(1, 6, figsize=(18, 5), gridspec_kw={'width_ratios': [0.1, 1, 1, 1, 1, 1]})
fig.suptitle('', fontsize=24)

for idx, genre in enumerate(genres):
    data_file = f"{genre}.csv.pickle_test_MSE.pickle"
    if os.path.exists(os.path.join(ROOT, data_file)):
        with open(os.path.join(ROOT, data_file), 'rb') as handle:
            data_ = pickle.load(handle)
        
        keys = list(data_.keys())[:-1]
        values = list(data_.values())[:-1]
        
        axs[idx+1].plot(keys, values)
        axs[idx+1].set_title(title_map[genre].title())
        axs[idx+1].set_xlabel('')
        axs[idx+1].grid(True)
        
        # Set x-ticks to every 5 units
        axs[idx+1].set_xticks(range(0, max(keys)+1, 10))
        
        # Rotate x-axis labels for better readability
        axs[idx+1].tick_params(axis='x', rotation=45)
        
        # Add dotted line at x = 24
        axs[idx+1].axvline(x=24, color='r', linestyle=':', linewidth=2)
        
        # Ensure x-axis includes the dotted line
        x_min, x_max = axs[idx+1].get_xlim()
        axs[idx+1].set_xlim([min(x_min, 24), max(x_max, 24)])
        
        # Remove y-label from individual subplots
        axs[idx+1].set_ylabel('')

# Set the MSE label on the left side
axs[0].set_visible(False)
fig.text(0.05, 0.5, 'Mean Squared Error', va='center', rotation=90, fontsize=14)

# Set the Ranks label at the bottom center
fig.text(0.5, 0.05, 'Ranks', ha='center', fontsize=14)

# Adjust the layout and save the figure
plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])  # Adjust the rect parameter to accommodate all labels
plt.savefig('images/MSE/all_genres_MSE.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plot saved as 'images/MSE/all_genres_MSE.png'")