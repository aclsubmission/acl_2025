import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

ROOT = 'results/results/'
title_map = {'articles_wikipedia':'Articles', 'descriptions_airbnb':'Advertisements', 'gutenberg_domesticfiction':'Novels', 'review_yelp':'Business Review', 'yrics_hot100':'Music Lyrics'}
def load_data(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def process_genre_data(fat, slim, baseline, liwc, model, genre):
    bert_res = list(fat[model][genre]['clean'].values())[:18]
    slim_liwc = list(fat[model][genre]['dirty'].values())[:18]
    slim_bert = list(slim[model][genre]['dirty'].values())[:18]
    rank = list(fat[model][genre]['dirty'].keys())[:18]
    if 'lyrics' not in genre:
        genre = genre.replace('yrics','lyrics')
    bert_ = baseline[model][genre]
    liwc_ = liwc[model][genre]
    
    return bert_res, slim_liwc, slim_bert, bert_, liwc_, rank

def plot_genre(ax, rank, slim_bert, slim_liwc, bert_res, bert_, liwc_, genre):
    ax.plot(rank, slim_bert, color='blue', linestyle=':', marker='s', markersize=4)
    ax.plot(rank, slim_liwc, color='green', linestyle='-.', marker='o', markersize=4)
    ax.plot(rank, bert_res, color='red', linestyle='-.', marker='x', markersize=4)
    ax.axhline(y=bert_, color='purple', linestyle='--', linewidth=1)
    ax.axhline(y=liwc_, color='orange', linestyle='--', linewidth=1)
    ax.set_title(title_map[genre],fontsize=18)
    ax.grid(False)
    ax.tick_params(axis='x', labelsize=14)  # Increase xtick size to 14
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', alpha=0.2, color='gray')

files = [f for f in os.listdir(ROOT) if not f.endswith('_U_vec.pickle') and not f.startswith('.')]
fig, axs = plt.subplots(1, 5, figsize=(18, 5))
model = 'bert-base-uncased'
fig.suptitle('')

for f in files:
    model = f.split('_')[0].replace('/','')
    if model.startswith('bert-base-uncased'):
        print(f)
        fat = load_data(ROOT + f)
        slim = load_data(ROOT + f.replace('_yrics','_lyrics').replace('.pickle','_U_vec.pickle'))
        baseline = load_data('results_baseline/' + f.replace('_yrics','_lyrics'))
        liwc = load_data('results_base_style/' + f.replace('_yrics','_lyrics'))
        for model_ in fat:
            genres = list(fat[model_].keys())
        genres = ['articles_wikipedia','descriptions_airbnb','gutenberg_domesticfiction','review_yelp','yrics_hot100']
        
        for i, genre in enumerate(genres):
            if genre in fat[model_]:
                bert_res, slim_liwc, slim_bert, bert_, liwc_, rank = process_genre_data(fat, slim, baseline, liwc, model_, genre)
                plot_genre(axs[i], rank, slim_bert, slim_liwc, bert_res, bert_, liwc_, genre)

# Remove x and y labels from individual subplots
for ax in axs:
    ax.set_xlabel('')
    ax.set_ylabel('')

# Adjust the layout to make room for the y-label
plt.subplots_adjust(left=0.1)

# Add a common y-label
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=18)

# Add a common x-label
fig.text(0.5, 0.00, 'Rank', ha='center', fontsize=18)

# Create a single legend for all subplots
lines = [
    plt.Line2D([0], [0], color='blue', linestyle=':', marker='s', markersize=4),
    plt.Line2D([0], [0], color='green', linestyle='-.', marker='o', markersize=4),
    plt.Line2D([0], [0], color='red', linestyle='-.', marker='x', markersize=4),
    plt.Line2D([0], [0], color='purple', linestyle='--', linewidth=1),
    plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=1)
]
labels = ['SLIM-BERT+Latent LIWC', 'SLIM-BERT+LIWC', 'SLIM-BERT', 'BERT', 'LIWC']
fig.legend(lines, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.55, -0.125), fontsize=18)  # Increased fontsize to 14


plt.tight_layout()

# Adjust the layout again after tight_layout to ensure the y-label doesn't overlap
plt.subplots_adjust(left=0.1)

plt.savefig(f'images/acc/accuracy_vs_rank_{model}_panel.pdf', dpi=300, bbox_inches='tight')
plt.close()