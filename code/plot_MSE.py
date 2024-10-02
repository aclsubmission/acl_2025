import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from LIWC import liwc
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


liwcDict = liwc().getLIWCCount('')
liwcHeaders = list(liwcDict.keys())

liwc_sorted  = sorted(liwcHeaders)
sorted_liwc_idx = {}
for i in range(0,len(liwc_sorted)):
    sorted_liwc_idx[liwc_sorted[i]] = i

shuffle_order = {}
for i in range(0,len(liwcHeaders)):
    cat = liwcHeaders[i] 
    print(cat,sorted_liwc_idx[cat])
    shuffle_order[i] =sorted_liwc_idx[cat]

ROOT = './style_rank/'
genres = ['articles_wikipedia','descriptions_airbnb','gutenberg_domesticfiction','review_yelp','lyrics_hot100']
for data in os.listdir(ROOT):
    if data.endswith('test_MSE.pickle') and data.split('.')[0] in genres:
        name = data.split('.')[0]
        with open(os.path.join(ROOT, data), 'rb') as handle:
            data_ = pickle.load(handle)
        
        keys = list(data_.keys())[:-1]
        values = list(data_.values())[:-1]
        plt.figure(figsize=(12, 8))        
        plt.plot(keys, values, label=name)

        plt.xlabel('Keys')
        plt.ylabel('Values (MSE)')
        plt.title('Test MSE for Different Styles')
        plt.legend()
        plt.grid(True)
        
        # Set y-axis to logarithmic scale
        #plt.yscale('log')

        plt.savefig('images/MSE2/'+name+'_MSE.png')
        #plt.show()
        plt.close()


import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from LIWC import liwc
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ... (previous code remains the same)

cmap = LinearSegmentedColormap.from_list("custom_wb", ["white", "black"])

for data in os.listdir(ROOT):
    if data.endswith('.pickle_U_RANK.pickle'):
        name = data.split('.')[0]
        with open(os.path.join(ROOT, data), 'rb') as handle:
            data_ = pickle.load(handle)
        
        len_map = {}
        for i in range(0, len(data_)):
            len_map[len(data_[i][0])] = i

        U_vec = data_[len_map[24]]
        if U_vec.ndim == 1:
            # Assuming U_vec is a square matrix flattened to 1D
            size = int(np.sqrt(U_vec.shape[0]))
            U_vec = U_vec.reshape(size, size)
        
        U_vec = np.abs(U_vec[1:])
        
        # Reorder U_vec based on shuffle_order
        reorder_indices = [shuffle_order[i] for i in range(U_vec.shape[0])]
        U_vec_reordered = U_vec[reorder_indices]
        
        # Normalize U_vec to [0, 1] range
        U_vec_normalized = (U_vec_reordered - U_vec_reordered.min()) / (U_vec_reordered.max() - U_vec_reordered.min())
        
        plt.figure(figsize=(15, 14))  # Increased figure size for better readability
        
        # Create heatmap with liwc_sorted as y-axis labels, using the custom colormap
        sns.heatmap(U_vec_normalized, cmap=cmap, cbar_kws={'label': 'Normalized Absolute Value'},
                    yticklabels=liwcHeaders[:U_vec.shape[0]])  # Use liwc_sorted for y-axis labels
        
        plt.title(f'Heatmap of Normalized Absolute U_vec for {name} (Reordered)')
        plt.xlabel('Column Index')
        plt.ylabel('LIWC Categories')
        
        plt.tight_layout()
        plt.savefig(f'images/U_vec_heatmaps/{name}_U_vec_heatmap_wb_reordered.png', dpi=300, bbox_inches='tight')
        plt.close()
