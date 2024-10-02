import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pickle
import os

ROOT = 'results/results/'

files = os.listdir(ROOT)
for f in files:
    try:
        if f.endswith('_U_vec.pickle')==False and f.startswith('.')==False:
            model = f.split('_')[0].replace('/','')
            if model.startswith('bert-base-uncased'):
                with open(ROOT+f, 'rb') as handle:
                    fat = pickle.load(handle)  
                
                with open(ROOT+f.replace('_yrics','_lyrics').replace('.pickle','_U_vec.pickle'), 'rb') as handle:
                    slim = pickle.load(handle)  


                f = f.replace('_yrics','_lyrics')
                with open('results_baseline/'+f, 'rb') as handle:
                    baseline = pickle.load(handle)  

                with open('results_base_style/'+f, 'rb') as handle:
                    liwc= pickle.load(handle)  
                
                for model in fat:
                    for genre in fat[model]:
                        bert_res = fat[model][genre]['clean']
                        slim_liwc = fat[model][genre]['dirty']
                        slim_bert = slim[model][genre]['dirty']
                        if 'lyrics' not in genre:
                            genre = genre.replace('yrics','lyrics')
                        bert_ = baseline[model][genre]
                        liwc_ = liwc[model][genre]
                        rank = list(slim_liwc.keys())[:18]
                        bert_res = list(bert_res.values())[:18]
                        slim_liwc = list(slim_liwc.values())[:18]
                        slim_bert = list(slim_bert.values())[:18]
                        print(model,genre,slim_liwc[-1])
                        model = model.replace('/','')
                        
                        # Create the plot
                        plt.figure(figsize=(12, 8))
                        
                        # Plot lines with distinct styles and markers
                        plt.plot(rank, slim_bert, color='black', linestyle=':', marker='s', markersize=6, label='SLIM-BERT+Latent LIWC')                
                        plt.plot(rank, slim_liwc, color='black', linestyle='-.', marker='o', markersize=6, label='SLIM-BERT+LIWC')
                        plt.plot(rank, bert_res, color='black', linestyle='-.', marker='x', markersize=6, label='SLIM-BERT')


                        # Add horizontal lines with distinct styles
                        plt.axhline(y=bert_, color='black', linestyle='--', linewidth=2, label=model.split('_')[0].title())
                        plt.axhline(y=liwc_, color='black', linestyle='--', linewidth=2, label='LIWC')

                        plt.xlabel('Rank', fontsize=14)
                        plt.ylabel('Accuracy', fontsize=14)
                        #plt.title(f'Accuracy vs Rank for {model} - {genre}')
                        plt.legend()
                        plt.grid(False)

                        # Add minor gridlines
                        plt.minorticks_on()
                        plt.grid(which='minor', linestyle=':', alpha=0.2, color='gray')

                        plt.tight_layout()
                        plt.savefig(f'images/acc/accuracy_vs_rank_{model}_{genre}.png', dpi=300)
                        plt.close()
    except:
        continue