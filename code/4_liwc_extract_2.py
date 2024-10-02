import pickle
import numpy as np
import os
import tqdm
import re
from LIWC import liwc
from functools import lru_cache

@lru_cache(maxsize=None)
def liwcMaker(text):
    text = text.lower().replace('[MASK]', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    liwcDict = liwc().getLIWCCount(text)
    liwcHeaders = sorted(liwcDict.keys())
    
    wc_value = liwcDict['WC']
    if wc_value > 0:
        return np.array([liwcDict[h] / wc_value if h != 'WC' else liwcDict[h] for h in liwcHeaders])
    return np.zeros(len(liwcHeaders))

ROOT = 'liwc_data/'
model_list = ['distilbert-base-uncased_style/', 'bert-base-uncased_style/', 'bert-large-uncased_style/', 'roberta-base_style/']

liwc_instance = liwc()

for model in model_list:
    os.makedirs(os.path.join(ROOT, model), exist_ok=True)
    for f in os.listdir(model):
        if f.endswith('.pickle'):
            print(model, f)
            file_name = f[:-7]  # Remove '.pickle'
            
            with open(os.path.join(model, f), 'rb') as handle:
                encoder_embedding = pickle.load(handle)

            sentences = []
            for item in tqdm.tqdm(encoder_embedding):
                text = item[0].lower()
                text = text.lower().replace('[MASK]', ' ')
                text = re.sub(r'\s+', ' ', text)
                liwcDict = liwc_instance.getLIWCCount(text)
                liwcHeaders = sorted(liwcDict.keys())
                wc_value = liwcDict['WC']
                vec = [liwcDict[h] / wc_value if h != 'WC' else liwcDict[h] for h in liwcHeaders]
                sentences.append([item[0], item[1], vec, item[3]])

            with open(os.path.join(ROOT, model, f'{file_name}_liwc.pickle'), 'wb') as f:
                pickle.dump(sentences, f)