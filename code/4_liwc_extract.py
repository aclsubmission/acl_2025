import pickle
import numpy as np
import dask.array as da
import os
import tqdm
import re
from LIWC import liwc
import pickle
import numpy as np

def liwcMaker(text):
    
    text = text.replace('[MASK]', ' ')
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    
    liwcDict = liwc().getLIWCCount(text)
    liwcHeaders = list(liwcDict.keys())
    liwcHeaders.sort()
    
    # Create a NumPy array from the liwcDict values, excluding 'WC'
    wc_value = liwcDict['WC']
    if wc_value > 0:
        liwcArray = np.array([liwcDict[h] / wc_value if h != 'WC' else liwcDict[h] for h in liwcHeaders])
    else:
        liwcArray = np.zeros(len(liwcHeaders))
    
    return liwcArray


ROOT = 'liwc_data/'


model_list = ['distilbert-base-uncased_style/', 'bert-base-uncased_style/', 'bert-large-uncased_style/', 'roberta-base_style/']
for model in model_list:
    files = os.listdir(model)
    for f in files:
        print(model,f)
        if f.endswith('.pickle'):
            file_name=f.strip('.pickle')
            with open(model+f, 'rb') as handle:
                encoder_embedding = pickle.load(handle)

            sentences = []                
            X = []
            for item in tqdm.tqdm(encoder_embedding):
                text =item[0]
                text = text.lower()
                liwcDict = liwc().getLIWCCount(text)
                liwcHeaders = list(liwcDict.keys())
                liwcHeaders.sort()
                vec = []
                for h in liwcHeaders:
                    vec.append(liwcDict[h])
                sentences.append([item[0], item[1], vec, item[3]])

            if not os.path.exists(ROOT):
                os.makedirs(ROOT)
            if not os.path.exists(ROOT+model):
                os.makedirs(ROOT+model)                
            with open(ROOT+model+file_name+'_liwc.pickle', 'wb') as f:
                pickle.dump(sentences, f)    
