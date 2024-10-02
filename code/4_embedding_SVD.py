import pickle
import numpy as np
import dask.array as da
import os
import tqdm
ROOT = 'SLIM_LLM/'
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
                X.append(item[2][0])


            X = np.array(X)
            X = da.from_array(X)  # Adjust chunks as needed
            u, s, v = da.linalg.svd_compressed(X, k=768)
            u_numpy = u.compute()

            for i in range(0,len(u_numpy)):
                sentences.append([encoder_embedding[i][0], encoder_embedding[i][1], u_numpy[i], encoder_embedding[i][3]])

            if not os.path.exists(ROOT):
                os.makedirs(ROOT)
            if not os.path.exists(ROOT+model):
                os.makedirs(ROOT+model)                
            with open(ROOT+model+file_name+'_svd.pickle', 'wb') as f:
                pickle.dump(sentences, f)    