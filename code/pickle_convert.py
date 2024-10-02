import pickle
import os
ROOT = './style_rank/'
for data in os.listdir(ROOT):
    if data.endswith('.pickle'):
        
        with open(ROOT+data, 'rb') as handle:
            data_ = pickle.load(handle)
        
        # Write to pickle
        with open(ROOT + data, 'wb') as handle:
            pickle.dump(data_, handle)