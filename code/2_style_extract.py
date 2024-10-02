N = 300000
import sys
import random
import csv
import tqdm
import re
csv.field_size_limit(sys.maxsize)

#source ../../Other/joham/textbooks/virtenv/text/bin/activate
import os
#Measures the similiarity of sensorial representations
import math
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




ROOT = 'sense_data/'
files = os.listdir(ROOT)
for file in files:
    print(file)
    if file.startswith('.')==False:
        if file == 'lyrics_hot100.csv':
            with open(ROOT+file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                data = []
                for row in tqdm.tqdm(csv_reader):
                    data.append(row)
                if len(data)>=N:
                    data_set = []
                    random_data = random.sample(data, N)
                    for data_ in tqdm.tqdm(random_data):
                        text = data_[0]
                        liwcArray= liwcMaker(text)
                        data_.append(liwcArray)
                        data_set.append(data_)
                    # Write data_set to pickle file
                    if not os.path.exists('liwc_style'):
                        os.makedirs('liwc_style')
                    with open('liwc_style/'+file+'.pickle', 'wb') as f:
                        pickle.dump(data_set, f)
                    
