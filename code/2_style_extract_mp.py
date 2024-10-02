import sys
import random
import csv
import re
import os
import pickle
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from LIWC import liwc

csv.field_size_limit(sys.maxsize)

# Precompile regex pattern
mask_re = re.compile(r'\s+')

def liwcMaker(text):
    text = text.lower().replace('[MASK]', ' ')
    text = mask_re.sub(' ', text)
    
    liwcDict = liwc().getLIWCCount(text)
    liwcHeaders = list(liwcDict.keys())
    liwcHeaders.sort()

    wc_value = liwcDict['WC']
    if wc_value > 0:
        liwcArray = np.array([liwcDict[h] / wc_value if h != 'WC' else liwcDict[h] for h in liwcHeaders])
    else:
        liwcArray = np.zeros(len(liwcHeaders))
    
    return liwcArray

# Process CSV in chunks
def process_chunk(data_chunk):
    data_set = []
    for data_ in data_chunk:
        text = data_[0]
        liwcArray = liwcMaker(text)
        data_.append(liwcArray)
        data_set.append(data_)
    return data_set

# Parallelize processing
def process_file_in_parallel(filepath, n_samples, n_processes=4):
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = [row for row in csv_reader]  # Read all data

        if len(data) >= n_samples:
            random_data = random.sample(data, n_samples)

            # Split the random data into chunks for parallel processing
            chunk_size = len(random_data) // n_processes
            chunks = [random_data[i:i + chunk_size] for i in range(0, len(random_data), chunk_size)]

            with mp.Pool(processes=n_processes) as pool:
                result_chunks = pool.map(process_chunk, chunks)

            # Combine all results from the chunks
            data_set = [item for sublist in result_chunks for item in sublist]

            # Save the data_set as a pickle
            if not os.path.exists('liwc_style'):
                os.makedirs('liwc_style')
            with open(f'liwc_style/{os.path.basename(filepath)}.pickle', 'wb') as f:
                pickle.dump(data_set, f)

# Main process
if __name__ == '__main__':  # Protect multiprocessing code
    N = 300000
    ROOT = 'sense_data/'
    files = os.listdir(ROOT)

    for file in files:
        if not file.startswith('.'):
            print(f"Processing file: {file}")
            process_file_in_parallel(ROOT + file, N)
