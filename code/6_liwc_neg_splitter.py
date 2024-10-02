linguistic_dimensions = {
    'Linguistic_Dimensions': [
        'funct', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they',
        'ipron', 'article', 'prep', 'auxverb', 'adverb', 'conj', 'negate'
    ],
    'Other_Grammar': [
        'verb', 'adj', 'compare', 'interrog', 'number', 'quant'
    ],
    'Affective_Processes': [
        'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad'
    ],
    'Social_Processes': [
        'social', 'family', 'friend', 'female', 'male'
    ],
    'Cognitive_Processes': [
        'cogproc', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'differ'
    ],
    'Perceptual_Processes': [
        'percept', 'see', 'hear', 'feel'
    ],
    'Biological_Processes': [
        'bio', 'body', 'health', 'sexual', 'ingest'
    ],
    'Drives': [
        'drives', 'affiliation', 'achieve', 'power', 'reward', 'risk'
    ],
    'Time_Orientations': [
        'focuspast', 'focuspresent', 'focusfuture'
    ],
    'Relativity': [
        'relativ', 'motion', 'space', 'time'
    ],
    'Personal_Concerns': [
        'work', 'leisure', 'home', 'money', 'relig', 'death'
    ],
    'Informal_language': [
        'informal', 'swear', 'netspeak', 'assent', 'nonflu', 'filler'
    ]
}
import pickle
import os

from LIWC import liwc
import numpy as np
import tqdm

liwcDict = liwc().getLIWCCount('')
liwcHeaders = list(liwcDict.keys())

liwc_sorted  = sorted(liwcHeaders)
sorted_liwc_idx = {}
for i in range(0,len(liwc_sorted)):
    sorted_liwc_idx[liwc_sorted[i]] = i

shuffle_order = {}
for i in range(0,len(liwcHeaders)):
    cat = liwcHeaders[i] 
    #print(cat,sorted_liwc_idx[cat])
    shuffle_order[sorted_liwc_idx[cat]] =i
ROOT = './liwc_style/'
for data in os.listdir(ROOT):

    with open(ROOT+data, 'rb') as handle:
        data_ = pickle.load(handle)


    for category in tqdm.tqdm(linguistic_dimensions):
        neg_dimensions = linguistic_dimensions[category]
        dimensions = []
        for dim in liwcHeaders:
            if dim not in neg_dimensions:
                dimensions.append(dim)
        valid_indices = [shuffle_order[liwcHeaders.index(d)] for d in dimensions]
        new_data = []
        for item in data_:
            filtered_vals = []
            for i in valid_indices:
                filtered_vals.append(item[2][i])
            new_data.append([item[0], item[1], np.array(filtered_vals)])
        
        ROOT2 = './liwc_neg_dimensions/'
        if not os.path.exists(ROOT2):
            os.makedirs(ROOT2)
        
        if not os.path.exists(ROOT2+category):
            os.makedirs(ROOT2+category)
        
        with open(ROOT2+category+'/'+data, 'wb') as handle:
                pickle.dump(new_data, handle)
        