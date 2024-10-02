import sys
import csv

csv.field_size_limit(sys.maxsize)
LANG = 'en'
#source ../../Other/joham/textbooks/virtenv/text/bin/activate
import os
#Measures the similiarity of sensorial representations
import math
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import spatial
import tqdm
import numpy as np
import re
import matplotlib.pyplot as plt
import csv 
import random
from nltk.stem import WordNetLemmatizer
from collections import Counter



#load sense:
sense_headers = []
senses = {}
sense_strength = {}
sense_rarity = {}
rarity_counter = 0
with open('dict/lexicon_weight.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        sense_headers.append(row[1])
        senses[row[0]]=row[1]
        sense_strength[row[0]] = float(row[2])/5
        

def senseMaker(sent):
    '''
    MASKS sense words in sentences
    '''
    sentence_set = []

    sent = sent.lower()
    words = word_tokenize(sent)    
    intersection = set(words).intersection(set(senses.keys()))
    if len(intersection)>0:
        for item in intersection:
            idx = sent.find(' '+item+' ')
            sentence = sent[:idx]+' [MASK] '+sent[idx+len(item)+2:]
            sentence_set.append([sentence,item])
    return(sentence_set)

ROOT = 'raw_data/'
files = os.listdir(ROOT)
for file in files:
    print(file)


    if file.startswith('.')==False:
        with open(ROOT+file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            data = []
            for row in tqdm.tqdm(csv_reader):
                res = senseMaker(row[0])
                if len(res)>0:
                    for item_ in res:
                        data.append(item_)
                    
        if not os.path.exists('sense_data'):
            os.makedirs('sense_data')
        print(file,len(data))
        with open('sense_data/'+file, 'w') as f:
            writer = csv.writer(f)
            for sentence in data:
                writer.writerow(sentence)