N = 300000
import matplotlib.pyplot as plt
import tqdm
import random


import time

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

liwc_categories = {'all':list(range(0,74)),'function_words':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],'other_grammar':[16,17,18,19,20,21],'affective':[22,23,24,25,26,27],'social':[28,29,30,31,32],'cognitive':[33,34,35,36,37,38,39],'perceptual':[40,41,42,43],'bio':[44,45,46,47,48],'drive':[49,50,51,52,53,54],'time':[55,56,57],'relativity':[58,59,60,61],'personal':[62,63,64,65,66,67],'informal':[68,69,70,71,72,73]}

liwc_map_sorted = {}
liwc_map = {}
liwcDict = liwc().getLIWCCount('')
liwcHeaders = list(liwcDict.keys())
for i in range(0,len(liwcHeaders)):
    liwc_map[i] = liwcHeaders[i]

liwcHeaders.sort()
for i in range(0,len(liwcHeaders)):
    liwc_map_sorted[liwcHeaders[i]] = i








ROOT = 'liwc_style/'
files = os.listdir(ROOT)
output= {}
for file in files:
    #print(file)
    if file.startswith('.')==False:
        with open(ROOT+file, 'rb') as handle:
            liwc_style = pickle.load(handle)
        
        style_vec = []
        
        y_map = {}
        c=0
        for item in tqdm.tqdm(liwc_style):
            style_vec.append(item[2])
            label = item[1]
            if label not in y_map:
                y_map[label] = c
                c+=1


        Y = []
        X = []
        Y_test = []
        X_test = []
        for i in tqdm.tqdm(range(0,len(style_vec))):
            item = style_vec[i]
            if i < len(style_vec)*0.8:
            
                    x = np.array(item)
                    
                    X.append(x)
                    

                    y=np.zeros(len(y_map))
                    label = liwc_style[i][1]
                    y[y_map[label]] = 1
                    Y.append(np.array(y))
            else:
                    x = np.array(item)
                    X_test.append(x)
                    y=np.zeros(len(y_map))
                    label = liwc_style[i][1]
                    y[y_map[label]] = 1
                    Y_test.append(np.array(y))

        errors = {}
        for category in liwc_categories:
            
            valid_indices = [liwc_map_sorted[liwc_map[i]] for i in liwc_categories[category]]

            Y_vecs = np.array(Y)
            X_vecs = np.array(X)[:, valid_indices]  # Use only the valid columns
            
            Y_test_vecs = np.array(Y_test)
            X_test_vecs = np.array(X_test)[:, valid_indices]  # Use only the valid columns for test data as well


            X_vecs = np.insert(X_vecs, 0, 1, axis=1)
            X_test_vecs = np.insert(X_test_vecs, 0, 1, axis=1)
            beta = np.dot(np.linalg.inv(np.dot(X_vecs.T, X_vecs)), np.dot(X_vecs.T, Y_vecs))
            Y_pred = np.dot(X_test_vecs, beta)
            frobenius_norm = np.linalg.norm(Y_test_vecs - Y_pred, ord='fro')/(Y_vecs.shape[0]*Y_vecs.shape[1])
            #print(category,len(valid_indices),frobenius_norm)
            errors[category] = frobenius_norm
        for item in errors:
            print(file,item,(errors[item]-errors['all'])/errors['all'],errors['all'])
                