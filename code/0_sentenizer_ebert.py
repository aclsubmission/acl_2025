# source ../../Other/joham/textbooks/virtenv/text/bin/activate
import os
import pickle
import html
import tqdm

import csv
from nltk.tokenize import sent_tokenize

ROOT = '../Mental Health Padmini/Proposal/'

file = 'reviewDataSet.pickle'
source_str = 'RogerEbert.com'
genre_str = 'Reviews'
data_set = pickle.load(open(ROOT+file, "rb"))
l=0
raw_sentence = []
dates = []
for url in tqdm.tqdm(data_set):
    url = url
    title_text = data_set[url][0]
    date_text = data_set[url][1]
    date = date_text.split(' ')[-1]
    if len(date)==4:
        dates.append(int(date))

    text = html.unescape(data_set[url][2].strip(' '))
    sentences = sent_tokenize(text)
    for sentence in sentences:
        raw_sentence.append(sentence)
    
import csv
if not os.path.exists('raw_data'):
    os.makedirs('raw_data')

if not os.path.exists('raw_stats'):
    os.makedirs('raw_stats')

with open('raw_data/reviews_ebert.csv', 'w') as f:
    writer = csv.writer(f)
    for sentence in raw_sentence:
        writer.writerow([sentence])
dates.sort()
file = open('raw_stats/reviews_ebert.txt', 'w')
file.write(str(min(dates)) + ',' + str(max(dates)) + ','+str(len(raw_sentence)))
