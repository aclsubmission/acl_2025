# source ../../Other/joham/textbooks/virtenv/text/bin/activate
import os
import pickle
import html
import tqdm
import re
import csv
from nltk.tokenize import sent_tokenize
import json



raw_sentence = []
dates = []
data_set = {}
with open('../Mental Health Padmini/lancesterLexicon/yelp_dataset/yelp_academic_dataset_review.json','r',encoding='utf-8') as f:
    for i, line in tqdm.tqdm(enumerate(f)):

        file=json.loads(line)
        if len(file['text'])>2:
            date = file['date']
            date=date.split('-')[0]
            
            if len(date)==4:
                dates.append(int(date))

            text = html.unescape(file['text'].strip(' ').replace('\n',' '))
            text = re.sub(' +', ' ', text)
            text = re.sub('<.*?>', '', text)
            sentences = sent_tokenize(text)
            for sentence in sentences:
                if len(sentence)>1:
                    raw_sentence.append(sentence)
            
    # process each row of the CSV file
    # add your code here

    
import csv
if not os.path.exists('raw_data'):
    os.makedirs('raw_data')

with open('raw_data/review_yelp.csv', 'w') as f:
    writer = csv.writer(f)
    for sentence in raw_sentence:
        writer.writerow([sentence])
dates.sort()
print(min(dates), max(dates))
print(len(raw_sentence))

dates.sort()
file = open('raw_stats/review_yelp.txt', 'w')
file.write(str(min(dates)) + ',' + str(max(dates)) + ','+str(len(raw_sentence)))
