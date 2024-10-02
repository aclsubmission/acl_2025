# source ../../Other/joham/textbooks/virtenv/text/bin/activate
import os
import pickle
import html
import tqdm
import re
import csv
from nltk.tokenize import sent_tokenize

ROOT = '../Mental Health Padmini/lancesterLexicon/yelp_dataset/Lyrics/Hot_100.new.dump.csv'
raw_sentence = []
dates = []
data_set = {}
with open(ROOT, 'r') as csvfile:
    reader = csv.reader(csvfile)
    line_count = 0
    for row in tqdm.tqdm(reader):

        date = row[0]
        
        if len(date)==4:
            dates.append(int(date))

        text = html.unescape(row[3].strip(' ').replace('\n','. '))
        text = re.sub('<.*?>', '', text)
        sentences = sent_tokenize(text)
        for sentence in sentences:
            if len(sentence)>1:
                raw_sentence.append(sentence)
        line_count+=1
    # process each row of the CSV file
    # add your code here

    
import csv
if not os.path.exists('raw_data'):
    os.makedirs('raw_data')

with open('raw_data/lyrics_hot100.csv', 'w') as f:
    writer = csv.writer(f)
    for sentence in raw_sentence:
        writer.writerow([sentence])
dates.sort()
print(min(dates), max(dates))
print(len(raw_sentence))

dates.sort()
file = open('raw_stats/lyrics_hot100.txt', 'w')
file.write(str(min(dates)) + ',' + str(max(dates)) + ','+str(len(raw_sentence)))
