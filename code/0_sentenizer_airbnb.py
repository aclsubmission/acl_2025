# source ../../Other/joham/textbooks/virtenv/text/bin/activate
import os
import pickle
import html
import tqdm
import re
import csv
from nltk.tokenize import sent_tokenize

ROOT = '../Mental Health Padmini/Proposal/airbnb'
raw_sentence = []
dates = []
for file in tqdm.tqdm(os.listdir(ROOT)):
    if file.endswith('.csv'):
        data_set = {}
        with open(os.path.join(ROOT, file), 'r') as csvfile:
            reader = csv.reader(csvfile)
            line_count = 0
            for row in reader:
                
                if line_count>0:
                    if len(row)>58:
                        date = row[58]
                        date = date.split('-')[0]
                        if len(date)==4:
                            dates.append(int(date))

                    text = html.unescape(row[5].strip(' '))
                    text = re.sub('<.*?>', '', text)
                    sentences = sent_tokenize(text)
                    for sentence in sentences:
                        raw_sentence.append(sentence)
                line_count+=1
            # process each row of the CSV file
            # add your code here

    
import csv
if not os.path.exists('raw_data'):
    os.makedirs('raw_data')

with open('raw_data/descriptions_airbnb.csv', 'w') as f:
    writer = csv.writer(f)
    for sentence in raw_sentence:
        writer.writerow([sentence])
dates.sort()
print(min(dates), max(dates))
print(len(raw_sentence))

dates.sort()
file = open('raw_stats/descriptions_airbnb.txt', 'w')
file.write(str(min(dates)) + ',' + str(max(dates)) + ','+str(len(raw_sentence)))
