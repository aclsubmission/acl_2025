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
file='../Mental Health Padmini/lancesterLexicon/yelp_dataset/Gutenberg/2514.csv'    
with open(file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        d = row[1].strip(' ')
        if '-' in d:
            date_start = d.split('-')[0]
            date_end = d.split('-')[1]
            if len(date_start)==4 and len(date_end)==4:
                dates.append((int(date_start)+int(date_end))//2)


root='../Mental Health Padmini/lancesterLexicon/yelp_dataset/Gutenberg/2514/'
files=os.listdir(root)


for name in tqdm.tqdm(files):

    f=root+name
    if f.find('.txt')>0:    
        bookValid=[]
        book=open(f,'r').read().split('\n\n\n')
        for i in range(0,len(book)):
            if len(book[i])>1000:
                text=book[i].replace('\n',' ')
                text = re.sub(' +', ' ', text)
                bookValid.append(text)

        text = html.unescape(' '.join(bookValid))
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

with open('raw_data/gutenberg_domesticfiction.csv', 'w') as f:
    writer = csv.writer(f)
    for sentence in raw_sentence:
        writer.writerow([sentence])
dates.sort()
print(min(dates), max(dates))
print(len(raw_sentence))

dates.sort()
file = open('raw_stats/gutenberg_domesticfiction.txt', 'w')
file.write(str(min(dates)) + ',' + str(max(dates)) + ','+str(len(raw_sentence)))
file.close()
