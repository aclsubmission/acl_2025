# tripadvisor_hotel_reviews.csv

# source ../../Other/joham/textbooks/virtenv/text/bin/activate
import os
import pickle
import html
import tqdm
import re
import csv
from nltk.tokenize import sent_tokenize

ROOT = '../sense_BERT/data/wikitext_all.pickle'
raw_sentence = []
dates = []
with open(ROOT, 'rb') as f:
    data = pickle.load(f)
    for category in tqdm.tqdm(data['en']):
        for article in data['en'][category]:
            text = article[3]
            text = html.unescape(text)    
            text = re.sub('<.*?>', '', text)
            sentences = sent_tokenize(text)
            for sentence in sentences:
                raw_sentence.append(sentence)
        


with open('raw_data/articles_wikipedia.csv', 'w') as f:
    writer = csv.writer(f)
    for sentence in raw_sentence:
        writer.writerow([sentence])
print(len(raw_sentence))


file = open('raw_stats/articles_wikipedia.txt', 'w')
file.write( ','  + ','+str(len(raw_sentence)))
file.close()