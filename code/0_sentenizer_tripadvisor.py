# tripadvisor_hotel_reviews.csv

# source ../../Other/joham/textbooks/virtenv/text/bin/activate
import os
import pickle
import html
import tqdm
import re
import csv
from nltk.tokenize import sent_tokenize

ROOT = '../Mental Health Padmini/Proposal/tripadvisor_hotel_reviews.csv'
raw_sentence = []
dates = []
with open(os.path.join(ROOT), 'r') as csvfile:
    reader = csv.reader(csvfile)
    line_count = 0
    for row in reader:
        if line_count>0:    
            
                
            text = html.unescape(row[0].strip(' '))
            text = re.sub('<.*?>', '', text)
            sentences = sent_tokenize(text)
            for sentence in sentences:
                raw_sentence.append(sentence)
        line_count += 1


with open('raw_data/reviews_tripadvisor_hotel.csv', 'w') as f:
    writer = csv.writer(f)
    for sentence in raw_sentence:
        writer.writerow([sentence])
print(len(raw_sentence))


file = open('raw_stats/reviews_tripadvisor_hotel.txt', 'w')
file.write( ','  + ','+str(len(raw_sentence)))
