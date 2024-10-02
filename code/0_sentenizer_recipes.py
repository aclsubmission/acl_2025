# tripadvisor_hotel_reviews.csv

# source ../../Other/joham/textbooks/virtenv/text/bin/activate
import os
import pickle
import html
import tqdm
import re
import csv
from nltk.tokenize import sent_tokenize

FILES = ['../chatGPT/FINAL/food/data/recipes1_en.pickle','../chatGPT/FINAL/food/data/recipes2_en.pickle']

raw_sentence = []
dates = []
for ROOT in FILES:
    with open(ROOT, 'rb') as f:
        data = pickle.load(f)
        for recipe in tqdm.tqdm(data['en']):
            
            text = recipe[1].split('<div dir="auto" class="mb-sm')[1:]
            if '"datePublished":"' in recipe[1]:
                date = recipe[1].split('"datePublished":"')[1].split('-')[0]
                if len(date)==4:
                    dates.append(int(date))
            instructions = []
            for t in text:
                inst = t.split('<p')[1].split('</p>')[0].split('">')[1]

                instructions.append(inst)
            text = '. '.join(instructions)
        
            text = html.unescape(text)    
            text = re.sub('<.*?>', '', text)
            sentences = sent_tokenize(text)
            for sentence in sentences:
                raw_sentence.append(sentence)
    

dates = list(set(dates))
with open('raw_data/articles_recipes.csv', 'w') as f:
    writer = csv.writer(f)
    for sentence in raw_sentence:
        writer.writerow([sentence])
print(len(raw_sentence))


file = open('raw_stats/articles_recipes.txt', 'w')
file.write(str(min(dates)) + ',' + str(max(dates)) + ','+str(len(raw_sentence)))
print(str(min(dates)) + ',' + str(max(dates)) + ','+str(len(raw_sentence)))
