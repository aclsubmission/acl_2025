import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random
import os
import pickle
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import time

ROOT = 'liwc_style/'

model_list = ['distilbert/distilbert-base-uncased', 'bert-base-uncased', 'bert-large-uncased', 'FacebookAI/roberta-base'][::-1]

for model_name in model_list:
    model_id = model_name.split('/')[-1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)

    files = os.listdir(ROOT)
    output = {}
    for file in files:
        print(file)
        if not os.path.exists(f'{model_id}_style/{os.path.splitext(file)[0]}.pickle'):
        
            sentences = []
            if not file.startswith('.'):
                with open(os.path.join(ROOT, file), 'rb') as handle:
                    liwc_style = pickle.load(handle)
                
                for item in tqdm.tqdm(liwc_style):
                    label = tokenizer(item[1], return_tensors='pt')
                    sentence = item[0].replace('[MASK]', tokenizer.mask_token)
                    
                    inputs = tokenizer(sentence, return_tensors='pt')
                    if len(inputs['input_ids'][0]) < 512:
                        inputs = inputs.to(device)
                        
                        # Get the index of the masked token
                        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

                        with torch.no_grad():
                            outputs = model(**inputs, output_hidden_states=True)
                            hidden_states = outputs.hidden_states

                        # Get the last hidden state
                        last_hidden_state = hidden_states[-1]
                        mask_token_hidden_state = last_hidden_state[0, mask_token_index, :]
                        label_id = label['input_ids'][0][1:-1].numpy()

                        sentences.append([sentence, label, mask_token_hidden_state.cpu().numpy(), label_id])

                if not os.path.exists(f'{model_id}_style/'):
                    os.makedirs(f'{model_id}_style/')                
                with open(f'{model_id}_style/{os.path.splitext(file)[0]}.pickle', 'wb') as f:
                    pickle.dump(sentences, f)