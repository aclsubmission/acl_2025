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
BATCH_SIZE = 32

model_list = ['distilbert/distilbert-base-uncased', 'bert-base-uncased', 'bert-large-uncased', 'FacebookAI/roberta-base']

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
                
                batches = [liwc_style[i:i + BATCH_SIZE] for i in range(0, len(liwc_style), BATCH_SIZE)]
                
                for batch in tqdm.tqdm(batches):
                    batch_sentences = []
                    batch_labels = []
                    batch_inputs = []
                    
                    for item in batch:
                        label = tokenizer(item[1], return_tensors='pt')
                        sentence = item[0].replace('[MASK]', tokenizer.mask_token)
                        
                        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
                        
                        batch_sentences.append(sentence)
                        batch_labels.append(label)
                        batch_inputs.append(inputs)
                    
                    # Pad and stack inputs
                    padded_input_ids = torch.nn.utils.rnn.pad_sequence([inputs['input_ids'][0] for inputs in batch_inputs], batch_first=True)
                    padded_attention_mask = torch.nn.utils.rnn.pad_sequence([inputs['attention_mask'][0] for inputs in batch_inputs], batch_first=True)
                    
                    batch_inputs = {
                        'input_ids': padded_input_ids.to(device),
                        'attention_mask': padded_attention_mask.to(device)
                    }
                    
                    with torch.no_grad():
                        outputs = model(**batch_inputs, output_hidden_states=True)
                        hidden_states = outputs.hidden_states
                    
                    # Get the last hidden state
                    last_hidden_state = hidden_states[-1]
                    
                    for i, (sentence, label, inputs) in enumerate(zip(batch_sentences, batch_labels, batch_inputs['input_ids'])):
                        mask_token_index = torch.where(inputs == tokenizer.mask_token_id)[0]
                        if len(mask_token_index) > 0:  # Check if mask token exists
                            mask_token_hidden_state = last_hidden_state[i, mask_token_index, :]
                            label_id = label['input_ids'][0][1:-1].cpu().numpy()
                            sentences.append([sentence, label, mask_token_hidden_state.cpu().numpy(), label_id])

                if not os.path.exists(f'{model_id}_style/'):
                    os.makedirs(f'{model_id}_style/')                
                with open(f'{model_id}_style/{os.path.splitext(file)[0]}.pickle', 'wb') as f:
                    pickle.dump(sentences, f)