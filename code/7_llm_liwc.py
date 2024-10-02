import pickle
import numpy as np
import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BertMLP(nn.Module):
    def __init__(self, input_size=768, hidden_size=512, output_size=10009):
        super(BertMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
    def forward(self, x):
        return self.layers(x)

class CustomDataset(Dataset):
    def __init__(self, encoder_embedding, style_embedding, y_map, R, type='clean'):
        self.encoder_embedding = encoder_embedding
        self.style_embedding = style_embedding
        self.y_map = y_map
        self.R = R
        self.type = type

    def __len__(self):
        return len(self.encoder_embedding)

    def __getitem__(self, idx):
        x1 = np.array(self.encoder_embedding[idx][2][:self.R], dtype=np.float32)
        if self.type == 'dirty':
            x2 = np.array(self.style_embedding[idx][2], dtype=np.float32)
            if sum(x2)!=0:
                x2 = x2/np.linalg.norm(x2)
            x1 = x1/np.linalg.norm(x1)
            x = np.concatenate((x1, x2), axis=0)
        else:
            x = x1
        
        label = self.encoder_embedding[idx][3][0]
        y = np.zeros(len(self.y_map), dtype=np.float32)
        y[self.y_map[label]] = 1
        
        return torch.from_numpy(x), torch.from_numpy(y)

def train_and_evaluate(encoder_embedding, style_embedding, R, type):
    y_map = {}
    c = 0
    for i in range(len(encoder_embedding)):
        label = encoder_embedding[i][3][0]
        if label not in y_map:
            y_map[label] = c
            c += 1

    dataset = CustomDataset(encoder_embedding, style_embedding, y_map, R, type)
    split_index = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(split_index))
    test_dataset = torch.utils.data.Subset(dataset, range(split_index, len(dataset)))
    
    # Hyperparameters
    input_size = R if type == 'clean' else R + len(style_embedding[0][2])
    output_size = len(y_map)
    hidden_size = 512
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # Create the model and move it to GPU
    model = BertMLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader for training data
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader):.4f}")

    # Evaluation on test set
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(torch.argmax(batch_y, dim=1).cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"R: {R}, Accuracy: {accuracy}")

    # Clear GPU memory
    del model
    torch.cuda.empty_cache()

    return accuracy

# Main execution
model_list = ['bert-base-uncased_style/','distilbert-base-uncased_style/',  'bert-large-uncased_style/', 'roberta-base_style/']
data_list = ['articles_wikipedia','descriptions_airbnb','gutenberg_domesticfiction','review_yelp','lyrics_hot100']
ROOT = 'results_dims/'
RESULTS = {}
for model in model_list:
    for data in data_list:
        
        if os.path.isfile(ROOT+model.replace('/','')+'_'+data+'.pickle')==False:
            NEW_ROOT = './liwc_dimensions/' 
            try:
                with open('SLIM_LLM/'+model+'/'+data+'.csv_svd.pickle', 'rb') as handle:
                    encoder_embedding = pickle.load(handle) 
            except:
                data = data.replace('lyrics','yrics')
                with open('SLIM_LLM/'+model+'/'+data+'.csv_svd.pickle', 'rb') as handle:
                    encoder_embedding = pickle.load(handle) 
            for categories in os.listdir(NEW_ROOT):
                models = os.listdir(NEW_ROOT + categories+'/')
                for liwc_ in models:
                    if data in liwc_:
                        file = NEW_ROOT+categories+'/'+liwc_
                        # Load data
                        
                        with open(file, 'rb') as handle:
                            style_embedding = pickle.load(handle)
            
                print(model, data,categories)
                
                CUTS = [80,160,240]
                if model not in RESULTS:
                    RESULTS[model] = {}
                if data not in RESULTS[model]:
                    RESULTS[model][data] = {}

                for type in ['dirty']:
                    
                    
                    for R in tqdm.tqdm(CUTS):
                        accuracy = train_and_evaluate(encoder_embedding, style_embedding, R, type)
                        if categories not in RESULTS[model][data]:
                            RESULTS[model][data][categories] = {}
                        RESULTS[model][data][categories][R] = accuracy

with open(ROOT+'data.pickle', 'wb') as handle:
    pickle.dump(RESULTS, handle)