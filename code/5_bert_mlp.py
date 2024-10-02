import torch
import torch.nn as nn

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

# Assuming you have a pre-trained model
input_size = 768  # BERT embedding size
output_size = 10009  # Prediction variable length
model = BertMLP(input_size=input_size, output_size=output_size)

# Load the pre-trained weights
model.load_state_dict(torch.load('path_to_your_pretrained_model.pth'))

# Set the model to evaluation mode
model.eval()

# Assuming you have BERT embeddings to predict from
# This could be a single embedding or a batch of embeddings
bert_embeddings = torch.randn(5, 768)  # Example: 5 embeddings of size 768

# Perform prediction
with torch.no_grad():  # We don't need to calculate gradients for inference
    predictions = model(bert_embeddings)

# The predictions tensor now contains the output for each input embedding
# If you need probabilities, you can apply softmax
probabilities = torch.nn.functional.softmax(predictions, dim=1)

# If you need the most likely class for each prediction
_, predicted_classes = torch.max(predictions, dim=1)

print(f"Input shape: {bert_embeddings.shape}")
print(f"Predictions shape: {predictions.shape}")
print(f"Probabilities shape: {probabilities.shape}")
print(f"Predicted classes: {predicted_classes}")

# Example of how to use the predictions
for i, pred in enumerate(predicted_classes):
    print(f"Embedding {i+1}: Predicted class {pred.item()}")