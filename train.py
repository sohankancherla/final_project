import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from models import MultimodalModel
from dataset import MultimodalDataset

# Load processed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create datasets
train_dataset = MultimodalDataset(
    dataframe=X_train,
    tokenizer=tokenizer,
    max_length=64
)

test_dataset = MultimodalDataset(
    dataframe=X_test,
    tokenizer=tokenizer,
    max_length=64
)

# Create data loaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=2,
    shuffle=False
)

# Initialize model
model = MultimodalModel()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(
    params=model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in train_loader:
            measurements = batch['measurements']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label'].unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(measurements, input_ids, attention_mask)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

    return model

# Train the model
model = train_model(model, train_loader, criterion, optimizer, 10)

# Save the model
torch.save(model.state_dict(), 'multimodal_model.pt')
print("Model training complete. Saved as multimodal_model.pt")
