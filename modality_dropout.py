import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score

from models import MultimodalModel, MultimodalModelWithDropout
from dataset import MultimodalDataset

# Load processed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')

def train_model_with_dropout(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    history = {'loss': []}

    for epoch in range(num_epochs):
        running_loss = 0.0
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

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        history['loss'].append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return model, history

def evaluate_modality_settings(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    results = {}

    model.eval()

    # Evaluate with both modalities
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            measurements = batch['measurements']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label'].unsqueeze(1)

            outputs = model(measurements, input_ids, attention_mask, training=False)

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    roc = roc_auc_score(all_labels, all_preds)
    pr = average_precision_score(all_labels, all_preds)

    results['both'] = {'auc_roc': roc, 'auc_pr': pr}
    print(f"Both modalities - AUC-ROC: {roc:.4f}, AUC-PR: {pr:.4f}")

    # Evaluate with only measurements
    print("\nEvaluating with only measurements...")
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            measurements = batch['measurements']
            input_ids = batch['input_ids']
            labels = batch['label'].unsqueeze(1)

            batch_size = input_ids.size(0)
            text_emb = model.missing_text_token.expand(batch_size, -1)
            measurement_emb = model.measurement_encoder(measurements)

            combined = torch.cat([measurement_emb, text_emb], dim=1)
            output = model.sigmoid(model.classifier(combined))

            all_preds.extend(output.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    roc = roc_auc_score(all_labels, all_preds)
    pr = average_precision_score(all_labels, all_preds)

    results['measurements_only'] = {'auc_roc': roc, 'auc_pr': pr}
    print(f"Measurements only - AUC-ROC: {roc:.4f}, AUC-PR: {pr:.4f}")

    # Evaluate with only text
    print("\nEvaluating with only text...")
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            measurements = batch['measurements']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label'].unsqueeze(1)

            batch_size = measurements.size(0)
            measurement_emb = model.missing_measurement_token.expand(batch_size, -1)
            text_emb = model.text_encoder(input_ids, attention_mask)

            combined = torch.cat([measurement_emb, text_emb], dim=1)
            output = model.sigmoid(model.classifier(combined))

            all_preds.extend(output.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    roc = roc_auc_score(all_labels, all_preds)
    pr = average_precision_score(all_labels, all_preds)

    results['text_only'] = {'auc_roc': roc, 'auc_pr': pr}
    print(f"Text only - AUC-ROC: {roc:.4f}, AUC-PR: {pr:.4f}")

    return results

def plot_modality_comparison(results):
    settings = ['both', 'measurements_only', 'text_only']
    settings_labels = ['Both Modalities', 'Measurements Only', 'Text Only']

    roc_values = [results[s]['auc_roc'] for s in settings]
    pr_values = [results[s]['auc_pr'] for s in settings]

    x = np.arange(len(settings))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, roc_values, width, label='AUC-ROC')
    rects2 = ax.bar(x + width/2, pr_values, width, label='AUC-PR')

    ax.set_ylabel('Score')
    ax.set_title('Performance by Modality Setting')
    ax.set_xticks(x)
    ax.set_xticklabels(settings_labels)
    ax.legend()

    for rects in (rects1, rects2):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig('modality_comparison.png')
    plt.show()

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create datasets
train_dataset = MultimodalDataset(X_train, tokenizer, max_length=64)
test_dataset = MultimodalDataset(X_test, tokenizer, max_length=64)

# Create data loader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Initialize model with dropout
model = MultimodalModelWithDropout(dropout_prob=0.3)

# Load weights from original model
original_model = MultimodalModel()
original_model.load_state_dict(torch.load('multimodal_model.pt'))

# Copy weights from original model to dropout model
for target, source in zip([model.measurement_encoder, model.text_encoder],
                         [original_model.measurement_encoder, original_model.text_encoder]):
    target.load_state_dict(source.state_dict())

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Train model with dropout
model, history = train_model_with_dropout(model, train_loader, criterion, optimizer, num_epochs=5)

# Save model
torch.save(model.state_dict(), 'multimodal_model_with_dropout.pt')

# Evaluate model with different modality settings
modality_results = evaluate_modality_settings(model, test_dataset)

# Plot modality comparison
plot_modality_comparison(modality_results)

print("Modality dropout extension complete. Results saved to modality_comparison.png")
