import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from models import MultimodalModel
from dataset import MultimodalDataset

# Load processed data
X_test = pd.read_csv('X_test.csv')

def evaluate_saved_model(model_path, test_dataset, batch_size=2):
    model = MultimodalModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            measurements = batch['measurements']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label'].unsqueeze(1)

            outputs = model(measurements, input_ids, attention_mask)

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    metrics = {
        'auc_roc': roc_auc_score(all_labels, all_preds),
        'auc_pr': average_precision_score(all_labels, all_preds),
        'predictions': all_preds,
        'labels': all_labels
    }

    return metrics

def zero_shot_evaluation_saved(model_path, test_dataset, tokenizer):
    model = MultimodalModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    phrases = {
        "positive": "patient deceased",
        "negative": "discharged today"
    }

    encodings = {}
    for key, phrase in phrases.items():
        encodings[key] = tokenizer(
            phrase,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    all_preds, all_labels = [], []

    with torch.no_grad():
        text_encoder = model.text_encoder
        pos_emb = text_encoder(encodings["positive"]['input_ids'],
                              encodings["positive"]['attention_mask'])
        neg_emb = text_encoder(encodings["negative"]['input_ids'],
                              encodings["negative"]['attention_mask'])

        for batch in test_loader:
            measurements = batch['measurements']
            labels = batch['label']

            meas_emb = model.measurement_encoder(measurements)

            pos_sim = torch.nn.functional.cosine_similarity(
                meas_emb, pos_emb.expand(meas_emb.size(0), -1), dim=1)
            neg_sim = torch.nn.functional.cosine_similarity(
                meas_emb, neg_emb.expand(meas_emb.size(0), -1), dim=1)

            probs = torch.nn.functional.softmax(
                torch.stack([neg_sim, pos_sim], dim=1), dim=1)

            all_preds.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        'auc_roc': roc_auc_score(all_labels, all_preds),
        'auc_pr': average_precision_score(all_labels, all_preds),
        'predictions': all_preds,
        'labels': all_labels
    }

    return metrics

def plot_curves(results):
    y_pred = results['predictions']
    y_true = results['labels']

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, label=f'AUC-ROC = {results["auc_roc"]:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    axes[1].plot(recall, precision, label=f'AUC-PR = {results["auc_pr"]:.4f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('performance_curves.png')
    plt.show()

def compare_with_paper(our_results):
    paper_results = {
        'supervised': {'auc_roc': 0.856, 'auc_roc_std': 0.004, 'auc_pr': 0.495, 'auc_pr_std': 0.005},
        'zero_shot': {'auc_roc': 0.709, 'auc_pr': 0.214}
    }

    print("| Evaluation | AUC-ROC | Paper AUC-ROC | AUC-PR | Paper AUC-PR |")
    print("|------------|---------|---------------|--------|-------------|")

    for eval_type in ['supervised', 'zero_shot']:
        display_name = "Supervised" if eval_type == 'supervised' else "Zero-shot"

        our_roc = our_results[eval_type]['auc_roc']
        our_pr = our_results[eval_type]['auc_pr']
        paper_roc = paper_results[eval_type]['auc_roc']
        paper_pr = paper_results[eval_type]['auc_pr']

        print(f"| {display_name:<11} | {our_roc:.4f} | {paper_roc:.4f} | {our_pr:.4f} | {paper_pr:.4f} |")

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create test dataset
test_dataset = MultimodalDataset(
    dataframe=X_test,
    tokenizer=tokenizer,
    max_length=64
)

# Evaluate model
model_path = 'multimodal_model.pt'
supervised_results = evaluate_saved_model(model_path, test_dataset)
zero_shot_results = zero_shot_evaluation_saved(model_path, test_dataset, tokenizer)

# Plot results
plot_curves(supervised_results)

# Compare with paper results
results = {
    'supervised': supervised_results,
    'zero_shot': zero_shot_results
}
compare_with_paper(results)

print("Evaluation complete. Results saved to performance_curves.png")
