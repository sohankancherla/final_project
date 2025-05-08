import torch
import torch.nn as nn
from transformers import BertModel

class MeasurementEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, hidden_dim=32, freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.projection = nn.Linear(768, hidden_dim)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = bert_output.last_hidden_state[:, 0, :]
        return self.projection(cls_token)

class MultimodalModel(nn.Module):
    def __init__(self, measurement_dim=1, hidden_dim=32):
        super(MultimodalModel, self).__init__()

        self.measurement_encoder = MeasurementEncoder(measurement_dim, hidden_dim)
        self.text_encoder = TextEncoder(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, measurements, input_ids, attention_mask):
        measurement_features = self.measurement_encoder(measurements)
        text_features = self.text_encoder(input_ids, attention_mask)

        combined_features = torch.cat([measurement_features, text_features], dim=1)

        return self.classifier(combined_features)

class MultimodalModelWithDropout(nn.Module):
    def __init__(self, measurement_dim=1, hidden_dim=32, dropout_prob=0.5):
        super().__init__()
        self.measurement_encoder = MeasurementEncoder(measurement_dim, hidden_dim)
        self.text_encoder = TextEncoder(hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

        self.dropout_prob = dropout_prob

        self.missing_measurement_token = nn.Parameter(torch.randn(hidden_dim))
        self.missing_text_token = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, measurements, input_ids, attention_mask, training=True):
        drop_measurement = False
        drop_text = False

        if training:
            drop_measurement = torch.rand(1).item() < self.dropout_prob
            drop_text = torch.rand(1).item() < self.dropout_prob

            if drop_measurement and drop_text:
                drop_measurement = torch.rand(1).item() < 0.5
                drop_text = not drop_measurement

        batch_size = measurements.size(0)

        measurement_emb = (
            self.missing_measurement_token.expand(batch_size, -1)
            if drop_measurement
            else self.measurement_encoder(measurements)
        )

        text_emb = (
            self.missing_text_token.expand(batch_size, -1)
            if drop_text
            else self.text_encoder(input_ids, attention_mask)
        )

        combined = torch.cat([measurement_emb, text_emb], dim=1)
        return self.sigmoid(self.classifier(combined))
