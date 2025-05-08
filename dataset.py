import torch
from torch.utils.data import Dataset, DataLoader

class MultimodalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_char_limit = max_length * 10

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        measurements = torch.tensor([row['VALUENUM']], dtype=torch.float32)
        text = str(row['TEXT'])[:self.text_char_limit]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        label = torch.tensor(row['IN_HOSPITAL_MORTALITY'], dtype=torch.float32)

        return {
            'measurements': measurements,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }
