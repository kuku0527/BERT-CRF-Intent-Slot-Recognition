import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class IntentSlotDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=64):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                text, intent, slots = line.split('\t')
                self.samples.append((text, intent, slots.split()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, intent, slots = self.samples[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, intent, slots
