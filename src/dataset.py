import torch
import kagglehub
import os
from torch.utils.data import Dataset, DataLoader

class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_length, tokenizer):
        self.tokenizer = tokenizer
        self.data = tokenizer.encode(text)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # chunk setup: input (x) and target (y) are shifted by 1
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class Tokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

def download_data():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("adarshpathak/shakespeare-text")
    print("Path to dataset files:", path)
    # Find the text file
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("txt"): # Assuming it's a text file
                return os.path.join(root, file)
    return None

def get_dataloader(batch_size, seq_length):
    file_path = download_data()
    if not file_path:
        raise FileNotFoundError("Could not find dataset file")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = Tokenizer(text)
    dataset = ShakespeareDataset(text, seq_length, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, tokenizer

if __name__ == "__main__":
    dataloader, tokenizer = get_dataloader(batch_size=4, seq_length=10)
    print("Data loaded successfully")
    print("Vocab size:", tokenizer.vocab_size)
    print("First batch:", next(iter(dataloader)))
