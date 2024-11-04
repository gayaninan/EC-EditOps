from torch.utils.data import Dataset

class ECTextDataset(Dataset):
    def __init__(self, tokenizer, texts, block_size):
        self.tokenizer = tokenizer
        self.texts = texts
        self.block_size = block_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])  # Convert to string
        tokenized_text = self.tokenizer(text, add_special_tokens=True, truncation=True, max_length=self.block_size)
        return tokenized_text
