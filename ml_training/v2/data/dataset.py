import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class ImageDataset(Dataset):
    """Dataset for image-only training (DeiT, EfficientNet)."""

    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = self.img_dir / row['image']

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (512, 512))

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(row['label'], dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.df)


class TextDataset(Dataset):
    """Dataset for text-only training (BERT, XLM-RoBERTa)."""

    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = torch.tensor(row['label'], dtype=torch.long)
        return row['title'], label

    def __len__(self):
        return len(self.df)


class MultiModalDataset(Dataset):
    """Dataset for multimodal training (DeiT + BERT)."""

    def __init__(self, df, img_dir, tokenizer, transform=None, max_len=128):
        self.df = df
        self.img_dir = Path(img_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Image
        img_path = self.img_dir / row['image']
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (512, 512))

        if self.transform is not None:
            img = self.transform(img)

        # Text
        text = str(row['title'])
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
        )
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        label = torch.tensor(row['label'], dtype=torch.long)
        return img, input_ids, attention_mask, label

    def __len__(self):
        return len(self.df)
