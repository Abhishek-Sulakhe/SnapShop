"""
Dataset classes for the Shopee training pipeline.

ImageDataset : image-only data for DeiT / EfficientNet metric learning.
TextDataset  : text-only data for BERT / XLM-RoBERTa metric learning.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class ImageDataset(Dataset):
    """Dataset for image-only metric learning training.

    Returns (image_tensor, label) pairs.
    Images that fail to load are replaced with a blank 512x512 placeholder
    so training never crashes on a corrupt file.
    """

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
    """Dataset for text-only metric learning training.

    Returns (title_string, label) pairs.
    Tokenization is handled inside the model's extract_feat() method
    so that dynamic padding works correctly per batch.
    """

    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = torch.tensor(row['label'], dtype=torch.long)
        return row['title'], label

    def __len__(self):
        return len(self.df)
