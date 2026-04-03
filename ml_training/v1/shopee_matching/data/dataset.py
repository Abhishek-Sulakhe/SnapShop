import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
from pathlib import Path

class ShopeeDataset(Dataset):

    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = self.img_dir / row['image']
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            img = Image.new('RGB', (512, 512))

        w, h = img.size
        # Handle file stats if file exists
        try:
            st_size = img_path.stat().st_size
        except:
            st_size = 0
            
        if self.transform is not None:
            img = self.transform(img)

        if 'label' in row:
            target = torch.tensor(row['label'], dtype=torch.long)
            return img, target

        return img, row['title'], h, w, st_size

    def __len__(self):
        return len(self.df)


class BertDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]

        if 'label' in row:
            target = torch.tensor(row['label'], dtype=torch.long)
            return row['title'], target
            
        if 'y' in row.keys():
            target = torch.tensor(row['y'], dtype=torch.long)
            return row['title'], target
        else:
            return row['title']

    def __len__(self):
        return len(self.df)


class GraphDataset(Dataset):

    def __init__(self, feats=None, labels=None, weights=None, pair_tuples=None, k=50, top_neighbors=None):
        self.feats = feats
        self.labels = labels
        self.weights = weights
        self.pair_tuples = pair_tuples
        self.k = k
        self.top_neighbors = top_neighbors

    def __getitem__(self, index):
        i, j = self.pair_tuples[index]
        feat = torch.FloatTensor(self.feats[i][j])

        padding_i = [[0] * feat.shape[0]] * (self.k - len(self.top_neighbors[i]))
        neighbor_feats_i = torch.FloatTensor([
            self.feats[i][neighbor]
            for neighbor in self.top_neighbors[i]
        ] + padding_i)
        padding_j = [[0] * feat.shape[0]] * (self.k - len(self.top_neighbors[j]))
        neighbor_feats_j = torch.FloatTensor([
            self.feats[j][neighbor]
            for neighbor in self.top_neighbors[j]
        ] + padding_j)
        neighbor_feats = torch.cat([feat.unsqueeze(0), neighbor_feats_i, neighbor_feats_j], dim=0)

        outputs = (feat, neighbor_feats)
        if self.labels is not None:
            outputs += (self.labels[i] == self.labels[j],)
        if self.weights is not None:
            outputs += (self.weights[i],)

        return outputs

    def __len__(self):
        return len(self.pair_tuples)


class MultiModalDataset(Dataset):
    def __init__(self, df, img_dir, tokenizer, transform=None, max_len=50):
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
        except:
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
            truncation=True
        )
        # return_tensors='pt' returns batch dim 1
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        
        # Label
        if 'label' in row:
            target = torch.tensor(row['label'], dtype=torch.long)
            return img, input_ids, attention_mask, target
            
        return img, input_ids, attention_mask

    def __len__(self):
        return len(self.df)
