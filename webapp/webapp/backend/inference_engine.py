import sys
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Resize, Normalize, Compose, CenterCrop
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

try:
    import timm
except ImportError:
    print("Warning: timm not found. Install with: pip install timm")

try:
    from transformers import BertConfig, BertModel, BertTokenizerFast, AutoTokenizer, AutoModel, AutoConfig
except ImportError:
    print("Warning: transformers not found.")

try:
    import faiss
except ImportError:
    print("Warning: faiss not found.")


# =============================================================================
# MODEL DEFINITIONS — Exact match to kaggler's lyakaap solution
# =============================================================================

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class ShopeeNet(nn.Module):
    def __init__(self, backbone, num_classes, fc_dim=512, s=30, margin=0.5, p=3):
        super(ShopeeNet, self).__init__()
        self.backbone = backbone
        self.backbone.reset_classifier(num_classes=0)
        self.fc = nn.Linear(self.backbone.num_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone.forward_features(x)

        if isinstance(x, tuple):
            x = (x[0] + x[1]) / 2
            x = self.bn(x)
        elif len(x.shape) == 3:
            # DeiT 3D output (B, N, C) in newer timm
            x = (x[:, 0] + x[:, 1]) / 2
            x = self.fc(x)
            x = self.bn(x)
        else:
            x = gem(x, p=self.p).view(batch_size, -1)
            x = self.fc(x)
            x = self.bn(x)
        return x


class MultiModalNet(nn.Module):
    def __init__(self, backbone, bert_model, num_classes, tokenizer,
                 max_len=32, fc_dim=512, p=3):
        super().__init__()
        self.backbone = backbone
        self.backbone.reset_classifier(num_classes=0)
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fc = nn.Linear(self.bert_model.config.hidden_size + self.backbone.num_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, img, title):
        batch_size = img.shape[0]
        img = self.backbone.forward_features(img)
        if len(img.shape) == 3:
            img = (img[:, 0] + img[:, 1]) / 2
        else:
            img = gem(img, p=self.p).view(batch_size, -1)

        device = img.device
        tokenizer_output = self.tokenizer(title, truncation=True, padding=True, max_length=self.max_len)
        input_ids = torch.LongTensor(tokenizer_output['input_ids']).to(device)
        token_type_ids = torch.LongTensor(tokenizer_output['token_type_ids']).to(device)
        attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to(device)

        title_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
        title_out = title_out.last_hidden_state.mean(dim=1)

        x = torch.cat([img, title_out], dim=1)
        x = self.fc(x)
        x = self.bn(x)
        return x


class BertNet(nn.Module):
    def __init__(self, bert_model, num_classes, tokenizer,
                 max_len=32, fc_dim=512, simple_mean=True, p=3):
        super().__init__()
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fc = nn.Linear(self.bert_model.config.hidden_size, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.simple_mean = simple_mean

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, x):
        tokenizer_output = self.tokenizer(x, truncation=True, padding=True, max_length=self.max_len)
        device = self.fc.weight.device

        if 'token_type_ids' in tokenizer_output:
            input_ids = torch.LongTensor(tokenizer_output['input_ids']).to(device)
            token_type_ids = torch.LongTensor(tokenizer_output['token_type_ids']).to(device)
            attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to(device)
            x = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        else:
            input_ids = torch.LongTensor(tokenizer_output['input_ids']).to(device)
            attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to(device)
            x = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)

        if self.simple_mean:
            x = x.last_hidden_state.mean(dim=1)
        else:
            x = torch.sum(x.last_hidden_state * attention_mask.unsqueeze(-1), dim=1) \
                / attention_mask.sum(dim=1, keepdims=True)

        x = self.fc(x)
        x = self.bn(x)
        return x


# =============================================================================
# SEARCH ENGINE — Uses separate indices for text/image/combined search
# =============================================================================

class SearchEngine:
    def __init__(self, data_dir, checkpoint_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_dir = self.checkpoint_dir

        # Data
        self.train_csv_path = self.data_dir / 'train.csv'
        self.train_img_dir = self.data_dir / 'train_images'
        if self.train_csv_path.exists():
            self.df = pd.read_csv(self.train_csv_path)
            print(f"Loaded {len(self.df)} products from train.csv")
        else:
            print(f"Warning: train.csv not found at {self.train_csv_path}")
            self.df = pd.DataFrame()

        # Models
        self.image_models = {}
        self.bert_models = {}
        self.model_params = {}
        self.transforms = None

        # Embeddings — stored separately for correct modality-specific search
        self.img_feats = None
        self.bert_feats = None
        self.combined_feats = None

        # FAISS indices — one per modality
        self.img_index = None
        self.bert_index = None
        self.combined_index = None

        # Cache
        self.cache_dir = Path(__file__).resolve().parent.parent / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._load_models()
        self._load_or_compute_embeddings()

    # ---- Path Resolution ----
    def _resolve_path(self, filename):
        path = self.model_dir / filename
        if path.is_file():
            return path
        if path.is_dir():
            nested = path / filename
            if nested.is_file():
                return nested
            pth_files = list(path.glob('*.pth'))
            if pth_files:
                return pth_files[0]
        matches = [m for m in self.model_dir.rglob(filename) if m.is_file()]
        if matches:
            return matches[0]
        return path

    # ---- Model Loading (matches kaggler exactly) ----
    def _load_models(self):
        print(f"Loading models on {self.device}...")

        # Image models
        self._load_image_model('deit_small.pth', 'deit_small')
        self._load_image_model('efficientnet_b3.pth', 'efficientnet_b3')

        # BERT Indonesian — simple_mean=True
        self._load_bert_model('bert_indonesian.pth', 'bert_indonesian', 'bert-indonesian',
                              simple_mean=True)

        # BERT Multilingual — simple_mean=False, uses bert_indonesian's fc_dim/max_len
        self._load_bert_model('bert_multilingual.pth', 'bert_multilingual', 'bert-multilingual',
                              simple_mean=False, override_params_from='bert_indonesian')

        # XLM-RoBERTa — simple_mean=False, uses its own params
        self._load_bert_model('xlm_roberta.pth', 'xlm_roberta', 'xlm-roberta', simple_mean=False)

        print(f"Loaded image models: {list(self.image_models.keys())}")
        print(f"Loaded BERT models: {list(self.bert_models.keys())}")

    def _load_image_model(self, filename, key):
        path = self._resolve_path(filename)
        if not path.exists():
            print(f"  [MISSING] {filename}")
            return

        print(f"  Loading {filename}...")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        params = ckpt['params']
        self.model_params[key] = params

        if self.transforms is None:
            self.transforms = Compose([
                Resize(size=params['test_size'] + 32, interpolation=Image.BICUBIC),
                CenterCrop((params['test_size'], params['test_size'])),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        backbone_name = params['backbone']
        if backbone_name.startswith('vit_deit_'):
            backbone_name = backbone_name.replace('vit_deit_', 'deit_')
            print(f"    [timm compat] {params['backbone']} -> {backbone_name}")

        backbone = timm.create_model(model_name=backbone_name, pretrained=False)
        model = ShopeeNet(backbone, num_classes=0, fc_dim=params['fc_dim']).to(self.device)
        model.load_state_dict(ckpt['model'], strict=False)
        model.eval()

        # Critical: kaggler sets p_eval for GeM pooling at inference time
        if 'p_eval' in params:
            model.p = params['p_eval']

        self.image_models[key] = model

    def _load_bert_model(self, filename, key, config_dir_name,
                         simple_mean=True, override_params_from=None):
        path = self._resolve_path(filename)
        if not path.exists():
            print(f"  [MISSING] {filename}")
            return

        print(f"  Loading {filename}...")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        params = ckpt['params']
        self.model_params[key] = params

        # Kaggler uses v75's fc_dim/max_len for v102
        if override_params_from and override_params_from in self.model_params:
            base = self.model_params[override_params_from]
            fc_dim = base['fc_dim']
            max_len = base['max_len']
        else:
            fc_dim = params['fc_dim']
            max_len = params['max_len']

        # Resolve tokenizer/config source
        config_dir = self.model_dir / config_dir_name
        hf_map = {
            'bert-indonesian': 'cahya/bert-base-indonesian-522M',
            'bert-multilingual': 'bert-base-multilingual-uncased',
            'xlm-roberta': 'xlm-roberta-base',
        }

        # Check nested dirs too
        if not config_dir.exists():
            candidates = [p for p in self.model_dir.rglob(config_dir_name) if p.is_dir()]
            if candidates:
                config_dir = candidates[0]

        try:
            if config_dir.exists() and config_dir_name == 'bert-indonesian':
                tokenizer = BertTokenizerFast(vocab_file=str(config_dir / 'vocab.txt'))
                bert_config = BertConfig.from_json_file(str(config_dir / 'config.json'))
                bert_model = BertModel(bert_config)
            elif config_dir.exists():
                tokenizer = AutoTokenizer.from_pretrained(str(config_dir))
                bert_config = AutoConfig.from_pretrained(str(config_dir))
                bert_model = AutoModel.from_config(bert_config)
            else:
                source = hf_map.get(config_dir_name, config_dir_name)
                print(f"    [HF Hub fallback] {source}")
                tokenizer = AutoTokenizer.from_pretrained(source)
                bert_config = AutoConfig.from_pretrained(source)
                bert_model = AutoModel.from_config(bert_config)

            model = BertNet(bert_model, num_classes=0, tokenizer=tokenizer,
                            max_len=max_len, fc_dim=fc_dim,
                            simple_mean=simple_mean).to(self.device)
            model.load_state_dict(ckpt['model'], strict=False)
            model.eval()
            self.bert_models[key] = model
        except Exception as e:
            print(f"    Failed to load {key}: {e}")

    # ---- Embedding Computation ----
    def _load_or_compute_embeddings(self):
        img_cache = self.cache_dir / 'img_feats_v2.npy'
        bert_cache = self.cache_dir / 'bert_feats_v2.npy'
        combined_cache = self.cache_dir / 'combined_feats_v2.npy'

        loaded = False
        if img_cache.exists() and bert_cache.exists() and combined_cache.exists():
            print("Loading cached embeddings...")
            try:
                self.img_feats = np.load(img_cache).astype(np.float32)
                self.bert_feats = np.load(bert_cache).astype(np.float32)
                self.combined_feats = np.load(combined_cache).astype(np.float32)
                if len(self.img_feats) == len(self.df):
                    loaded = True
                    print(f"  img_feats:      {self.img_feats.shape}")
                    print(f"  bert_feats:     {self.bert_feats.shape}")
                    print(f"  combined_feats: {self.combined_feats.shape}")
                else:
                    print(f"  Cache size mismatch. Recomputing...")
            except Exception as e:
                print(f"  Cache error: {e}. Recomputing...")

        if not loaded:
            self._compute_all_embeddings()
            if self.img_feats is not None:
                np.save(img_cache, self.img_feats)
            if self.bert_feats is not None:
                np.save(bert_cache, self.bert_feats)
            if self.combined_feats is not None:
                np.save(combined_cache, self.combined_feats)
            print("Embeddings cached.")

        self._build_faiss_indices()

    def _compute_all_embeddings(self):
        if self.df.empty:
            print("No data to index.")
            return
        if not self.image_models and not self.bert_models:
            print("No models loaded. Cannot compute embeddings.")
            return

        print(f"Computing embeddings for {len(self.df)} products...")

        class TempDataset(Dataset):
            def __init__(ds, df, img_dir):
                ds.df = df
                ds.img_dir = img_dir
            def __getitem__(ds, idx):
                row = ds.df.iloc[idx]
                try:
                    img = read_image(str(ds.img_dir / row['image']))
                    if img.shape[0] == 1:
                        img = img.repeat(3, 1, 1)
                except:
                    img = torch.zeros((3, 512, 512), dtype=torch.uint8)
                return img, row['title']
            def __len__(ds):
                return len(ds.df)

        dataset = TempDataset(self.df, self.train_img_dir)
        loader = DataLoader(dataset, batch_size=8, shuffle=False,
                            num_workers=0, collate_fn=lambda x: x)

        all_f1, all_f2 = [], []
        all_bf1, all_bf2, all_bf3 = [], [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Indexing"):
                imgs_raw, titles = list(zip(*batch))
                titles = list(titles)

                # Transform images exactly like kaggler
                if self.transforms:
                    imgs = torch.cat([
                        self.transforms(x.to(self.device).float() / 255)[None]
                        for x in imgs_raw
                    ], dim=0)

                # Image features
                if 'deit_small' in self.image_models:
                    all_f1.append(self.image_models['deit_small'].extract_feat(imgs).cpu().numpy())
                if 'efficientnet_b3' in self.image_models:
                    all_f2.append(self.image_models['efficientnet_b3'].extract_feat(imgs).cpu().numpy())

                # BERT features
                if 'bert_indonesian' in self.bert_models:
                    all_bf1.append(self.bert_models['bert_indonesian'].extract_feat(titles).cpu().numpy())
                if 'bert_multilingual' in self.bert_models:
                    all_bf2.append(self.bert_models['bert_multilingual'].extract_feat(titles).cpu().numpy())
                if 'xlm_roberta' in self.bert_models:
                    all_bf3.append(self.bert_models['xlm_roberta'].extract_feat(titles).cpu().numpy())

        # Concatenate and normalize — exactly matching kaggler's pipeline
        # Image: normalize each model separately, concat, re-normalize
        if all_f1 and all_f2:
            f1 = np.concatenate(all_f1)
            f1 /= np.linalg.norm(f1, 2, axis=1, keepdims=True)
            f2 = np.concatenate(all_f2)
            f2 /= np.linalg.norm(f2, 2, axis=1, keepdims=True)
            self.img_feats = np.concatenate([f1, f2], axis=1).astype(np.float32)
            self.img_feats /= np.linalg.norm(self.img_feats, 2, axis=1, keepdims=True)

        # BERT: normalize each model separately, concat all 3, re-normalize
        if all_bf1 and all_bf2 and all_bf3:
            bf1 = np.concatenate(all_bf1)
            bf1 /= np.linalg.norm(bf1, 2, axis=1, keepdims=True)
            bf2 = np.concatenate(all_bf2)
            bf2 /= np.linalg.norm(bf2, 2, axis=1, keepdims=True)
            bf3 = np.concatenate(all_bf3)
            bf3 /= np.linalg.norm(bf3, 2, axis=1, keepdims=True)
            self.bert_feats = np.concatenate([bf1, bf2, bf3], axis=1).astype(np.float32)
            self.bert_feats /= np.linalg.norm(self.bert_feats, 2, axis=1, keepdims=True)

        # Combined: [bert, img] concat then normalize (kaggler's bth_feats)
        if self.img_feats is not None and self.bert_feats is not None:
            self.combined_feats = np.concatenate([self.bert_feats, self.img_feats], axis=1).astype(np.float32)
            self.combined_feats /= np.linalg.norm(self.combined_feats, 2, axis=1, keepdims=True)

        print("Embedding computation complete.")

    def _build_faiss_indices(self):
        for name, feats, attr in [
            ("Image", self.img_feats, "img_index"),
            ("BERT", self.bert_feats, "bert_index"),
            ("Combined", self.combined_feats, "combined_index"),
        ]:
            if feats is not None and len(feats) > 0:
                d = feats.shape[1]
                index = faiss.IndexFlatIP(d)
                index.add(feats.astype(np.float32))
                setattr(self, attr, index)
                print(f"  {name} index: {index.ntotal} items, dim={d}")

    # ---- Query Feature Computation ----
    def _query_img_feats(self, image_input):
        if isinstance(image_input, (str, Path)):
            img = read_image(str(image_input))
        else:
            img = torch.from_numpy(np.array(image_input.convert('RGB'))).permute(2, 0, 1)

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        img_t = self.transforms(img.to(self.device).float() / 255).unsqueeze(0)

        with torch.no_grad():
            f1 = self.image_models['deit_small'].extract_feat(img_t).cpu().numpy() if 'deit_small' in self.image_models else None
            f2 = self.image_models['efficientnet_b3'].extract_feat(img_t).cpu().numpy() if 'efficientnet_b3' in self.image_models else None

        if f1 is not None and f2 is not None:
            f1 /= np.linalg.norm(f1, 2, axis=1, keepdims=True)
            f2 /= np.linalg.norm(f2, 2, axis=1, keepdims=True)
            feats = np.concatenate([f1, f2], axis=1).astype(np.float32)
            feats /= np.linalg.norm(feats, 2, axis=1, keepdims=True)
            return feats
        return None

    def _query_bert_feats(self, text_query):
        with torch.no_grad():
            bf1 = self.bert_models['bert_indonesian'].extract_feat([text_query]).cpu().numpy() if 'bert_indonesian' in self.bert_models else None
            bf2 = self.bert_models['bert_multilingual'].extract_feat([text_query]).cpu().numpy() if 'bert_multilingual' in self.bert_models else None
            bf3 = self.bert_models['xlm_roberta'].extract_feat([text_query]).cpu().numpy() if 'xlm_roberta' in self.bert_models else None

        if bf1 is not None and bf2 is not None and bf3 is not None:
            bf1 /= np.linalg.norm(bf1, 2, axis=1, keepdims=True)
            bf2 /= np.linalg.norm(bf2, 2, axis=1, keepdims=True)
            bf3 /= np.linalg.norm(bf3, 2, axis=1, keepdims=True)
            feats = np.concatenate([bf1, bf2, bf3], axis=1).astype(np.float32)
            feats /= np.linalg.norm(feats, 2, axis=1, keepdims=True)
            return feats
        return None

    # ---- Search ----
    def search_multimodal(self, image_input=None, text_query=None, k=10):
        img_f = None
        txt_f = None

        if image_input and self.transforms:
            img_f = self._query_img_feats(image_input)

        if text_query and str(text_query).strip():
            txt_f = self._query_bert_feats(text_query)

        # Pick the RIGHT index for the available modalities
        if img_f is not None and txt_f is not None and self.combined_index:
            query = np.concatenate([txt_f, img_f], axis=1).astype(np.float32)
            query /= np.linalg.norm(query, 2, axis=1, keepdims=True)
            D, I = self.combined_index.search(query, k)

        elif txt_f is not None and self.bert_index:
            D, I = self.bert_index.search(txt_f, k)

        elif img_f is not None and self.img_index:
            D, I = self.img_index.search(img_f, k)

        else:
            return []

        return self._format_results(I[0], D[0])

    def _format_results(self, indices, scores):
        results = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self.df):
                continue
            row = self.df.iloc[idx]
            results.append({
                'title': str(row['title']),
                'image': row['image'],
                'score': float(score),
                'price': f"${(abs(hash(row['title'])) % 100) + 10}.99",
                'posting_id': row['posting_id']
            })
        return results
