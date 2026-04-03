import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import faiss
import timm
from transformers import BertTokenizerFast, BertModel
from torchvision import transforms
from PIL import Image

from shopee_matching.config import Config
from shopee_matching.models.image_encoders import ShopeeNet
from shopee_matching.models.text_encoders import BertNet
from shopee_matching.data.dataset import ShopeeDataset, BertDataset

def extract_image_features(df, img_dir=Config.TRAIN_IMAGES, checkpoint_path=None, model=None, device='cuda'):
    # Device
    if isinstance(device, str):
        device = torch.device(device)
        
    # Model
    if model is None:
        backbone = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
        model = ShopeeNet(backbone, num_classes=11014, fc_dim=Config.FC_DIM)
        model.to(device)
        
        if checkpoint_path is None:
            checkpoint_path = Config.SHOPEE_MODEL
            
        try:
            # Flexible loading: strict=False in case of partial matches or wrapper keys
            state = torch.load(checkpoint_path, map_location=device)
            if 'model' in state:
                model.load_state_dict(state['model'], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            print(f"Loaded image model from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load image model from {checkpoint_path}: {e}")
            
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Data
    dataset = ShopeeDataset(df, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=Config.NUM_WORKERS, drop_last=False)
    
    # Inference
    embeds = []
    with torch.no_grad():
        for img, _, _, _, _ in tqdm(dataloader, desc="Extracting Image Features"):
            img = img.to(device)
            feat = model(img)
            embeds.append(feat.cpu().numpy())
            
    return np.concatenate(embeds)

def extract_text_features(df, checkpoint_path=None, model=None, device='cuda'):
    if isinstance(device, str):
        device = torch.device(device)
        
    # Tokenizer
    try:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    except:
        tokenizer = BertTokenizerFast.from_pretrained(Config.BERT_MULTILINGUAL_DIR)
        
    if model is None:
        try:
            base = BertModel.from_pretrained('bert-base-multilingual-cased')
        except:
            base = BertModel.from_pretrained(Config.BERT_MULTILINGUAL_DIR)

        model = BertNet(base, num_classes=11014, tokenizer=tokenizer, fc_dim=Config.FC_DIM)
        model.to(device)
        
        if checkpoint_path is None:
            checkpoint_path = Config.BERT_MODEL
            
        try:
            state = torch.load(checkpoint_path, map_location=device)
            if 'model' in state:
                model.load_state_dict(state['model'], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            print(f"Loaded text model from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load text model from {checkpoint_path}: {e}")
            
    model.eval()
    
    dataset = BertDataset(df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=Config.NUM_WORKERS, drop_last=False)
    
    embeds = []
    with torch.no_grad():
        for text in tqdm(dataloader, desc="Extracting Text Features"):
            # BertDataset returns text strings or (text, label)
            if isinstance(text, list) or isinstance(text, tuple):
                 # if batch returns tuple (text, label)
                 if len(text) == 2 and isinstance(text[1], torch.Tensor):
                     text = text[0]
            
            feat = model.extract_feat(text)
            embeds.append(feat.cpu().numpy())
            
    return np.concatenate(embeds)

def get_neighbors(feats, k=Config.K):
    # Check if we can use GPU
    try:
        import faiss.contrib.torch_utils
    except ImportError:
        pass
        
    d = feats.shape[1]
    
    # Use simple Inner Product index
    index = faiss.IndexFlatIP(d)
    
    # Try GPU
    if torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print(f"FAISS GPU failed, using CPU: {e}")
            
    index.add(feats)
    # Search
    D, I = index.search(feats, k)
    return D, I
