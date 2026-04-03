import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent.parent

    INPUT_DIR = PROJECT_ROOT / 'input' / 'shopee-product-matching'
    OUTPUT_DIR = PROJECT_ROOT / 'output'
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'

    TRAIN_CSV = INPUT_DIR / 'train.csv'
    TRAIN_IMAGES = INPUT_DIR / 'train_images'

    SEED = 42
    NUM_WORKERS = 2
    DEVICE = 'cuda'

    # ── Image Models ──────────────────────────────────────────────
    IMAGE_MODELS = {
        'deit_small': {
            'backbone': 'vit_deit_small_distilled_patch16_224',
            'img_size': 224,
            'test_size': 224,
            'fc_dim': 512,
            'batch_size': 16,
            'epochs': 15,
            'lr': 3e-4,
            's': 30,
            'margin': 0.5,
            'p': 3,
            'loss_fn': 'curricularface',
            'optimizer': 'sam',
        },
        'efficientnet_b3': {
            'backbone': 'tf_efficientnet_b3_ns',
            'img_size': 512,
            'test_size': 384,
            'fc_dim': 512,
            'batch_size': 8,
            'epochs': 15,
            'lr': 3e-4,
            's': 30,
            'margin': 0.5,
            'p': 3,
            'p_eval': 4,
            'loss_fn': 'curricularface',
            'optimizer': 'sam',
        },
    }

    # ── Text Models ───────────────────────────────────────────────
    TEXT_MODELS = {
        'bert_indonesian': {
            'bert_name': 'cahya/bert-base-indonesian-522M',
            'fc_dim': 512,
            'max_len': 128,
            'batch_size': 32,
            'epochs': 10,
            'lr': 5e-5,
            's': 30,
            'margin': 0.5,
            'simple_mean': True,
            'loss_fn': 'arcface',
        },
        'bert_multilingual': {
            'bert_name': 'bert-base-multilingual-uncased',
            'fc_dim': 512,
            'max_len': 128,
            'batch_size': 32,
            'epochs': 10,
            'lr': 5e-5,
            's': 30,
            'margin': 0.5,
            'simple_mean': False,
            'loss_fn': 'arcface',
        },
        'xlm_roberta': {
            'bert_name': 'xlm-roberta-base',
            'fc_dim': 512,
            'max_len': 128,
            'batch_size': 32,
            'epochs': 10,
            'lr': 5e-5,
            's': 30,
            'margin': 0.5,
            'simple_mean': False,
            'loss_fn': 'arcface',
        },
    }

    # ── Multimodal Models ─────────────────────────────────────────
    MULTIMODAL_MODELS = {
        'deit_bert_multimodal': {
            'backbone': 'vit_deit_small_distilled_patch16_224',
            'bert_name': 'cahya/bert-base-indonesian-522M',
            'img_size': 224,
            'test_size': 224,
            'fc_dim': 512,
            'max_len': 128,
            'batch_size': 8,
            'epochs': 12,
            'lr': 3e-4,
            's': 30,
            'margin': 0.5,
            'p': 3,
            'loss_fn': 'arcface',
            'optimizer': 'sam',
        },
    }

    @classmethod
    def setup(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
