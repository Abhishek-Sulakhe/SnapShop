# Shopee Product Matching — Web Application

FastAPI-based search interface powered by a 6-model ensemble for multimodal
product matching.

## Architecture

The inference engine uses three search modalities:

| Modality     | Models                                            | FAISS Index      |
| ------------ | ------------------------------------------------- | ---------------- |
| **Image**    | DeiT Small + EfficientNet-B3                      | `img_index`      |
| **Text**     | BERT Indonesian + BERT Multilingual + XLM-RoBERTa | `bert_index`     |
| **Combined** | All 5 models concatenated                         | `combined_index` |

## Model Files

Place these in `webapp/webapp/models/` (or the configured checkpoint directory):

| File                    | Architecture                             | Type          |
| ----------------------- | ---------------------------------------- | ------------- |
| `deit_small.pth`        | `vit_deit_small_distilled_patch16_224`   | Image encoder |
| `efficientnet_b3.pth`   | `tf_efficientnet_b3_ns` with GeM pooling | Image encoder |
| `bert_indonesian.pth`   | `cahya/bert-base-indonesian-522M`        | Text encoder  |
| `bert_multilingual.pth` | `bert-base-multilingual-uncased`         | Text encoder  |
| `xlm_roberta.pth`       | `xlm-roberta-base`                       | Text encoder  |

Tokenizer config directories (auto-downloaded from HuggingFace if missing):

- `bert-indonesian/` — vocab.txt + config.json
- `bert-multilingual/` — AutoTokenizer files
- `xlm-roberta/` — AutoTokenizer files

## Embedding Cache

Pre-computed embeddings go in `webapp/webapp/cache/`:

- `img_feats_v2.npy` — Image feature vectors
- `bert_feats_v2.npy` — Text feature vectors
- `combined_feats_v2.npy` — Concatenated [text, image] vectors

## API

### `POST /api/search`

Multipart form with optional `image` file and/or `text` field.

Returns:

```json
{
  "results": [
    {
      "title": "Product name",
      "image": "filename.jpg",
      "score": 0.92,
      "price": "$29.99",
      "posting_id": "abc123"
    }
  ]
}
```

## Running

```bash
cd webapp/webapp
pip install -r requirements.txt
python run.py
```

Server starts at `http://localhost:8000`.
