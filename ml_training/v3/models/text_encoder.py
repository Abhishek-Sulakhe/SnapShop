"""
Text feature encoder for metric learning.

Architecture matches the inference engine's BertNet exactly so that
checkpoints produced here can be loaded at serving time.

Supports three transformer backbones:
  - cahya/bert-base-indonesian-522M  (simple mean pooling)
  - bert-base-multilingual-uncased   (attention-weighted mean pooling)
  - xlm-roberta-base                 (attention-weighted mean, no token_type_ids)

Output: 512-d embeddings after FC + BatchNorm, ready for L2-normalization.
"""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Text feature extractor wrapping a HuggingFace transformer model.

    Pooling strategy:
      simple_mean=True  : average all token hidden states equally
      simple_mean=False : attention-weighted mean (ignores padding tokens)

    XLM-RoBERTa uses SentencePiece tokenization and does NOT produce
    token_type_ids.  The code handles this automatically by checking
    whether the tokenizer output contains that key.
    """

    def __init__(self, bert_model, num_classes, tokenizer,
                 max_len=128, fc_dim=512, simple_mean=True):
        super().__init__()
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.simple_mean = simple_mean
        self.fc = nn.Linear(self.bert_model.config.hidden_size, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, x):
        tokenizer_output = self.tokenizer(
            x, truncation=True, padding=True, max_length=self.max_len
        )
        device = self.fc.weight.device

        input_ids = torch.LongTensor(tokenizer_output['input_ids']).to(device)
        attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to(device)

        if 'token_type_ids' in tokenizer_output:
            token_type_ids = torch.LongTensor(tokenizer_output['token_type_ids']).to(device)
            out = self.bert_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        else:
            # XLM-RoBERTa does not use token_type_ids
            out = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        if self.simple_mean:
            x = out.last_hidden_state.mean(dim=1)
        else:
            # Attention-weighted mean: ignores [PAD] tokens
            x = (
                torch.sum(out.last_hidden_state * attention_mask.unsqueeze(-1), dim=1)
                / attention_mask.sum(dim=1, keepdims=True)
            )

        x = self.fc(x)
        x = self.bn(x)
        return x

    def forward(self, x, label=None):
        return self.extract_feat(x)
