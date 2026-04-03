import torch
import torch.nn as nn
from shopee_matching.utils.common import gem

class BertNet(nn.Module):

    def __init__(self,
                 bert_model,
                 num_classes,
                 tokenizer,
                 max_len=32,
                 fc_dim=512,
                 simple_mean=True,
                 s=30, margin=0.5, p=3, loss='ArcMarginProduct'):
        super().__init__()

        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fc = nn.Linear(self.bert_model.config.hidden_size, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p
        self.simple_mean = simple_mean

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, x):
        # x is assumed to be a list of strings (titles)
        tokenizer_output = self.tokenizer(x, truncation=True, padding=True, max_length=self.max_len)
        
        device = next(self.parameters()).device
        
        if 'token_type_ids' in tokenizer_output:
            input_ids = torch.LongTensor(tokenizer_output['input_ids']).to(device)
            token_type_ids = torch.LongTensor(tokenizer_output['token_type_ids']).to(device)
            attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to(device)
            x = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            input_ids = torch.LongTensor(tokenizer_output['input_ids']).to(device)
            attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to(device)
            x = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            
        if self.simple_mean:
            x = x.last_hidden_state.mean(dim=1)
        else:
            x = torch.sum(x.last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdims=True)
            
        x = self.fc(x)
        x = self.bn(x)
        return x
