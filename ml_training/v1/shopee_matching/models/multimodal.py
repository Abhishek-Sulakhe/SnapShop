import torch
import torch.nn as nn
from shopee_matching.utils.common import gem

class MultiModalNet(nn.Module):

    def __init__(self,
                 backbone,
                 bert_model,
                 num_classes,
                 tokenizer,
                 max_len=32,
                 fc_dim=512,
                 s=30, margin=0.5, p=3, loss='ArcMarginProduct'):
        super().__init__()

        self.backbone = backbone
        self.backbone.reset_classifier(num_classes=0)  # remove classifier

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
        device = next(self.parameters()).device
        
        img = self.backbone.forward_features(img)
        img = gem(img, p=self.p).view(batch_size, -1)

        tokenizer_output = self.tokenizer(title, truncation=True, padding=True, max_length=self.max_len)
        input_ids = torch.LongTensor(tokenizer_output['input_ids']).to(device)
        token_type_ids = torch.LongTensor(tokenizer_output['token_type_ids']).to(device)
        attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to(device)
        
        title_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        title_feat = title_out.last_hidden_state.mean(dim=1)

        x = torch.cat([img, title_feat], dim=1)
        x = self.fc(x)
        x = self.bn(x)
        return x
