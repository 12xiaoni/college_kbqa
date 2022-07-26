from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from torchcrf import CRF
# 基于bert模型做序列标注
class bert_crf(BertPreTrainedModel):
    def __init__(self, config):
        super(bert_crf, self).__init__(config)
        self.bert = BertModel(config)
        embedding_dim = 768
        self.num_labels = 4
        self.dropout = nn.Dropout(0.5)
        output_size = 4
        self.fc = nn.Linear(embedding_dim, output_size)
        self.init_weights()
        self.crf = CRF(output_size,batch_first=True)
        

    def forward(self, x):
        out = self.bert(x)
        out = self.dropout(out[0])
        out = self.fc(out)
        return out