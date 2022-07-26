from torch import nn,optim
import torch
import torch.nn.functional as F
from transformers import BertModel

class TextRCNN(nn.Module):
    def __init__(self, config):
        super(TextRCNN, self).__init__()
        hidden_size = config.hidden_size
        embedding_dim = config.embedding_dim
        output_size = config.out_size
        num_layer = config.num_layers
        dropout = config.drop
        self.bert = BertModel.from_pretrained('/mnt/e/data/bert-base-chinese')
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layer, bidirectional = True, batch_first = True, dropout = dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.fc_concat = nn.Linear(hidden_size*2+embedding_dim, hidden_size*2)
    def forward(self, x):
        x = self.bert(**x)[0]
        out,(h_0,c_0) = self.lstm(x)
        out = self.fc_concat(torch.cat((x,out),2))
        out = F.sigmoid(out)
        out = out.permute(0,2,1)
        # try:
        #     maxpool = F.max_pool1d(out,out.shape[2].item()).squeeze(2)
        # except:
        maxpool = F.max_pool1d(out,out.shape[2]).squeeze(2)
        out = self.fc(maxpool)
        return out


class TextCNN(nn.Module):
    def __init__(self,config):
        super(TextCNN, self).__init__()
        filter = (3,4,5)
        hidden_size = config.hidden_size
        embedding_dim = config.embedding_dim
        output_size = config.out_size
        dropout = config.drop
        self.bert = BertModel.from_pretrained('/mnt/e/data/bert-base-chinese')
        self.cn = nn.ModuleList([nn.Conv2d(1, hidden_size,(k, embedding_dim)) for k in filter])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * len(filter), output_size)

    def forward(self,x):
        x = self.bert(**x)[0]
        x = x.unsqueeze(1) # x中应该batch_Size, chaannel,high, 
        x = [F.sigmoid(conv(x)).squeeze(3) for conv in self.cn]
        out_new = []
        for output in x:
            try:
                out_new.append(F.max_pool1d(output,output.shape[2].item()).squeeze(2))
            except:
                out_new.append(F.max_pool1d(output,output.shape[2]).squeeze(2))
        # print([i.shape for i in out_new])
        x = out_new
        x = torch.cat(x, 1) # (N, filter_num * len(filter_size)) -> (163, 100 * 3)
        # print(x.shape)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 构建分类模型
import torch.nn as nn
class TextRNN(nn.Module):
    def __init__(self,config):
        super(TextRNN, self).__init__()
        hidden_size = config.hidden_size
        embedding_dim = config.embedding_dim
        output_size = config.out_size
        num_layer = config.num_layers
        dropout = config.drop
        self.bert = BertModel.from_pretrained('/mnt/e/data/bert-base-chinese')
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layer, bidirectional = True,batch_first = True, dropout= dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)
    def forward(self,x):
        x = self.bert(**x)[0]
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return out