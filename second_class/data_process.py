import tokenizers
from data_generation import get_question
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertConfig, BertModel, BertTokenizer

random.seed(2022)
tokenizer = BertTokenizer.from_pretrained('/mnt/e/data/bert-base-chinese')

# 生word2idx，idx2word
def get_id():
    idx2word = dict()
    data,_ = get_question()
    # label = list(set([i[1] for i in data]))
    # for i in range(len(label)):
    #     idx2word[i] = label[i]
    # word2idx = dict(zip(idx2word.values(), idx2word.keys()))
    random.seed(2022)
    random.shuffle(data)
    # 划分训练集和数据集
    train_data, test_data = data[:int(0.8*len(data))],data[int(0.8*len(data)):]
    return train_data, test_data

class config:
    def __init__(self):
        train_data, test_data = get_id()
        self.train_data = train_data
        self.test_data = test_data
        self.word2idx = {'result':0, 'address':1, 'recommend':2, 'sort':3, 'grade':4, 'history':5,\
            'ratio':6, 'admissions':7, 'employment':8, 'code':9, 'tag':10, 'type':11}
        self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys()))
        self.out_size = len(self.word2idx)
        self.batch_size = 8
        self.epoch = 10
        self.hidden_size = 64
        self.drop = 0.5
        self.embedding_dim = 768
        self.num_layers = 2
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        


class my_data(Dataset):
    def __init__(self,config, train_data, word2idx):
        self.config = config
        self.data= train_data
        self.word2idx = word2idx
        

    def __getitem__(self, index):
        input, output = self.data[index][0], self.data[index][1]
        # self.config.tokenizer(input)['input_ids']
        return input, self.word2idx[output]

    def __len__(self):
        return len(self.data)
    
def my_collate_fn(batch):
    input, output = [], []
    for i in batch:
        input.append(i[0])
        output.append(i[1])    
    input = tokenizer(input, padding='max_length', max_length=30, return_tensors='pt', truncation=True )
    return input, torch.LongTensor(output)


def get_para():
    con = config()
    train_dataset = my_data(con, con.train_data, con.word2idx)
    test_dataset = my_data(con, con.test_data, con.word2idx)
    train_load = DataLoader(train_dataset, batch_size=con.batch_size, shuffle=True, drop_last=True, collate_fn=my_collate_fn)
    test_load = DataLoader(test_dataset, batch_size=con.batch_size, shuffle=True, drop_last=True, collate_fn=my_collate_fn)
    return train_load, test_load, con