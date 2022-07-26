import torch
from wsgiref.simple_server import demo_app
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from args import write_args

class NER_DATASET(Dataset):
    def __init__(self, args,inputs, targets):
        self.input = inputs
        self.output = targets
        self.label_dict = args.label_dict
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    
    def __getitem__(self, index):
        input = self.tokenizer.convert_tokens_to_ids(self.input[index])
        target = [self.label_dict[i] for i in self.output[index]]
        
        return input, target
    
    def __len__(self):
        return len(self.input)
    
def my_collate_fn(batch):
    ip,out,max_len = [], [], 0
    for i in batch:
        ip.append(i[0])
        out.append(i[1])
        if len(i[0]) > max_len:
            max_len = len(i[0])
    for idx in range(len(ip)):
        if len(ip[idx]) < max_len:
            ip[idx] = ip[idx]+[0]*(max_len-len(ip[idx]))
            out[idx] = out[idx]+[1]*(max_len-len(out[idx]))
    return torch.LongTensor(ip), torch.LongTensor(out)

def get_parameter():
    args = write_args()
    train_dataset = NER_DATASET(args, args.train_sentence, args.train_sentence_label)
    train_dataload = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, \
        shuffle=True, collate_fn=my_collate_fn)
    test_dataset = NER_DATASET(args, args.test_sentence, args.test_sentence_label)
    test_dataload = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, \
        shuffle=True, collate_fn=my_collate_fn)
    return args, train_dataload, test_dataload

