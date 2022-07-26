from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from ner_dataset import get_parameter
from ner_model import bert_crf
from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer, BertPreTrainedModel

def para():
    args, train_dataload, test_dataload = get_parameter()
    config = BertConfig.from_pretrained(args.bert_path)
    model = bert_crf.from_pretrained(args.bert_path, config = config).to(args.device)
    # 确定训练权重
    full_finetuning = True
    if full_finetuning:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
                'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
                'weight_decay': 0.0}
            ]
    else: 
            param_optimizer = list(model.fc.named_parameters()) 
            optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    # 确定训练的优化器和学习策略
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
    train_steps_per_epoch = 2278 // args.batch_size
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=args.epoch * train_steps_per_epoch)
    # batch = Batch(args)
    return model, train_dataload, test_dataload, args, optimizer, scheduler

model, train_dataload, test_dataload, args, optimizer, scheduler = para()

# 得到测试集的准确率
def evaluate():
    fil, num = 0,0
    model.eval()
    for idx, (input, output) in tqdm(enumerate(test_dataload)):
        input = input.to(args.device)
        output = output.to(args.device)
        y_pred = model(input)
        _,index = torch.max(y_pred, 2)
        num = num+len(index.tolist())
        for i in range(len(index.tolist())):
            if index.tolist()[i] == output.tolist()[i]:
                fil = fil+1
    return fil/num

# 确定训练模式
def train():
    model.train()
    min_loss = 300000
    al_loss = []
    for i in range(args.epoch):     
        for idx, (input, output) in tqdm(enumerate(train_dataload)):
            input = input.to(args.device)
            output = output.to(args.device)
            y_pred = model(input)
            loss_fn = -model.crf(y_pred, output)
            al_loss.append(loss_fn.item())
            if idx % 10 == 0:
                print(sum(al_loss)/len(al_loss))
                al_loss = []
                if min_loss > loss_fn:
                    min_loss = loss_fn
                    torch.save(model.state_dict(), 'ner/model/bert-crf.h5')
                    print(f"epoch:{i+1}/{args.epoch}, loss:{loss_fn}, acc:{evaluate()}")
            optimizer.zero_grad()
            loss_fn.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5)
            scheduler.step()
            optimizer.step()

if __name__ == '__main__':
    train()