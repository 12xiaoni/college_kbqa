import torch
from data_process import get_para
from bert_model import *
# import csv
# import print
# from transformers.utils.dummy_pt_objects import AdamW

train_load, test_load, con = get_para()


# 训练模型
def train_model(model_name):
    if model_name == 'bert_rnn':
        model = TextRNN(con).to(con.device)
    if model_name == 'bert_cnn':
        model = TextCNN(con).to(con.device)
    if model_name == 'bert_rcnn':
        model = TextRCNN(con).to(con.device)
    min_loss = 100
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-4, eps=10e-8)
    sheduler = None
    model.train()
    criterion = nn.CrossEntropyLoss()
    for i in range(con.epoch):
        for idx, (ip,op) in enumerate(train_load):
            ip = ip.to(con.device)
            op = op.to(con.device)
            out = model(ip)
            loss = criterion(out, op)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            if idx%10 == 0:
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    acc = test_model(model)
                    torch.save(model.state_dict(),'second_class/data/model/' + model_name+'.h5')
                    print('epoch:'+str(i+1)+' model:'+ model_name + ' loss:'+ str(loss.item()), ' acc:'+str(acc))

# 测试模型返回模型预测准确率
def test_model(model):
    model.eval()
    num, fal = 0, 0
    for idx,(ip,op) in enumerate(test_load):
        ip = ip.to(con.device)
        op = op.to(con.device)
        pre = model(ip)
        _, y_pre = torch.max(pre,1)
        for i in range(len(y_pre)):
            num = num + len(y_pre)
            if y_pre.tolist()[i] != op.tolist()[i]:
                fal = fal + 1
    return 1-fal/num

if __name__ == '__main__':
    model = ['bert_rnn', 'bert_cnn', 'bert_rcnn']
    # bert_rnn = TextRNN(con).to(device)
    # bert_cnn = TextCNN(con).to(device)
    # bert_rcnn = TextRCNN(con).to(device)
    for i in model:
        train_model(i)


