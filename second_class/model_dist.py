import torch
import random
from tqdm import tqdm
import torch.nn as nn
from bert_model import TextRNN
import torch.nn.functional as F
from data_process import config, get_question
from torch.utils.data import Dataset, DataLoader
from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer

"""
用传统lstm模型对bert_rnn 文本分类模型做蒸馏
"""
random.seed(2022)
torch.manual_seed(2022)
question,_  = get_question()
random.shuffle(question)
cfg = config()
# 生成idx
def tran_idx():
    label = {'result':0, 'address':1, 'recommend':2, 'sort':3, 'grade':4, 'history':5,\
            'ratio':6, 'admissions':7, 'employment':8, 'code':9, 'tag':10, 'type':11}
    return label

class my_stu_data(Dataset):
    def __init__(self, question, label):
        self.question = question
        self.label = label

    def __getitem__(self, index):
        input = question[index][0]
        output = self.label[question[index][1]]
        
        return input, output
        
    def __len__(self):
        return len(self.question)

def my_collate_fn(batch):
    tokenizer = BertTokenizer.from_pretrained('/mnt/e/data/bert-base-chinese')
    input, output = [], []
    for i in batch:
        input.append(i[0])
        output.append(i[1])
    input = tokenizer(input, padding='max_length', max_length=30, return_tensors='pt', truncation=True )
    return input, torch.LongTensor(output)
    

# 构建分类模型

class biLSTM(nn.Module):

    def __init__(self):
        super(biLSTM, self).__init__()
        self.Embedding = nn.Embedding(21128,300)
        self.lstm = nn.LSTM(input_size=300, hidden_size=64,
                            num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        # self.linear = nn.Linear(in_features=256, out_features=2)
        self.fc1 = nn.Linear(64*2, 12)

    def forward(self, x, hidden=None):
        x = self.Embedding(x)
        lstm_out, hidden = self.lstm(x, hidden)     # LSTM 的返回很多
        
        # activated_t = F.relu(lstm_out)
        # linear_out = self.fc2(activated_t)
        linear_out = lstm_out[:,-1,:]
        out = self.fc1(linear_out)

        return out


# 返回训练模型和测试模型的所有参数
def get_parameter(stu_dataload):
    # train_dataload, test_dataload = get_data()
    # 得到一个gpu的判定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 实例化模型，并放到gpu上
    model = biLSTM().to(device)
    teacher_model = TextRNN(cfg).to(device)
    # https://blog.csdn.net/weixin_45743001/article/details/120472616
    # 优化器 sgd....
    optimizer = AdamW(model.parameters(),lr=0.001)
    train_steps_per_epoch = 720 // 8
    # 损失函数 ，交叉熵损失函数
    # criterion(y_pred, output)
    criterion1 = nn.CrossEntropyLoss(ignore_index=-1)
    criterion2 = nn.MSELoss()
    # 更换改变学习率
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=10 * train_steps_per_epoch)
    return teacher_model,model, device, optimizer, criterion1, criterion2, scheduler, stu_dataload

def eval_tearcher_model(teacher_model, input):
    teacher_model.load_state_dict(torch.load('second_class/data/model/bert_rnn.h5'))
    teacher_model.eval()
    with torch.no_grad():
        out = teacher_model(input)
    return out

# 训练模型
def train_cls_model(teacher_model, model, device, optimizer, criterion1, criterion2,scheduler, stu_dataload):
    # 模型的声明
    model.train()
    min_loss = 10
    print('开始训练模型:')
    # 开始训练模型
    # config_epoch 数据训练次数
    for i in range(10):
        for idx,(input, output) in tqdm(enumerate(stu_dataload)):
            # 把输入值放在gpu
            input = input.to(device)
            # 把输出值放在gpu上边
            output = output.to(device)
            # 将input放入模型进行训练
            out1 = model(input['input_ids'])
            out2 = eval_tearcher_model(teacher_model, input)
            # output = torch.LongTensor([1,1,1,1,2,2,2,1,1,2,1,2,1,2,1,2])
            # 计算损失函数
            loss = criterion1(out1, output) + 2*criterion2(out2, out1)
            if idx % 10 == 0:  
                if loss.item() < min_loss:
                    min_loss = loss
                    # 保存我们训练好的模型
                    # config_save_model 模型保存地址
                    torch.save(model.state_dict(), r'second_class/data/model/student.ckpt') 
                    # acc = test()
                    print(f"epoch:{i+1},loss:{loss}")
                    # print(loss.item())
            # 清空磁盘
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5)
            scheduler.step()
            optimizer.step()
    print('模型训练结束')

if __name__ == '__main__':
    label = tran_idx()
    stu_dataset = my_stu_data(question, label)
    stu_dataload = DataLoader(dataset=stu_dataset, batch_size=8, shuffle=True, collate_fn=my_collate_fn)
    teacher_model,model, device, optimizer, criterion1, criterion2, scheduler, stu_dataload = get_parameter(stu_dataload)
    train_cls_model(teacher_model, model, device, optimizer, criterion1, criterion2, scheduler, stu_dataload)