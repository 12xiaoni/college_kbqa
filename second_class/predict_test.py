from transformers import BertConfig, BertTokenizer, BertModel
import torch
import torch.nn.functional as F
from second_class.bert_model import *
from second_class.data_process import config, get_para, tokenizer

# 返回模型最后输出的结果
def second_predict(ans):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, _, config = get_para()
    model = TextRNN(config).to(device)
    model.load_state_dict(torch.load('second_class/data/model/bert_rnn.h5'))
    # qus = '清华大学的王牌专业是什么？'
    qus_bl = tokenizer(ans, return_tensors='pt', truncation=True).to(device)
    res = model(qus_bl)
    res = F.softmax(res)
    _,idx = torch.max(res,1)
    cls_res = config.idx2word[idx.tolist()[0]]
    return res, cls_res

if __name__ == '__main__':
    second_predict('许昌学院毕业生的就业情况怎么样？')
