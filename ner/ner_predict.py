import torch
from ner.args import write_args
from transformers import BertConfig, BertTokenizer
from ner.ner_model import bert_crf
import torch.nn.functional as F

def ner_predict(qus):
    entity,xh_en = [],[]
    args = write_args()
    model = bert_crf.from_pretrained(args.bert_path).to(args.device)
    model.load_state_dict(torch.load('ner/model/bert-crf.h5'))
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    # q = '清华大学的历史?'
    list_qus = list(qus)
    index = tokenizer.convert_tokens_to_ids(list_qus)
    index = torch.LongTensor(index).view(1,-1).to(args.device)
    out = model(index)
    _,pre = torch.max(out,2)
    pre = pre.tolist()[0]
    for i in range(len(pre)):
        if pre[i] != 0:
            xh_en.append(list_qus[i])
        if pre[i] == 0:
            if pre[i-1] != 0:
                entity.append(''.join(xh_en))
                xh_en = []
    return entity





    

if __name__ == '__main__':
    ner_predict('许昌学院和南阳理工学院那个学校大')
