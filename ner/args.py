from sklearn.model_selection import train_test_split
from ner.ner_data_process import writer_label
import argparse
import torch


# 模型的所有参数
def write_args():
    label_dict = {'O':0, 'PAD':1, 'B-collage':2, 'I-collage':3}
    sentence, sentence_label = writer_label()
    train_sentence, test_sentence, train_sentence_label, test_sentence_label \
        = train_test_split(sentence, sentence_label, test_size=0.2, random_state=2022)
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dict', type=dict, default=label_dict)
    parser.add_argument('--dict_label', type = dict, default = dict(zip(label_dict.values(), label_dict.keys())))
    parser.add_argument('--output_size', type=int, default=len(label_dict))
    parser.add_argument('--bert_embed', type=int, default=768)
    parser.add_argument('--bert_path', type=str, default='/mnt/e/data/bert-base-chinese/')
    parser.add_argument('--lr', type = float, default=3e-5)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--train_sentence', type=list, default=train_sentence)
    parser.add_argument('--test_sentence', type=list, default=test_sentence)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_sentence_label', type=list, default=train_sentence_label)
    parser.add_argument('--test_sentence_label', type=list, default=test_sentence_label)
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--epoch', type=int, default=15)
    args = parser.parse_args()
    return args