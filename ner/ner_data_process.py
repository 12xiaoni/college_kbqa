import os
import json
import random
from tqdm import tqdm

# 读取已经标注好的文件
def read_ner_file():
    print('开始读取数据：')
    sen_dict = dict()
    file_path = 'ner/outputs'
    for file in tqdm(os.listdir(file_path)):
        f = open(file_path+"/"+file, 'r', encoding='utf8').read()
        ner_sen = json.loads(f)
        sen_dict[ner_sen['content']] = []
        for i in ner_sen['outputs']['annotation']['T']:
            if i != '':
                sen_dict[ner_sen['content']].append([i['start'], i['end']])
    
    return sen_dict

# 对数据进行随机初始化
def random_st(sentence, sentence_label):
    new_sen, new_label = [], []
    random.seed(2022)
    lens = len(sentence)
    len_list = list(range(0,lens))
    random.shuffle(len_list)
    for i in len_list:
        new_sen.append(sentence[i])
        new_label.append(sentence_label[i])
    return new_sen, new_label

# 写入标签
def writer_label():
    sentence, sentence_label = [], []
    sen_dict = read_ner_file()
    print('写入标签：')
    for sen in tqdm(sen_dict.keys()):
        new_sen = list(sen)
        sen_label = ['O']*len(new_sen)
        for se in sen_dict[sen]:
            sen_label[se[0]] = 'B-collage'
            sen_label[se[0]+1:se[1]] = ['I-collage']*(se[1]-se[0]-1)
        sentence.append(new_sen)
        sentence_label.append(sen_label)
    sentence, sentence_label  = random_st(sentence, sentence_label)
    return sentence, sentence_label            
        


    