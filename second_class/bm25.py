import pandas as pd
from tqdm import tqdm
from data_generation import get_question
from rank_bm25 import BM25Okapi

# 基于bm250做问题召回，对分类问题做增强
def get_data():
    # init_data = []
    _, init_data = get_question()
    # init_data = [i[0] for i in gen_data]
    recall_data = pd.read_csv('faq/faq_worm/data/qa.csv', encoding='utf8').q.tolist()
    return init_data, recall_data

def bm25_model():
    new_q = []
    init_data, recall_data = get_data()
    corpus = [list(i) for i in recall_data]
    bm25 = BM25Okapi(corpus)
    for i in tqdm(init_data):
        new_q.append(i)
        need_q = bm25.get_top_n(list(i[0]),corpus,n=3)
        for word in need_q:
            new_q.append([''.join(word), i[1]])
    return new_q

if __name__ == '__main__':
    bm25_model()



