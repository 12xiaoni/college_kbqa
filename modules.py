import re
from first_class.first_result import model_result
from second_class.predict_test import second_predict
from ner.ner_predict import ner_predict
from faq.simbert import read_index
import torch
from config import *
import time
import random
from py2neo import Graph


graph = Graph(
        host = "127.0.0.1",
        http_host = 7474,
        user = 'neo4j',
        password = "123456")
        
def chatbot(qus):
    m_r = model_result()
    first_res = m_r.get_class_res([qus.lower()])
    if first_res in ['greet', 'goodbye', 'isbot', 'deny']:
        return random.choice(gossip_corpus[first_res])
    else:
        sec_res, cls_res = second_predict(qus)
        if max(sec_res.tolist()[0]) > intent_threshold_config['accept']:
            ans = []
            rel = cls_res
            ner_res = ner_predict(qus)
            for i in ner_res:
                neo4j_res = graph.run(semantic_slot[rel]["cql_template"].replace('{name}', i)).data()[0]['q.name']
                ans.append(semantic_slot[rel]['reply_template'].replace('{name}', i)+neo4j_res)
            return '\n'.join(ans)
        if intent_threshold_config['deny']<max(sec_res.tolist()[0])<intent_threshold_config['accept']:
            ans = []
            rel = cls_res
            ner_res = ner_predict(qus)
            if len(ner_res) > 1:
                return semantic_slot[rel]["ask_template"].replace('{name}','和'.join(ner_res))
            else:
                return semantic_slot[rel]["ask_template"].replace('{name}',ner_res[0])
        
        if max(sec_res.tolist()[0]) < intent_threshold_config['deny']:
            ans = read_index(qus)
            return ans
                
def qa():
    
    t = True
    while t:
        start = time.time()
        qus = input('请输入你的问题:')
        chatbot(qus)
        end = time.time() - start
        if end > 30:
            t = False

if __name__ == '__main__':
    qa()