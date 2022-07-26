from sklearn import svm #导出svm
import numpy as np
import random
import os
import pickle as pk
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
# from xgboost import XGBClassifier

class model_result:
    def __init__(self):
        # 加载已经训练好的模型
        self.gbdt = pk.load(open('first_class/train_model/gbdt.pkl','rb'))
        self.LR = pk.load(open('first_class/train_model/LR.pkl','rb'))
        self.svm = pk.load(open('first_class/train_model/svm.pkl', 'rb'))
        self.label2id = pk.load(open('first_class/train_model/label2id.pkl', 'rb'))
        self.vector = pk.load(open('first_class/train_model/vector.pkl', 'rb'))
    
    # 将训练好的结果进行模型融合
    def get_class_res(self, qus):
        tran_data = self.vector.transform(qus)
        gbda_pre = self.gbdt.predict_proba(tran_data.toarray())
        svm_pre = self.svm.predict_proba(tran_data.toarray())
        LR_pre = self.LR.predict_proba(tran_data.toarray())
        label = np.argmax((gbda_pre+svm_pre+LR_pre)/3, axis=1)
        self.id2label = dict(zip(self.label2id.values(), self.label2id.keys()))
        cls_res = self.id2label[label[0]]
        return cls_res

if __name__  == '__main__':
    mr = model_result()
    ans = mr.get_class_res(['不对，这样错了'.lower()])