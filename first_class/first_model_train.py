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
from xgboost import XGBClassifier


seed = 222
random.seed(seed)
np.random.seed(seed)

class train_ml:
    def __init__(self):
        self.input = []
        self.output = []
        self.stop_words = []
        self.data_path = r'data\chatbot.txt'
        self.model_path = r'train_model'
        
    def read_data(self):
        all_word = []
        for i in open(self.data_path, 'r', encoding='utf8').readlines():
            # ip, op = i.strip().split(',')
            # self.input.append(ip)
            # self.output.append(op)
            text,label = i.strip().split(',')
            text = text.lower()

            self.input.append(text)
            self.output.append(label)
        index = np.arange(len(self.input))
        np.random.shuffle(index)
        self.input = [self.input[i] for i in index]
        self.output = [self.output[i] for i in index]


        # for i in open('stop_words.txt', 'r', encoding='utf8').readlines():
        #     self.stop_words.append(i.strip())

    # z
    def load_tfidf(self):
        # class_le = LabelEncoder()
        # out = class_le.fit_transform(self.output)    
        # label_set = sorted(list(set(self.output)))
        # label2id = {label:idx for idx,label in enumerate(label_set)}
        label2id = dict()
        for i in list(set(self.output)):
            label2id[i] = len(label2id.keys())
        out = [label2id[i] for i in self.output]
        x_train,x_test,y_train,y_test = train_test_split(self.input, out, test_size=0.15, random_state=26)
        vector = TfidfVectorizer(ngram_range=(1,3), min_df=0, max_df=0.9,analyzer='char',use_idf=1,smooth_idf=1, sublinear_tf=1)
        x_train = vector.fit_transform(x_train)
        x_test = vector.transform(x_test)
        pk.dump(label2id,open(os.path.join(self.model_path,'label2id.pkl'),'wb'))
        pk.dump(vector,open(os.path.join(self.model_path,'vector.pkl'),'wb'))
        return x_train,x_test,y_train,y_test
    
    def train_model(self, x_train,x_test,y_train,y_test):
        
        # 使用svm，xgboost，randomforeast做分类
        svm_model = svm.SVC(C=8, probability=True,kernel='rbf', random_state=24)
        xg_model = XGBClassifier(n_estimators=450, learning_rate=0.01,max_depth=8,random_state=24) 
        gbdt = GradientBoostingClassifier(n_estimators=450, learning_rate=0.01,max_depth=8, random_state=24) 
        LR = LogisticRegression(C=8, dual=False,n_jobs=4,max_iter=400,multi_class='ovr',random_state=102)
        # sg_model = 
        # params = [{'kernel':['linear'],'C':[1,10,100,1000]},{'kernel':['poly'],'C':[1,10],'degree':[2,3]},{'kernel':['rbf'],'C':[1,10,100,1000], 'gamma':[1,0.1, 0.01, 0.001]}]
        # C_range = np.logspace(-2, 10, 13)# logspace(a,b,N)把10的a次方到10的b次方区间分成N份
        # gamma_range = np.logspace(-9, 3, 13)
        # param_grid = dict(gamma=gamma_range, C=C_range)

        # model = GridSearchCV(estimator=model, param_grid = params, cv=5)
        LR.fit(x_train, y_train)
        pred = LR.predict(x_test)
        LR_acc = accuracy_score(pred, y_test)
        gbdt.fit(x_train, y_train)
        pred = gbdt.predict(x_test)
        gbdt_acc = accuracy_score(pred, y_test)
        xg_model.fit(x_train,y_train) 
        xg_pre = xg_model.predict(x_test)
        xg_acc = accuracy_score(xg_pre, y_test)
        svm_model.fit(x_train, y_train)
        svm_pre = svm_model.predict(x_test)
        svm_acc = accuracy_score(svm_pre, y_test)
      
        pk.dump(svm_model,open(os.path.join(self.model_path,'svm.pkl'),'wb'))
        pk.dump(LR,open(os.path.join(self.model_path,'LR.pkl'),'wb'))
        pk.dump(gbdt,open(os.path.join(self.model_path,'gbdt.pkl'),'wb'))
        # pk.dump(xg_model,open(os.path.join(self.model_path,'xg_model.pkl'),'wb'))




if __name__ == '__main__':
    tm = train_ml()
    tm.read_data()
    x_train,x_test,y_train,y_test = tm.load_tfidf()
    tm.train_model(x_train,x_test,y_train,y_test)
