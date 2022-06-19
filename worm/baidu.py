import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import json
import pandas as pd
from tqdm import tqdm
import re

#对爬虫爬到数据进行简单清洗
def data_pro(label, name):
    data_dict = dict()
    clear_label = [i.replace('\xa0', '').replace('\u062c','').replace('\u0627','') for i in label]
    clear_name = [ii.strip().replace('\xa0', '').replace('\u062c','').replace('\u0627','') for ii in name]
    clear_name = [re.sub('\[\d+\]','',j).strip() for j in clear_name]
    for i in range(len(clear_label)):
        data_dict[clear_label[i]] = clear_name[i]
    return data_dict


# 对百度百科所有高校进行爬虫
def getData(collage_name):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36",
    }
    url = "https://baike.baidu.com/item/"

    new_url = url +collage_name
    response = requests.get(new_url, headers=headers)
    response.encoding = 'utf8'
    bt = BeautifulSoup(response.text)
    all_new = bt.find_all(class_='basic-info J-basic-info cmn-clearfix')
    bt_all = BeautifulSoup(str(all_new[0]))
    label = [i.string for i in bt.find_all(class_="basicInfo-item name")]
    name = [ii.text for ii in bt.find_all(class_="basicInfo-item value")]
    data = data_pro(label, name)    
 
    return data
 
 
def main():
    data = pd.read_csv('data/collagedata/collage.csv', encoding = 'gbk')
    with open('data/json_data/collage.json', 'w', encoding='utf8')as f:
        for i in tqdm(range(len(data))):
            try:
                data_dict = getData(data.iloc[i]['学校名称'])
                f.write(json.dumps(data_dict,ensure_ascii=False)+'\n')
            except:
                print(1)
        f.close()
 
if __name__ == '__main__':
    main()


