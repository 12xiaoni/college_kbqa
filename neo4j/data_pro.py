import pandas as pd
import json
from tqdm import tqdm

def re_csv():
    data = pd.read_csv('data\collagedata\collages.csv', encoding='gbk')
    with open('data\json_data\collages.json', 'w', encoding='utf8') as f:
        for i in tqdm(range(len(data))):
            collage_dict = dict()
            collage_dict["name"] = data.iloc[i]['中文名字']
            collage_dict["school_ranking"] = str(data.iloc[i]['排名'])
            collage_dict["employment_status"] = str(data.iloc[i]['就业情况'])
            collage_dict["recommend_professional"] = str('、'.join(data.iloc[i]['推荐专业'].split('|')))
            collage_dict["address"] = str(data.iloc[i]['通讯地址'])
            collage_dict["assessment_result"] = str(data.iloc[i]['评估结果'])
            collage_dict["grade"] = str(data.iloc[i]['本科or专科'])
            collage_dict["admissions_office"] = str(data.iloc[i]['招办电话'])
            collage_dict["code"] = str(data.iloc[i]['学校代码'])
            collage_dict["type"] = str(data.iloc[i]['学校类型'])
            collage_dict["ratio"] = '女生'+str(data.iloc[i]['女生比例'])+ ':' +'男生'+str(data.iloc[i]['男生比例'])
            collage_dict["tag"] = str(data.iloc[i]['汇总标签'])
            collage_dict["history"] = str(data.iloc[i]['描述'])
            collage_dict['official_website'] = str(data.iloc[i]['官网'])
            # 用json.dumps存储是怕解码出现错误
            f.write(json.dumps(collage_dict,ensure_ascii=False)+'\n')

if __name__ == '__main__':
    re_csv()


