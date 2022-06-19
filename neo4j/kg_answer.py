import json
from tqdm import tqdm
import threading
from py2neo import Graph

class BondExtractor:
    def __init__(self):
        self.graph = Graph(
            host = "127.0.0.1",
            http_host = 7474,
            user = 'neo4j',
            password = "123456"
        )
        self.col_ad=[] # 学校地址关系
        self.col_sr=[] # 学校排名关系
        self.col_employment=[] # 学校就业关系
        self.col_recommend=[] # 学校推荐专业关系
        # self.col_pf=[] # 学校专业关系
        self.col_result=[] # 学校评估结果关系
        self.col_grade=[] # 院校等级关系
        self.col_admissions=[] # 院校招生办关系
        self.col_code=[] # 院校学校代码
        self.col_type=[] # 院校学校类型
        self.col_ratio=[] # 院校学生比例
        self.col_tag=[] # 院校学校标签
        self.col_history=[] #院校历史关系
        self.col_official_website=[] # 院校官网关系
        self.name=[] # 院校名字
        self.adress=[] # 地址
        self.school_ranking=[] # 排名
        self.employment_status=[] # 就业情况
        self.recommend_professional=[] # 推荐专业
        # self.professional=[] # 专业
        self.assessment_result=[] # 评估结果
        self.grade=[] # 本科or专科
        self.admissions_office=[] # 招生办电话
        self.code=[] # 学校代码
        self.type=[] # 院校类型
        self.ratio=[] # 男女比例
        self.tag =[] # 汇总标签
        self.history = [] # 历史 
        self.official_website = [] # 官网
    
    def extraction_triad(self):
        print('开始抽取字典中的数据:')
        self.all_collage = []
        for i in open('data/json_data/collages.json',encoding='utf8').readlines():
            collage_dict = dict()
            i = json.loads(i)
            keys = i.keys()
            if 'name' in keys and i['name'] != "-": 
                self.name += [i['name']] 
                collage_dict['name'] = i['name']

            if 'address' in keys and i['address'] != "-": 
                self.adress += [i['address']]
                collage_dict['adress'] = i['address']

            if 'school_ranking' in keys and i['school_ranking'] != "-": 
                self.school_ranking += [i['school_ranking']]
                collage_dict['school_ranking'] = i['school_ranking']

            if 'employment_status' in keys and i['employment_status'] != "-": 
                self.employment_status += [i['employment_status']]
                collage_dict['employment_status'] = i['employment_status']

            if 'recommend_professional' in keys and i['recommend_professional'] != "-": 
                self.recommend_professional += [i['recommend_professional']]
                collage_dict['recommend_professional'] = i['recommend_professional']

            if 'assessment_result' in keys and i['assessment_result'] != "-": 
                self.assessment_result += [i['assessment_result']]
                collage_dict['assessment_result'] = i['assessment_result']

            if 'grade' in keys and i['grade'] != "-": 
                self.grade += [i['grade']]
                collage_dict['grade'] = i['grade']

            if 'admissions_office' in keys and i['admissions_office'] != "-": 
                self.admissions_office += [i['admissions_office']]
                collage_dict['admissions_office'] = i['admissions_office']

            if 'code' in keys and i['code'] != "-": 
                self.code += [i['code']]
                collage_dict['code'] = i['code']

            if 'type' in keys and i['type'] != "-": 
                self.type += [i['type']]
                collage_dict['type'] = i['type']

            if 'ratio' in keys and i['ratio'] != "-": 
                self.ratio += [i['ratio']]
                collage_dict['ratio'] = i['ratio']

            if 'tag' in keys and i['tag'] != "-": 
                self.tag += [i['tag']]
                collage_dict['tag'] = i['tag']
            
            if 'history' in keys and i['history'] != "-":
                self.history += [i['history']]
                collage_dict['history'] = i['history']

            if 'official_website' in keys and i['official_website'] != "-":
                self.official_website += [i['official_website']]
                collage_dict['official_website'] = i['official_website']
            

            if 'name' in keys and 'address' in keys and i['name'] != "-" and i['address'] != "-":
                self.col_ad.append([i['name'], 'rel_adress', i['address']])
            
            if 'name' in keys and 'school_ranking' in keys and i['name'] != "-" and i['school_ranking'] != "-":
                self.col_sr.append([i['name'], 'rel_school_ranking', i['school_ranking']])
            
            if 'name' in keys and 'employment_status' in keys and i['name'] != "-" and i['employment_status'] != "-":
                self.col_employment.append([i['name'], 'rel_employment_status', i['employment_status']])

            if 'name' in keys and 'recommend_professional' in keys and i['name'] != "-" and i['recommend_professional'] != "-":
                self.col_recommend.append([i['name'], 'rel_recommend_professional', i['recommend_professional']])

            if 'name' in keys and 'assessment_result' in keys and i['name'] != "-" and i['assessment_result'] != "-":
                self.col_result.append([i['name'], 'rel_assessment_result', i['assessment_result']])
            
            if 'name' in keys and 'grade' in keys and i['name'] != "-" and i['grade'] != "-":
                self.col_grade.append([i['name'], 'rel_grade', i['grade']])
            
            if 'name' in keys and 'admissions_office' in keys and i['name'] != "-" and i['admissions_office'] != "-":
                self.col_admissions.append([i['name'], 'rel_admissions_office', i['admissions_office']])
            
            if 'name' in keys and 'code' in keys and i['name'] != "-" and i['code'] != "-":
                self.col_code.append([i['name'], 'rel_code', i['code']])

            if 'name' in keys and 'type' in keys and i['name'] != "-" and i['type'] != "-":
                self.col_type.append([i['name'], 'rel_type', i['type']])

            if 'name' in keys and 'ratio' in keys and i['name'] != "-" and i['ratio'] != "-":
                self.col_ratio.append([i['name'], 'rel_ratio', i['ratio']])
            
            if 'name' in keys and 'tag' in keys and i['name'] != "-" and i['tag'] != "-":
                self.col_tag.append([i['name'], 'rel_tag', i['tag']])

            if 'name' in keys and 'history' in keys and i['name'] != "-" and i['history'] != "-":
                self.col_history.append([i['name'], 'rel_history', i['history']])
            
            if 'name' in keys and 'official_website' in keys and i['name'] != "-" and i['official_website'] != "-":
                self.col_official_website.append([i['name'], 'rel_official_website', i['official_website']])

        
            self.all_collage.append(collage_dict)

        self.name = list(set(self.name))
        self.adress = list(set(self.adress))
        self.school_ranking = list(set(self.school_ranking))
        self.employment_status = list(set(self.employment_status))
        self.recommend_professional = list(set( self.recommend_professional))
        self.assessment_result = list(set(self.assessment_result))
        self.grade = list(set(self.grade))
        self.admissions_office = list(set(self.admissions_office))

        self.code = list(set(self.code))
        self.type = list(set(self.type))
        self.ratio = list(set(self.ratio))
        self.tag = list(set(self.tag))
        self.history = list(set(self.history))
        self.official_website = list(set(self.official_website))
    
    def write_nodes(self, entitys, entity_type):
        print(f"写入{entity_type}")
        for node in tqdm(set(entitys), ncols = 80):
            cql = """MERGE(n:{label}{{name:'{entity_name}'}})""".format(
                label=entity_type,entity_name=node.replace("'",""))
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)
    
    def write_edges(self,triples,head_type,tail_type):
        print("写入 {0} 关系".format(triples[0][1]))
        for head,relation,tail in tqdm(triples,ncols=80):
            cql = """MATCH(p:{head_type}),(q:{tail_type})
                WHERE p.name='{head}' AND q.name='{tail}'
                MERGE (p)-[r:{relation}]->(q)""".format(
                    head_type=head_type,tail_type=tail_type,head=head.replace("'",""),
                    tail=tail.replace("'",""),relation=relation)
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)


    def set_attributes(self,entity_infos,etype):
        print("写入 {0} 实体的属性".format(etype))
        for e_dict in tqdm(entity_infos[892:],ncols=80):
            name = e_dict['name']
            del e_dict['name']
            for k,v in e_dict.items():
                cql = """MATCH (n:{label})
                    WHERE n.name='{name}'
                    set n.{k}='{v}'""".format(label=etype,name=name.replace("'",""),k=k,v=v.replace("'","").replace("\n",""))
                try:
                    self.graph.run(cql)
                except Exception as e:
                    print(e)
                    print(cql)

    def create_entitys(self):
        self.write_nodes(self.name,'name')
        self.write_nodes(self.adress,'adress')
        self.write_nodes(self.school_ranking,'school_ranking')
        self.write_nodes(self.employment_status,'employment_status')
        self.write_nodes(self.recommend_professional,'recommend_professional')
        self.write_nodes(self.assessment_result,'assessment_result')
        self.write_nodes(self.grade,'grade')
        self.write_nodes(self.admissions_office,'admissions_office')
        self.write_nodes(self.code,'code')
        self.write_nodes(self.type,'type')
        self.write_nodes(self.ratio,'ratio')
        self.write_nodes(self.tag,'tag')
        self.write_nodes(self.history,'history')
        self.write_nodes(self.official_website,'official_website')


    def create_relations(self):
        self.write_edges(self.col_ad,'name','adress')
        self.write_edges(self.col_sr,'name','school_ranking')
        self.write_edges(self.col_employment,'name','employment_status')
        self.write_edges(self.col_recommend,'name','recommend_professional')
        self.write_edges(self.col_result,'name','assessment_result')
        self.write_edges(self.col_grade,'name','grade')
        self.write_edges(self.col_admissions,'name','admissions_office')
        self.write_edges(self.col_code,'name','code')
        self.write_edges(self.col_type,'name','type')
        self.write_edges(self.col_ratio,'name','ratio')
        self.write_edges(self.col_tag,'name','tag')
        self.write_edges(self.col_history,'name','hsitory')
        self.write_edges(self.col_official_website,'name','official_website')
    
    def set_diseases_attributes(self): 
        # self.set_attributes(self.disease_infos,"疾病")
        t=threading.Thread(target=self.set_attributes,args=(self.all_collage,"name"))
        t.setDaemon(False)
        t.start()

# 搭建kg_answer可以查看neo4j文件夹下的read.md 文件

if __name__ == '__main__':
    be =BondExtractor()
    be.extraction_triad()
    be.create_entitys()
    be.create_relations()
    be.set_diseases_attributes()
