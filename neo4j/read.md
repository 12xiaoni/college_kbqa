### 关于意图识别种类
> 注意：意图主要依据kg_answer中的number()函数
* 根据学校名字询问学校地址
* 依据学校名字查看校训
* 依据学校名字查看校领导
* 依据学校名字查看学校隶属于哪一个部门
* 查询学校开设专业
* 查询学校院系设置
* 查询院校的代码
* 查询院校的英文名称
* 查询院校的学校特色
* 查询院校的学校类别
* 查询院校建立时间



            if 'name' in keys and 'adress' in keys:
                self.col_ad.append([i['name'], 'rel_adress', i['adress']])
            
            if 'name' in keys and 'school_ranking' in keys:
                self.col_sr.append([i['name'], 'rel_school_ranking', i['school_ranking']])
            
            if 'name' in keys and 'employment_status' in keys:
                self.col_employment.append([i['name'], 'rel_employment_status', i['employment_status']])

            if 'name' in keys and 'recommend_professional' in keys:
                self.col_recommend([i['name'], 'rel_recommend_professional', i['recommend_professional']])

            if 'name' in keys and 'assessment_result' in keys:
                self.col_result.append([i['name'], 'rel_assessment_result', i['name']])
            
            if 'name' in keys and 'grade' in keys:
                self.col_grade.append([i['name'], 'rel_grade', i['grade']])
            
            if 'name' in keys and 'admissions_office' in keys:
                self.col_admissions.append([i['name'], 'rel_admissions_office', i['admissions_office']])
            
            if 'name' in keys and 'code' in keys:
                self.col_code.append([i['name'], 'rel_code', i['code']])

            if 'name' in keys and 'type' in keys:
                self.col_type.append([i['name'], 'rel_type', i['rel_type']])

            if 'name' in keys and 'ratio' in keys:
                self.col_ratio.append([i['name'], 'rel_ratio', i['ratio']])
            
            if 'name' in keys and 'tag' in keys:
                self.col_tag.append([i['name'], 'rel_tag', i['tag']])

            if 'name' in keys and 'history' in keys:
                self.col_history.append([i['name'], 'rel_history', i['history']])
            
            if 'name' in keys and 'official_website' in keys:
                self.col_official_website.append([i['name'], 'rel_official_website', i['official_website']])