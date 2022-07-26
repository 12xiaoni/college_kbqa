import numpy as np
import pandas as pd


def get_question():
    all_college_name = pd.read_csv('data/collagedata/college.csv', encoding='gbk')['中文名字'].to_list()
    all_question,recall_question = [], []
    # 问题的生成
    explain_qwds = ['能否介绍一下','想了解','可以问一下','','','能否告知','','可以介绍','解释一下','解释解释']
    greet_qwds = ['你好，麻烦问一下','打扰问一下','','','您好','','请问','请教一下','冒昧问一下','问一下','','']
    adress_qwds = ['在哪里', '位于什么省市', '在那个区', '地址', '是不是位于', '是在', '在哪?']
    sr_qwds = ['排第几', '排几名', '的高校排名', '能排多少', '有多厉害', '是第一吗', '能排多少名', '能进前100吗']
    employment_qwds = ['就业情况怎么样', '好就业吗', '能够干什么', '毕业生去向', '能找到好工作吗', '毕业可以找到工作吗', '就业可以吗', '毕业有好去处吗']
    recommend_qwds = ['有什么好专业', '有专业可以推荐吗', '王牌专业是什么', '可以选啥学院', '什么专业可以找到好工作', '有什么厉害的学科', '厉害的专业都有什么']
    result_qwds = ['学科评估结果', '建筑学是a+还是a', '经济学是b吗', 'a类学科有多少', '人工智能是a+吗', '今年学科评估情况', '有a类专业吗', '有几个b', 'c类专业有多少']
    grade_qwds = ['是专科吗', '是本科吗', '专科还是本科', '有没有专科', '有没有本科专业']
    admissions_qwds = ['招生办在哪里', '招生办电话', '招生政策', '招多少学生', '可不可以去这个学校']
    code_qwds = ['编码是多少', '代码' , '代码是几位数', '编号']
    tag_qwds = ['是985吗', '是211吗','是不是双一流高校', '民办还是私立', '是不是一本', '是国际知名高校吗', '是中国一流大学吗', '是不是职业学院', '是不是全国重点大学']
    type_qwds = ['是理工还是师范', '是艺术类学校吗', '是不是政法大学', '理工科强还是文科强', '是不是语言类学校']
    ratio_qwds = ['男女比例咋样', '男生多还是女生多', '可不可以交到女朋友', '有多少男生', '有多少女生', '男女比例平均吗']
    history_qwds = ['的历史', '由来', '有多少年历史', '来历', '的国际成就', '的占地面积', '所有信息', '综合信息']

    # 补充地址问题生成
    n_adress = ['北京', '上海', '广州', '深圳', '南阳', '九江', '邯郸', '甘肃', '河北', '广西', '湛江']

    # 询问地址greet-explain-name-adress
    # explain 和greet里有空格是为了增加语义丰富度
    for i in range(60):
        greet = np.random.choice(greet_qwds)
        explain = np.random.choice(explain_qwds)
        name = np.random.choice(all_college_name)
        adress = np.random.choice(adress_qwds)
        if adress == '是不是位于':
            adress = adress + np.random.choice(n_adress)
        elif adress == '是在':
            adress = adress + np.random.choice(n_adress) + '吗'
        
        if i % 12 == 0:
            name = ''
        adress_question = greet + explain + name + adress
        all_question.append([adress_question, 'address'])
        recall_question.append([greet + explain + adress, 'address'])
    
        
    # 询问学校排名情况
    # greet-explain-name-sort
    for i in range(60):
        greet = np.random.choice(greet_qwds)
        explain = np.random.choice(explain_qwds)
        name = np.random.choice(all_college_name)
        sort = np.random.choice(sr_qwds)
        if i % 12 == 0:
            name = ''
        sort_question = greet+explain+name+sort
        all_question.append([sort_question, 'sort'])
        recall_question.append([greet + explain + sort, 'sort'])
    
    # 询问学校专业 greet-explain-name-recommend
    for i in range(60):
        greet = np.random.choice(greet_qwds)
        explain = np.random.choice(explain_qwds)
        name = np.random.choice(all_college_name)
        recommend = np.random.choice(recommend_qwds)
        if i % 12 == 0:
            name = ''
        recommend_question = greet+explain+name+recommend
        all_question.append([recommend_question, 'recommend'])
        recall_question.append([greet + explain + recommend, 'recommend'])
        
    # 询问学校就业情况
    # greet-explain-name-employment
    for i in range(60):
        greet = np.random.choice(greet_qwds)
        explain = np.random.choice(explain_qwds)
        name = np.random.choice(all_college_name)
        employment = np.random.choice(employment_qwds)
        if i % 12 == 0:
            name = ''
        employment_question = greet+explain+name+employment
        all_question.append([employment_question, 'employment'])
        recall_question.append([greet + explain + employment, 'employment'])
        
    # 询问学校学科评估结果
    # greet-explain-name-result
    for i in range(60):
        greet = np.random.choice(greet_qwds)
        explain = np.random.choice(explain_qwds)
        name = np.random.choice(all_college_name)
        result = np.random.choice(result_qwds)
        if i % 12 == 0:
            name = ''
        result_question = greet+explain+name+result
        all_question.append([result_question, 'result'])
        recall_question.append([greet + explain + result, 'result'])
        
    # 询问学校等级
    # greet-explain-name-grade
    for i in range(60):
        greet = np.random.choice(greet_qwds)
        explain = np.random.choice(explain_qwds)
        name = np.random.choice(all_college_name)
        grade = np.random.choice(grade_qwds)
        if i % 12 == 0:
            name = ''
        grade_question = greet+explain+name+grade
        all_question.append([grade_question, 'grade'])
        recall_question.append([greet + explain + grade, 'grade'])
    
    # 查询学校招生办
    # greet-explain-name-admissions
    for i in range(60):
        greet = np.random.choice(greet_qwds)
        explain = np.random.choice(explain_qwds)
        name = np.random.choice(all_college_name)
        admissions = np.random.choice(admissions_qwds)
        if i % 12 == 0:
            name = ''
        admissions_question = greet+explain+name+admissions
        all_question.append([admissions_question, 'admissions'])
        recall_question.append([greet + explain + admissions, 'admissions'])
    
    # 查询学校编码
    # greet-explain-name-code
    for i in range(60):
        greet = np.random.choice(greet_qwds)
        explain = np.random.choice(explain_qwds)
        name = np.random.choice(all_college_name)
        code = np.random.choice(code_qwds)
        if i % 12 == 0:
            name = ''
        code_question = greet+explain+name+code
        all_question.append([code_question, 'code'])
        recall_question.append([greet + explain + code, 'code'])
    
    # 查询学校标签
    # greet-explain-name-tag
    for i in range(60):
        greet = np.random.choice(greet_qwds)
        explain = np.random.choice(explain_qwds)
        name = np.random.choice(all_college_name)
        tag = np.random.choice(tag_qwds)
        if i % 12 == 0:
            name = ''
        tag_question = greet+explain+name+tag
        all_question.append([tag_question, 'tag'])
        recall_question.append([greet + explain + tag, 'tag'])
    
    # 查询学校类型
    # greet-explain-name-type
    for i in range(60):
        greet = np.random.choice(greet_qwds)
        explain = np.random.choice(explain_qwds)
        name = np.random.choice(all_college_name)
        type = np.random.choice(type_qwds)
        if i % 12 == 0:
            name = ''
        type_question = greet+explain+name+type
        all_question.append([type_question, 'type'])
        recall_question.append([greet + explain + type, 'type'])
    
    # 查询学校男女比例
    # greet-explain-name-ratio
    for i in range(60):
        greet = np.random.choice(greet_qwds)
        explain = np.random.choice(explain_qwds)
        name = np.random.choice(all_college_name)
        ratio = np.random.choice(ratio_qwds)
        if i % 12 == 0:
            name = ''
        ratio_question = greet+explain+name+ratio
        all_question.append([ratio_question, 'ratio'])
        recall_question.append([greet + explain + ratio, 'ratio'])
    
    # 查询学校历史
    # greet-explain-name-history
    for i in range(60):
        greet = np.random.choice(greet_qwds)
        explain = np.random.choice(explain_qwds)
        name = np.random.choice(all_college_name)
        history = np.random.choice(history_qwds)
        if i % 12 == 0:
            name = ''
        history_question = greet+explain+name+history
        all_question.append([history_question, 'history'])
        recall_question.append([greet + explain + history, 'history'])
    return all_question, recall_question


# if __name__ == '__main__':
#     get_question()



