# -*- coding:utf-8 -*-

semantic_slot = {
    "address":{
        "slot_list" : ["name"],
        "slot_values":None,
        "cql_template" : "MATCH (p:名字})-[:rel_adress]->(q:`地址`) where p.name='{name}' return q.name",
        "reply_template" : "'{name}' 的地址是在：\n",
        "ask_template" : "您问的是 '{name}' 的地址吗？",
        "intent_strategy" : "",
        "deny_response":"很抱歉没有理解你的意思呢~"
    },
    "sort":{
        "slot_list" : ["name"],
        "slot_values":None,
        "cql_template" : "MATCH (p:名字)-[:rel_school_ranking]->(q:`排名`) where p.name='{name}' return q.name",
        "reply_template" : "'{name}' 的排名是：\n",
        "ask_template" : "您问的是 '{name}' 的排名吗？",
        "intent_strategy" : "",
        "deny_response":"您说的我有点不明白，您可以换个问法问我哦~"
    },
    "employment":{
        "slot_list" : ["name"],
        "slot_values":None,
        "cql_template" : "MATCH (p:名字)-[:rel_employment_status]->(q:`就业情况`) where p.name='{name}' return q.name",
        "reply_template" : "关于 '{name}' 的就业情况：\n",
        "ask_template" : "请问您问的是 '{name}' 的就业情况吗？",
        "intent_strategy" : "",
        "deny_response":"额~似乎有点不理解你说的是啥呢~"
    },
    "recommend":{
        "slot_list" : ["name"],
        "slot_values":None,
        "cql_template" : "MATCH (p:名字)-[:rel_recommend_professional]->(q:`推荐专业`) where p.name='{name}' return q.name",
        "reply_template" : "'{name}' 比较好的专业有：\n",
        "ask_template" : "您问的是 '{name}' 的推荐专业吗？",
        "intent_strategy" : "",
        "deny_response":"人类的语言太难了！！"
    },
    "result":{
        "slot_list" : ["name"],
        "slot_values":None,
        "cql_template" : "MATCH (p:名字)-[:rel_assessment_result]->(q:`评估结果`) where p.name='{name}' return q.name",
        "reply_template" : "'{Disease}' 的学科评估结果：\n",
        "ask_template" : "您问的是 '{Disease}' 的评估结果吗？",
        "intent_strategy" : "",
        "deny_response":"人类的语言太难了！！~"
    },
    "grade":{
        "slot_list" : ["name"],
        "slot_values":None,
        "cql_template" : "MATCH (p:名字)-[:rel_grade]->(q:`等级`) where p.name='{name}' return q.name",
        "reply_template" : "'{name}' 是：\n",
        "ask_template" : "您问的是 '{name}' 的学校层次吗？",
        "intent_strategy" : "",
        "deny_response":"没有理解您说的意思哦~"
    },
    "admissions":{
        "slot_list" : ["name"],
        "slot_values":None,
        "cql_template" : "MATCH (p:名字)-[:rel_admissions_office]->(q:`招生办`) where p.name='{name}' return q.name",
        "reply_template" : "'{name}' 招生办电话是：\n",
        "ask_template" : "您想问的是'{name}' 招生办联系方式吗？",
        "intent_strategy" : "",
        "deny_response":"您说的我有点不明白，您可以换个问法问我哦~"
    },
    "code":{
        "slot_list" : ["name"],
        "slot_values":None,
        "cql_template" : "MATCH (p:名字)-[:rel_code]->(q:`代码`) where p.name='{name}' return q.name",
        "reply_template" : "'{name}' 的院校编码为：\n",
        "ask_template" : "您想问的是 '{name}' 学校编码吗？",
        "intent_strategy" : "",
        "deny_response":"没有理解您说的意思哦~"
    },
    "type":{
        "slot_list" : ["name"],
        "slot_values":None,
        "cql_template" : "MATCH (p:名字)-[:rel_type]->(q:`类型`) where p.name='{name}' return q.name",
        "reply_template" : "'{name}' 的类型为：",
        "ask_template" : "您想问 '{name}' 的类型吗？",
        "intent_strategy" : "",
        "deny_response":"您说的我有点不明白，您可以换个问法问我哦~"
    },
    "ratio":{
        "slot_list" : ["name"],
        "slot_values":None,
        "cql_template" : "MATCH (p:名字)-[:rel_ratio]->(q:`比例`) where p.name='{name}' return q.name",
        "reply_template" : "'{name}' 的男女为：",
        "ask_template" : "您想问 '{name}' 的男女比例吗？",
        "intent_strategy" : "",
        "deny_response":"很抱歉没有理解你的意思呢~"
    },
    "tag":{
        "slot_list" : ["name"],
        "slot_values":None,
        "cql_template" : "MATCH (p:名字)-[:rel_tag]->(q:`标签`) where p.name='{name}' return q.name",
        "reply_template" : "'{name}' 有一下的标签：\n",
        "ask_template" : "您是想问 '{name}' 的标签吗？",
        "intent_strategy" : "",
        "deny_response":"您说的我有点不明白，您可以换个问法问我哦~"
    },
    "history":{
        "slot_list" : ["name"],
        "slot_values":None,
        "cql_template" : "MATCH (p:名字)-[:rel_history]->(q:`历史`) where p.name='{name}' return q.name",
        "reply_template" : "'{name}' 的历史经历：\n",
        "ask_template" : "您是想问 '{name}' 的历史吗？",
        "intent_strategy" : "",
        "deny_response":"额~似乎有点不理解你说的是啥呢~~"
    },
    # "unrecognized":{
    #     "slot_values":None,
    #     "replay_answer" : "非常抱歉，我还不知道如何回答您，我正在努力学习中~",
    # }
}

intent_threshold_config = {
    "accept":0.65,
    "deny":0.4
}

default_answer = """很抱歉我还不知道回答你这个问题\n
                    你可以问我一些有关疾病的\n
                    定义、原因、治疗方法、注意事项、挂什么科室\n
                    预防、禁忌等相关问题哦~"""

gossip_corpus = {
    "greet":[
            "hi",
            "你好呀",
            "我是高考信息查询机器人小ni，有什么可以帮助你吗",
            "hi，你好，你可以叫我小ni",
            "你好，你可以问我一些关于高校信息的问题哦"
        ],
    "goodbye":[
            "再见，很高兴为您服务",
            "bye",
            "再见，感谢使用我的服务",
            "再见啦，祝你健康"
        ],
    "deny":[
            "很抱歉没帮到您",
            "I am sorry",
            "那您可以试着问我其他问题哟"
        ],
    "isbot":[
            "我是小ni，你的高校信息服务机器人",
            "你可以叫我小ni哦~",
            "我是高校信息服务机器人小ni"
        ],
}