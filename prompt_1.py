from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
import json
import pandas as pd
from classes import Alphas
import codecs


def unicode_to_chinese_safe(unicode_str):
    # 将字符串中的 \\u 替换为 \u
    unicode_str = unicode_str.replace(r'\\u', r'\u')
    # 使用codecs模块解码Unicode字符串
    return codecs.decode(unicode_str, 'unicode_escape')


base_factor = pd.read_excel('./resource/base_factor.xls')
base_alpha = pd.read_excel('./resource/base_alpha_factors.xls')
base_operator = pd.read_excel('./resource/base_operator.xls')

categories = base_alpha.iloc[:, 3].unique()

parser = JsonOutputParser(pydantic_object=Alphas)

factor_description = {}
operator_description = base_operator.to_json(orient='records', force_ascii=False)
categories_description = {}
format_instruction = unicode_to_chinese_safe(parser.get_format_instructions())

for i in categories:
    factor_description[i] = []
    for _, s in base_alpha.iterrows():
        item = {}
        if s['category'] == i and not categories_description.get(i, None):
            item['name'] = s['name']
            item['meaning'] = s['meaning']
            factor_description[i].append(item)
            categories_description[i] = s['category_meaning']

        elif s['category'] == i and categories_description.get(i, None):
            item['name'] = s['name']
            item['meaning'] = s['meaning']
            factor_description[i].append(item)

factor_description = json.dumps(factor_description, ensure_ascii=False)
categories_description = json.dumps(categories_description, ensure_ascii=False)

example = [{'category': '技术因子', 'name': 'Mean Reversion',
            'meaning': '指收盘价相对于一段时间内均值的偏离程度，通常用于衡量价格的回归趋势。公式计算为最近20个交易日的收盘价均值减去当前的收盘价。',
            'rqfactor': 'MEAN(CLOSE,20)-CLOSE'},
           {'category': '技术因子', 'name': 'Z-Score Mean Reversion',
            'meaning': 'Z分数均值回归因子通过标准差来衡量当前价格相对于历史均值的偏离程度。它不仅考虑了价格的绝对偏离，还将这种偏离标准化，便于对不同时间段的数据进行比较。',
            'rqfactor': '(CLOSE-MEAN(CLOSE,20))/STD(CLOSE,20)'}]

system_message = 'You are a large language model specialized in extracting financial factor calculation formulas from text descriptions. Your task is to:\n'
system_message += 'Recognize and extract factors from a provided text. Factors are financial metrics such as stock prices, market data, or economic indicators.\n'
system_message += 'Interpret the meaning of factors based on their descriptions and identify the correct calculation formula. Use predefined basic factors and operations provided to you.\n'
system_message += 'Construct the formula for the factor using the following predefined basic factors and operators:\n'
system_message += 'base factors:\n'
system_message += f"''' {factor_description} '''"

system_message += 'operators:\n'
system_message += f"''' {operator_description} '''"

human_message_template = '### CONTEXT ###'
human_message_template += '\n'
human_message_template += '我正在从各种论文内容中，提取并研究影响股市量化交易的因子。\n'
human_message_template += '### OBJECTIVE ###\n'
human_message_template += '我希望你能在论文内容中正确找出有用的因子与其计算公式，让我们逐层思考一下：\n'
human_message_template += '论文内容如下:   \n\n{file_content}\n\n'

human_message = '1. 在阅读，理解文章的基础上，提取出所有表现优秀的因子,并提取出其对应的名称与含义，作用等相关内容，对应名称需要以英文的形式给出。\n'
human_message += '2. 根据以下定义的金融因子的分类情况与介绍，对上述因子进行分类，并对分类结果进行记忆。\n\n'
human_message += f"''' {categories_description} '''\n\n"
human_message += '3. 使用基本因子与基本算子，对上述每个因子构造其计算公式,如果是无法使用给出的基本因子与基本算子构造而成的公式，则舍弃该因子。\n'
human_message += '4. 最后，这个时候我们已经拥有了论文中提到的因子的名称，计算公式，种类，基本信息等内容。将这些内容以 json 格式输出。\n'
human_message += '### AUDIENCE ###\n'
human_message += '目标受众是专业的金融相关的编程者，因子挖掘者，要求生成内容正确，且最后生成的 json 形式字符串可以直接被计算机函数读取。\n'
human_message += '### RESPONSE ###\n'
human_message += f'{format_instruction}\n'
human_message += f'下面是一个生成的例子，请参考这个生成\n\n {example} \n\n'

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=system_message),
        HumanMessagePromptTemplate.from_template(human_message_template),
        HumanMessage(content=human_message)
    ]
)

