from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import codecs
from pydantic import BaseModel, Field
from typing import List
from filter import final_factors,risk_value
from langchain_openai import ChatOpenAI
from config import config


class FinalAlpha(BaseModel):
    """ 最终经过模式筛选出来的因子列表 """
    result: List = Field(description='大模型经过往年数据，对当前数据进行预测，并筛选出未来效益很好的多个因子',
                         enum=final_factors)
    explain: str = Field(description='大模型在经过对往年数据进行分析，给出对当前筛选结果的理由和解释')


parser = JsonOutputParser(pydantic_object=FinalAlpha)


def unicode_to_chinese_safe(unicode_str):
    # 将字符串中的 \\u 替换为 \u
    unicode_str = unicode_str.replace(r'\\u', r'\u')
    # 使用codecs模块解码Unicode字符串
    return codecs.decode(unicode_str, 'unicode_escape')


format_instruction = unicode_to_chinese_safe(parser.get_format_instructions())

market_2020_2022 = pd.read_csv('./resource/market_2020_2022.csv')
market_2021_2023 = pd.read_csv('./resource/market_2021_2023.csv')

market_IC = pd.read_excel('./resource/market_IC.xlsx').iloc[22:, :]
market_IR = pd.read_excel('./resource/market_IR.xlsx')

market_2020_2022 = market_2020_2022[final_factors].to_json(orient='records', force_ascii=False)
market_2021_2023 = market_2021_2023[final_factors].to_json(orient='records', force_ascii=False)
market_IR = market_IR[final_factors].to_json(orient='records', force_ascii=False)
market_IC = market_IC[final_factors].to_json(orient='records', force_ascii=False)

system_message = '你的主要职责是对股票市场数据进行深入、定量和定性的分析、预测与筛选。你应该将前沿的金融理论，计算方法和风险评估融入你的分析中，与拥有金融工程和定量金融高级学位的专业人士的专业知识相媲美。你需要做到以下几点：\n\n'
system_message += '1. 具有强大的数据处理和分析能力，可以通过分析因子财报数据与因子效益数据，来预测出未来可能效益会很高的优质因子。这包括精通统计分析，预测建模等策略。\n'
system_message += '2. 具有强大的研究能力，可以利用MFE项目专业课程的知识，对给定因子数据进行详细分析，预测和筛选。\n'
system_message += '3. 沟通和报告能力很强，可以用 中文 给出详细的解释和报告，内容清晰准确，并且专业投资者和知情的外行都能理解，并在分析中披露任何潜在的利益冲突。'

system_message = SystemMessage(content=system_message)

human_message = '### CONTENT ###\n'
human_message += '我正在分析2020~2022年这两年各因子的财报数据与2022~2023这一年各因子的效益相关数据，试图从中发现金融规律并进行预测，在掌握规律的基础上，最后根据2021~2023年这两年各因子的财报数据，对2023~2024年因子效益数据进行预测，根据预测结果，便可以筛选出可能会在2023~2024这年表现优秀的因子,最终得到的筛选因子不会超过6个。\n'
human_message += '### OBJECTIVE ###\n'
human_message += '我希望你能帮我完成这个任务，给出你觉得在2023~2024年表现优秀的因子，并给出你根据数据做出的分析，预测与筛选的解释。现在让我们逐步思考一下：\n'
human_message += '1. 先理解什么是各因子财报数据与各因子的效益相关数据。因子财报数据指的是各个因子，在不同交易日统计下的具体数值；因子效益数据是不同时期下因子与收益的相关度，即IC,IR等指标。‌ 这些因子涵盖了从公司的基本面信息到市场情绪等多个方面，帮助投资者理解市场动态和制定投资策略‌。\n'
human_message += "2. 下方使用'''包裹的 JSON 格式内容是2020~2022年这两年各因子的财报数据,顺序记录的是从2020~2022年每个交易日下统计的因子值。\n"
human_message += f"''' {market_2020_2022} '''\n记忆该数据里面包含的因子，并对整个数据集进行理解。\n"
human_message += "3. 接下来使用'''包裹传递的是2022~2023这一年各因子的效益相关数据，这个包含两个指标，分别为IC指标数据与IR指标数据，IC指标数据集 顺序记录的是2022~2023这一年各因子与收益的IC值:\n"
human_message += f"''' {market_IC} '''\n"
human_message += 'IR指标数据集是今年所有IC数据指标的平均值:\n'
human_message += f"''' {market_IR} '''\n"
human_message += '4. 在完成上述所有文件的分析与总结过后，你需要从原来包含的因子里筛选出你觉得未来效益好的因子，并给出筛选的理由和解释，你可以使用金融知识对这些因子进行排行，最终按照排行给出最多不超过6个因子的推荐。\n'
human_message += '### AUDIENCE ###\n'
human_message += '目标受众是专业的金融行业从业者，需要你根据数据内容筛选出正确且少量的优秀因子，最终筛选出的优秀因子个数不超过6个，且给出的中文解释也要详细具体，清晰合理。\n'
human_message += '### RESPONSE ###\n'
human_message += f'{format_instruction}'

prompt = [
    system_message,
    HumanMessage(content=human_message)
]

api_key=config.api_key

llm = ChatOpenAI(
    model="glm-4-plus",
    openai_api_key=api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.2,
    max_tokens=2048
)

chain = llm | parser



