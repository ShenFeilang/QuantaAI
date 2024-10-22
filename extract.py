from langchain_openai import ChatOpenAI
from prompt_1 import prompt
from langchain_core.output_parsers import JsonOutputParser
from classes import Alphas
import json
from zhipuai import ZhipuAI
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm
from config import config

from langchain_core.globals import set_debug

set_debug(True)

api_key = config.api_key

# 填写您自己的APIKey
client = ZhipuAI(api_key=api_key)

resource = []
filepath_en = './resource/En'
filepath_zh = './resource/Zh'
for file in os.listdir(filepath_zh):
    resource.append(os.path.join(filepath_zh, file))
for file in os.listdir(filepath_en):
    resource.append(os.path.join(filepath_en, file))

llm = ChatOpenAI(
    model="glm-4-long",
    openai_api_key=api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
    max_tokens=2048
)
parser = JsonOutputParser(pydantic_object=Alphas)

chain = prompt | llm | parser
# 根据定义的prompt，这里要提供论文内容 file_content

base_alpha = pd.read_excel('./resource/base_alpha_factors.xls')

num = 50
for i in tqdm(resource[50:]):
    file_object = client.files.create(
        file=Path(i), purpose="file-extract")

    # 获取文本内容
    file_content = json.loads(client.files.content(file_id=file_object.id).content)["content"]

    result = chain.invoke({'file_content': file_content})
    alpha_collections = pd.DataFrame(result)
    alpha_collections['source'] = i
    base_alpha = pd.concat([base_alpha, alpha_collections], axis=0)
    num += 1
    if num % 10 == 0:
        base_alpha.to_csv(f'new_all_factors_{num}.csv')

base_alpha.to_csv('all_factors.csv')
