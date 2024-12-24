from config import config
from langchain_openai import ChatOpenAI

api_key = config.Zhipu_api_key

llm = ChatOpenAI(
    model="glm-4-plus",
    openai_api_key=api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
    max_tokens=2048
)

