import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import time
from config import config

api_key=config.api_key

llm = ChatOpenAI(
    model="glm-4-long",
    openai_api_key=api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
    max_tokens=2048
)


def portfolio_explanation(portfolio, predicted_return):
    # 将每个股票的持仓转化为字符串列表
    holdings = [f"\t{stock}: {percentage*100:.2f}%" for stock, percentage in portfolio.items() if percentage > 0.01]

    # 将列表转为逗号分隔的字符串
    holdings_str = ",\n ".join(holdings)

    # 创建解释性文本
    explanation = f"根据模型预测，您的股票持仓组合包括\n {holdings_str}。\n预计整体年化收益率为 {predicted_return:.2f}%。"

    return explanation


# 使用函数生成解释性文本


def predict(message, history):
    history_langchain_format = []
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
    history_langchain_format.append(HumanMessage(content=message))
    partial_message = ""
    if 'select alpha' in message:
        from optimization import return_profit, return_position
        answer = portfolio_explanation(return_position, return_profit)
        for chunk in answer:
            time.sleep(0.05)
            partial_message = partial_message + chunk
            yield partial_message

    else:
        for chunk in llm.stream(history_langchain_format):
            if chunk is not None:
                partial_message = partial_message + chunk.content
                yield partial_message


def change_photo():
    return gr.Image(value='./resource/result.png', label='业绩走势',height=180)


with gr.Blocks(theme=gr.themes.Soft(), fill_height=True, fill_width=True) as demo:
    with gr.Column(scale=4):
        gr.ChatInterface(predict, type="messages", title='智能投资策略', fill_height=True)
    with gr.Column(scale=1):
        image = gr.Image(value='./resource/default.png', label='业绩走势',height=180)
        image.select(change_photo, outputs=image)

demo.launch(server_name="127.0.0.1", server_port=7860)
