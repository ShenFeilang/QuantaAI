from fastapi import FastAPI, Form
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 创建一个 FastAPI 应用
app = FastAPI()

# 定义一个全局变量来存储传递的 label
pre_value = None
risk_value = None

# 允许跨域的来源（前端的域名或IP地址）
origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许的来源
    allow_credentials=True,  # 允许携带cookie
    allow_methods=["*"],  # 允许的HTTP方法（POST, GET, DELETE等）
    allow_headers=["*"],  # 允许的请求头
)


# 处理 URL 参数的 FastAPI 路由
@app.post("/", response_class=JSONResponse)
async def read_label(preference: str = Form(...), riskTolerance: str = Form(...)):
    global pre_value
    global risk_value
    pre_value = eval(preference)[1:]
    risk_value = int(riskTolerance)
    print(pre_value)
    print(risk_value)
    if pre_value and risk_value:
        return JSONResponse(content={
            "success": True,
            "message": "提交成功",
            "data": {
                "preference": pre_value,
                "riskTolerance": risk_value
            }
        })
    else:
        return JSONResponse(content={
            "success": False,
            "message": "提交失败，缺少参数"
        })


@app.get('/get_data')
async def get_data():
    global pre_value
    global risk_value
    return pre_value, risk_value


uvicorn.run(app, host="127.0.0.1", port=8000)
