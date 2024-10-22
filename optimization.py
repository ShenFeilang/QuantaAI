from pymoo.core.problem import ElementwiseProblem  # 用于处理批处理问题
from pymoo.algorithms.soo.nonconvex.ga import GA  # 单目标遗传算法
from pymoo.optimize import minimize  # 导入优化函数
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from prompt_2 import prompt, risk_value, chain
from classes import quantitative_trading_system as qt
import warnings
import matplotlib.pyplot as plt
import time

# 忽略所有警告
warnings.filterwarnings("ignore")

system = qt()

# 获得训练集, 测试集, 交易日集, 数据日集
system.load_data_from_csv("train_days", "./resource/data/train_days.csv")
system.load_data_from_csv("test_days", "./resource/data/test_days.csv")
system.load_data_from_csv("trade_days", "./resource/data/trade_days.csv")
system.load_data_from_csv("data_days", "./resource/data/data_days.csv")

system.data["train_days_list"] = system.data["train_days"]["DATE"].tolist()
system.data["test_days_list"] = system.data["test_days"]["DATE"].tolist()
system.data["trade_days_list"] = system.data["trade_days"]["DATE"].tolist()
system.data["data_days_list"] = system.data["data_days"]["DATE"].tolist()

# 获取所有考虑的股票代码
system.load_data_from_csv("stock_code_universe", "./resource/data/stock_code_universe.csv")
system.data["stock_code_universe_list"] = system.data["stock_code_universe"]["CODE"].tolist()

for code in system.data["stock_code_universe_list"]:
    system.load_time_series_data_from_csv(code, "./resource/data/" + code + ".csv")

risk_aversion = (-9 / 40.0) * risk_value + 12.25

# 这里是可以注释的部分，换成以后的result
result = chain.invoke(prompt)  # 字典，result键
num = len(result['result'])

print(result)

# result = {'result': ['PS', 'RSI', 'MACD_DIFF', 'FORCE_INDEX', 'BIAS60', 'BIAS20'],
#           'explain': '根据历史数据分析和IC、IR指标，筛选出未来效益较好的因子如下：\n1. PS（市销率）：PS值越高，说明公司销售能力越强，未来发展潜力越大。\n2. RSI（相对强弱指标）：RSI值越高，说明股票处于超买状态，未来上涨空间较大。\n3. MACD_DIFF（MACD差值）：MACD_DIFF值越大，说明股票处于上涨趋势，未来收益可期。\n4. FORCE_INDEX（强力指数）：FORCE_INDEX值越高，说明股票上涨动力越强，未来收益可能越高。\n5. BIAS60（60日乖离率）：BIAS60值越高，说明股票价格偏离均线越远，未来回调可能性越大，可视为买入信号。\n6. BIAS20（20日乖离率）：BIAS20值越高，说明股票价格偏离均线越远，未来回调可能性越大，可视为买入信号。'}
#
# num = len(result['result'])


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=num, n_obj=1, xl=-1, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        weight_dict = dict(zip(result['result'], x))
        out['F'] = -system.backtest(strategy="multi_factors",
                                    date_list=system.data["train_days_list"],
                                    weight_dict=weight_dict,
                                    risk_aversion=risk_aversion,
                                    return_format="final")


problem = MyProblem()  # 完成实例化
algorithm = GA(pop_size=20, n_offsprings=2, sampling=FloatRandomSampling(),
               crossover=SBX(prob=0.5, eta=15), mutation=PM(eta=20), eliminate_duplicates=True)

res = minimize(problem,
               algorithm=algorithm,
               termination=get_termination('n_gen', 3),
               seed=42,
               save_history=True,
               verbose=True)

print(res.F)
print(res.X)

final_weight_dict = dict(zip(result['result'], res.X))

return_resource = system.backtest(strategy="multi_factors",
                                  date_list=system.data["trade_days_list"],
                                  weight_dict=final_weight_dict,
                                  risk_aversion=risk_aversion,
                                  return_format="all_results")

return_profit = return_resource[0]
return_history = return_resource[1]
return_position = return_resource[2]

print(return_profit)

system.merger_return_table(return_history)  # system.data["return_table"]使用这个来获取
resource_data = system.data['return_table']
resource_data = resource_data.reset_index()
resource_data.columns = ['DATE', 'STRATEGY', 'ESG', 'HS300']
# resource_data.to_csv('resource_data.csv')
resource_data['STRATEGY'] = resource_data['STRATEGY'] * 100
resource_data['ESG'] = resource_data['ESG'] * 100
resource_data['HS300'] = resource_data['HS300'] * 100

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(resource_data['DATE'], resource_data['STRATEGY'], label='Our Strategy', color='#1871ff', linewidth=2)
ax.plot(resource_data['DATE'], resource_data['ESG'], label='ESG Concept Average', color='#72C9DC', linewidth=2)
ax.plot(resource_data['DATE'], resource_data['HS300'], label='CSI 300', color='#CDCED2', linewidth=2)

# Add titles, labels, and grid with percentage format
ax.set_title('Fund Performance Comparison (Percentage)', fontsize=16, pad=20)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Return (%)', fontsize=12)
ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# plt.show()
plt.savefig('./resource/result.png')
