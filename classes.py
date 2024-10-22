from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
from scipy.optimize import minimize


class BaseAlpha(BaseModel):
    """ 这是用来规定大模型生成Alpha因子格式的类 """
    category: str = Field(description='因子的种类', enum=['财务因子', '技术因子', '宏观因子'])
    name: str = Field(description='因子的名称')
    meaning: str = Field(description='对因子基本含义与作用的介绍')
    rqfactor: str = Field(description='由基本算子与基本因子构成的因子的计算公式')


class Alphas(BaseModel):
    """ 用来组装成上述BaseAlpha的类"""
    content: List[BaseAlpha] = Field(description='由上述BaseAlpha类组成的列表')


class quantitative_trading_system:
    """
    证券量化交易系统
    """

    def __init__(self):
        self.data = {}
        self.parameters = {}
        pass

    def add_parameter(self, key, value):
        if key in self.parameters:
            raise KeyError(f"The key '{key}' already exists in the dictionary.")
        else:
            self.parameters[key] = value
        pass

    def change_parameter(self, key, value):
        if key not in self.parameters:
            raise KeyError(f"The key '{key}' is not found in the dictionary.")
        else:
            self.parameters[key] = value
        pass

    def get_parameter(self, key):
        if key not in self.parameters:
            raise KeyError(f"The key '{key}' is not found in the dictionary.")
        return self.parameters[key]

    def load_data_from_csv(self, field, filepath):
        self.data[field] = pd.read_csv(filepath)
        pass

    def load_time_series_data_from_csv(self, field, filepath):
        self.data[field] = pd.read_csv(filepath)
        self.data[field]["DATE"] = pd.to_datetime(self.data[field]["DATE"])
        self.data[field].set_index("DATE", inplace=True)
        pass

    def factor_analysis(self, factor, returns):
        pass

    def calculate_scores(self, date_list, weight_dict):
        scores = pd.DataFrame(index=pd.to_datetime(date_list),
                              columns=self.data["stock_code_universe_list"]).fillna(0)

        for key, value in weight_dict.items():
            for code in self.data["stock_code_universe_list"]:
                scores[code] = scores[code] + self.data[code][key] * value
                pass
            pass

        scores.to_csv("scores.csv")

        return scores

    def calculate_return(self, date, position):
        sum = 0
        for code, weight in position.items():
            sum = sum + weight * self.data[code].loc[date, "RETURN"]
            pass
        return sum

    def portfolio_performance(self, weights, mean_returns, cov_matrix):
        returns = np.dot(weights, mean_returns)  # 组合的预期回报
        variance = np.dot(weights.T, np.dot(cov_matrix, weights))  # 组合的方差（风险）
        return returns, variance

    def utility_function(self, weights, mean_returns, cov_matrix, risk_aversion):
        returns, variance = self.portfolio_performance(weights, mean_returns, cov_matrix)
        return -(returns - risk_aversion * variance / 2)  # 最大化效用时，最小化负效用

    def merger_return_table(self, return_history):
        self.load_time_series_data_from_csv("return_table", "./resource/data/market_return.csv")
        self.data["return_table"]["STRATEGY"] = return_history

    def backtest(self, strategy, date_list, weight_dict, risk_aversion, return_format="history"):
        if strategy == "multi_factors":
            return self.backtest_strategy_multi_factors(date_list, weight_dict, risk_aversion, return_format)
        else:
            raise ValueError(f"策略'{strategy}'不合法.")
        pass

    def backtest_strategy_multi_factors(self, date_list, weight_dict, risk_aversion, return_format):
        scores = self.calculate_scores(date_list, weight_dict)

        multiplier = 1.0
        multiplier_history = pd.DataFrame(index=pd.to_datetime(date_list))
        multiplier_history["MULTIPLIER"] = 1.0

        position = {}

        for date in date_list:
            # 计算date日期收盘时, 持仓的回报
            multiplier = multiplier * (1.0 + self.calculate_return(date, position))
            multiplier_history.loc[date, "MULTIPLIER"] = multiplier

            # (想象)持仓以收盘价全部出售
            position = {}

            # 计算date日期收盘时, 各股票分数
            # 已经计算并存储在 scores 中

            # 计算date日期收盘时, 新的持仓股票(新的持仓将获得明天的收益)
            invest_set = scores.loc[date].nlargest(10).index.tolist()

            # 计算date日期收盘时, 计算各股票的持仓
            INVS_returns = pd.DataFrame(index=pd.to_datetime(date_list),
                                        columns=invest_set).fillna(value=0)
            for code in invest_set:
                INVS_returns[code] = self.data[code]["RETURN"]
                pass

            INVS_mean_returns = INVS_returns.mean()
            INVS_covariance = INVS_returns.cov()

            # 设置优化约束条件
            num_assets = len(INVS_mean_returns)
            initial_weights = np.ones(num_assets) / num_assets  # 初始权重均分
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})  # 权重之和为1
            bounds = tuple((0.0, 1.0) for asset in range(num_assets))  # 每只股票的权重在0到1之间

            result = minimize(self.utility_function,
                              initial_weights,
                              args=(INVS_mean_returns, INVS_covariance, risk_aversion),
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints)

            index = 0
            for code in invest_set:
                position[code] = result.x[index]
                index = index + 1
                pass

            pass

        if return_format == "history":
            return (multiplier_history["MULTIPLIER"] - 1).to_list()  # total_return
        elif return_format == "final":
            return multiplier_history["MULTIPLIER"].iloc[-1] - 1  # return_history
        elif return_format == "all_results":
            return multiplier_history["MULTIPLIER"].iloc[-1] - 1, (multiplier_history[
                                                                       "MULTIPLIER"] - 1).to_list(), position  # total_return, return_history, position
        else:
            raise ValueError(f"输出格式'{return_format}'不合法.")
