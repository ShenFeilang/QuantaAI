import requests
import pandas as pd
import numpy as np

"""""  读取文件并定义处理函数  """

data = pd.read_excel('./resource/market_1.xlsx', sheet_name=0)
# 设置日期为索引
data.set_index('DATE', inplace=True)
# 检查并处理数据中的 NaN 和无穷大值
data = data.replace([np.inf, -np.inf], np.nan)
data.dropna(inplace=True)


# 计算各个因子的 IC 序列
def calculate_IC(data, period=20, method='pearson'):
    n = len(data)
    IC_data = {}
    for i in range(0, n, period):
        end_idx = min(i + period, n)
        if end_idx - i < period:
            break
        subset_returns = data['RETURN'][i:end_idx]

        for col in data.drop(columns='RETURN').columns:
            subset_other = data[col][i:end_idx]
            # 如果因子数据有缺失，使用中位数填充
            if pd.isnull(subset_other).any():
                median_value = subset_other.median()
                subset_other = subset_other.fillna(median_value)
            IC = subset_returns.corr(subset_other, method=method)
            if col not in IC_data:
                IC_data[col] = [np.nan] * i + [IC]
            else:
                IC_data[col].append(IC)
    return pd.DataFrame(IC_data)


""""" 得到IC_series """
rank_IC_series = calculate_IC(data, method='spearman')
IC_series = calculate_IC(data)


# result_IC.to_excel(r'D:\HuaweiMoveData\Users\28577\Desktop\market_IC.xlsx', sheet_name='sheet1', index=False)

# 计算各个因子的 IR，计算公式：IC 均值/标准差
def calculate_IR(data):
    IC_data = calculate_IC(data).fillna(0)
    IR_data = {}
    for col in data.drop(columns='RETURN').columns:
        values = IC_data[col].dropna()
        mean_value = np.mean(values)
        std_value = np.std(values)
        IR = mean_value / std_value
        if col not in IR_data:
            IR_data[col] = [IR]
    return pd.DataFrame(IR_data)


"""" 得到IR_series """
IR_series = calculate_IR(data)
# print(result_IR)
# result_IR.to_excel(r'D:\HuaweiMoveData\Users\28577\Desktop\market_IR.xlsx', sheet_name='sheet1', index=False)
# path = r'D:\HuaweiMoveData\Users\28577\Desktop\market_IC.xlsx'
# IC_series = pd.read_excel(path, sheet_name=0)

""" 再次读入IC,IR并使用 """
# 检查并处理数据中的 NaN 和无穷大值
rank_IC_series = rank_IC_series.replace([np.inf, -np.inf], np.nan)
rank_IC_series.dropna(inplace=True)

categories = pd.read_excel('./resource/categories.xlsx', sheet_name=0)
category_dict = dict(zip(categories['因子名称'], categories['因子类别']))

# num_rows = 50  # 自定义随机评分的用户人数
# question = {
#     'user': [f'user_{i + 1}' for i in range(num_rows)],
#     'FACTOR_1': np.random.randint(1, 6, num_rows),
#     'FACTOR_2': np.random.randint(1, 6, num_rows),
#     'FACTOR_3': np.random.randint(1, 6, num_rows),
#     'FACTOR_4': np.random.randint(1, 6, num_rows),
#     'FACTOR_5': np.random.randint(1, 6, num_rows)
# }

"""" 传进来的参数值 """
value = eval(requests.get('http://localhost:8000/get_data').content.decode())
pre_value = value[0]
risk_value = value[1]

"""" 得到根据IC筛选出的因子final_factors_IC """
question = {}
for i in range(1, 6):
    question[f'FACTOR_{i}'] = pre_value[i - 1]

row = pd.Series(question)

final_factors_IC = []  # 储存最终因子
# for index, row in df.iterrows():
factor1_weight = row.iloc[0] / (row.iloc[0] + row.iloc[1] + row.iloc[2] + row.iloc[3] + row.iloc[4])
factor2_weight = row.iloc[1] / (row.iloc[0] + row.iloc[1] + row.iloc[2] + row.iloc[3] + row.iloc[4])
factor3_weight = row.iloc[2] / (row.iloc[0] + row.iloc[1] + row.iloc[2] + row.iloc[3] + row.iloc[4])
factor4_weight = row.iloc[3] / (row.iloc[0] + row.iloc[1] + row.iloc[2] + row.iloc[3] + row.iloc[4])
factor5_weight = row.iloc[4] / (row.iloc[0] + row.iloc[1] + row.iloc[2] + row.iloc[3] + row.iloc[4])

temp = 1 / factor1_weight + 1 / factor2_weight + 1 / factor3_weight + 1 / factor4_weight + 1 / factor5_weight

factor1_weight = factor1_weight * temp / 5
factor2_weight = factor2_weight * temp / 5
factor3_weight = factor3_weight * temp / 5
factor4_weight = factor4_weight * temp / 5
factor5_weight = factor5_weight * temp / 5

for col in rank_IC_series.columns:
    values = rank_IC_series[col].dropna()
    factor_IC = np.mean(values)  # IC 均值
    category = category_dict[col]
    if category == 1:
        mean_IC = factor_IC * factor1_weight
    elif category == 2:
        mean_IC = factor_IC * factor2_weight
    elif category == 3:
        mean_IC = factor_IC * factor3_weight
    elif category == 4:
        mean_IC = factor_IC * factor4_weight
    elif category == 5:
        mean_IC = factor_IC * factor5_weight

    if abs(mean_IC) >= 0.05:
        final_factors_IC.append(col)

"""""得到根据IR筛选的因子final_factors_IR """
# path = r'D:\HuaweiMoveData\Users\28577\Desktop\market_IR.xlsx'
# IR_series = pd.read_excel(path, sheet_name=0)

# 检查并处理数据中的 NaN 和无穷大值
IR_series = IR_series.replace([np.inf, -np.inf], np.nan)
IR_series.dropna(inplace=True)

# path1 = r'D:\HuaweiMoveData\Users\28577\Desktop\categories.xlsx'
# categories = pd.read_excel(path1, sheet_name=0)
# category_dict = dict(zip(categories['因子名称'], categories['因子类别']))

final_factors_IR = []  # 储存最终因子
# for index, row in df.iterrows():
# user_name = row[0]
factor1_weight = row.iloc[0] / (row.iloc[0] + row.iloc[1] + row.iloc[2] + row.iloc[3] + row.iloc[4])
factor2_weight = row.iloc[1] / (row.iloc[0] + row.iloc[1] + row.iloc[2] + row.iloc[3] + row.iloc[4])
factor3_weight = row.iloc[2] / (row.iloc[0] + row.iloc[1] + row.iloc[2] + row.iloc[3] + row.iloc[4])
factor4_weight = row.iloc[3] / (row.iloc[0] + row.iloc[1] + row.iloc[2] + row.iloc[3] + row.iloc[4])
factor5_weight = row.iloc[4] / (
        row.iloc[0] + row.iloc[1] + row.iloc[2] + row.iloc[3] + row.iloc[4])  # 第 index 个 user 的评分数据

temp = 1 / factor1_weight + 1 / factor2_weight + 1 / factor3_weight + 1 / factor4_weight + 1 / factor5_weight

factor1_weight = factor1_weight * temp / 5
factor2_weight = factor2_weight * temp / 5
factor3_weight = factor3_weight * temp / 5
factor4_weight = factor4_weight * temp / 5
factor5_weight = factor5_weight * temp / 5  # 第 index 个 user 的评分数据

for col in IR_series.columns:
    values = IR_series[col].dropna()
    factor_IR = np.mean(values)  # IR 均值
    category = category_dict[col]
    if category == 1:
        mean_IR = factor_IR * factor1_weight
    elif category == 2:
        mean_IR = factor_IR * factor2_weight
    elif category == 3:
        mean_IR = factor_IR * factor3_weight
    elif category == 4:
        mean_IR = factor_IR * factor4_weight
    elif category == 5:
        mean_IR = factor_IR * factor5_weight

    if abs(mean_IR) >= 0.1:
        final_factors_IR.append(col)

""" 得到final_factors_IR与final_factors_IC 两个列表，并取交集,成为列表"""
print('IC:')
# print(len(final_factors_IC))
print(final_factors_IC)
print('IR:')
# print(len(final_factors_IR))
print(final_factors_IR)
print('+' * 50)
final_factors = list(set(final_factors_IR) & set(final_factors_IC))
print(final_factors)
# print(len(final_factors))
# print('==='*30)
# print('ESG' in final_factors)
# print('CPI' in final_factors)
