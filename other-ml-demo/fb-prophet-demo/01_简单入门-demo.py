import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

"""
    Prophet 的输入量往往是一个包含两列的数据框：ds 和 y 。ds 列必须包含日期（YYYY-MM-DD）
    或者是具体的时间点（YYYY-MM-DD HH:MM:SS）。
    y 列必须是数值变量，表示我们希望去预测的量。
"""

# 读入数据集
df = pd.read_csv('data/example_wp_log_peyton_manning.csv')
print(df.head())


# 拟合模型
m = Prophet()
m.fit(df)

# 持久化
import pickle
with open("hhhh", "wb") as f:
    pickle.dump(m, f)


# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
future = m.make_future_dataframe(periods=365, freq="d")
print(future.tail())


# 预测数据集, yhat 是预测值， yhat_lower和yhat_upper 是置信区间
forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


# 展示预测结果
fig = m.plot(forecast)
# fig.savefig("hh.png")


fig.set_figwidth(6)
fig.set_figheight(5)
fig.set_tight_layout(True)
plt.xticks(size=10, rotation=30)
plt.tight_layout(1.4)

# 预测的成分分析绘图，展示预测中的趋势、周效应和年度效应
m.plot_components(forecast)
plt.show()


"""
forecast 所包含的列：

['ds', 'trend', 'trend_lower', 'trend_upper', 
       'yhat',  'yhat_lower',  'yhat_upper',
       'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
       'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper', 
       'weekly', 'weekly_lower', 'weekly_upper',
       'yearly', 'yearly_lower', 'yearly_upper']
"""

