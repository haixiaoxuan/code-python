import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# 使用 R 语言的维基百科主页 访问量
df = pd.read_csv('data/example_wp_log_R.csv')


# 新建一列 cap 来指定承载能力的大小(通常情况下这个值应当通过市场规模的数据或专业知识来决定)
# 值得注意的是数据框的每行都必须指定 cap 的值，但并非需要是恒定值。如果市场规模在不断地增长，
# 那么 cap 也可以是不断增长的序列
df["cap"] = 8.5


# 指定采用 logistic 增长
m = Prophet(growth='logistic')
m.fit(df)


# 将未来的承载能力设为8.5，并且预测未来三年的数据
future = m.make_future_dataframe(periods=1826)
future['cap'] = 8.5     # 指定未来三年内的承载力，此列并不一定要是恒定的值
fcst = m.predict(future)
fig = m.plot(fcst)
plt.show()


# ----------------- #
# -- 预测饱和减少 -- #
# ----------------- #

df['y'] = 10 - df['y']
df['cap'] = 6
df['floor'] = 1.5
future['cap'] = 6
future['floor'] = 1.5
m = Prophet(growth='logistic')
m.fit(df)
fcst = m.predict(future)
fig = m.plot(fcst)

plt.show()















