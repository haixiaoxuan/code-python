
from fbprophet.plot import plot_forecast_component
from fbprophet.plot import plot_yearly
from fbprophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd

"""
    1. 子日数据 时间戳的格式应该是YYYY-MM-DD - HH:MM:SS
    
    2. 有规律间隔的数据
        # 只保留每天早上6点之前的数据
        df2 = df2[df2['ds'].dt.hour < 6]
        # 以同样的方式调整历史时间框
        
    3. 月数据
        如果测试数据中的日期为月数据，那么预测数据框中的数据也应是 月数据
        future = m.make_future_dataframe(periods=120, freq='M')
"""

df = pd.read_csv('data/example_yosemite_temps.csv')
m = Prophet(changepoint_prior_scale=0.01).fit(df)
future = m.make_future_dataframe(periods=300, freq='H')
fcst = m.predict(future)
fig = m.plot(fcst)
plt.show()














