from fbprophet.plot import plot_forecast_component
from fbprophet.plot import plot_yearly
from fbprophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd

"""
    默认情况下，Prophet能够满足附加的季节性，这意味着季节性的影响是加到趋势中得到了最后的预报（yhat）。
    航空旅客数量的时间序列是一个附加的季节性不起作用的例子：
    在这个时间序列中，季节性并不是Prophet所假定的是一个恒定的加性因子，而是随着趋势在增长。这就是乘法季节性
    
    Prophet可以通过设置seasonality_mode='multiplicative'来建模乘法季节性:
    Prophet(seasonality_mode='multiplicative')
    
    
    默认情况下，任何添加的季节性或额外的回归量都可以使用seasality_mode设置为加法或者是乘法。
    但假如在添加季节性或回归量时，可以通过指定mode=' addiative '或mode=' ative'作为参数来覆盖之前的设定。 
    例如，这个模块将内置的季节性设置为乘法，但使用一个附加的季度季节性来覆盖原本的乘法，这时候季度季节性就是加法了。
    m = Prophet(seasonality_mode='multiplicative')
    m.add_seasonality('quarterly', period=91.25, fourier_order=8, mode='additive')
     这个时候是时间序列的混合模型：forecast['yhat'] = forecast['trend']  * (1+forecast['multiplicative_terms']) + forecast['additive_terms']。
"""


df = pd.read_csv('data/example_air_passengers.csv')
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(50, freq='MS')     # 表示50个月
forecast = m.predict(future)
fig = m.plot(forecast)

plt.show()







