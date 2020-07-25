from fbprophet.plot import plot_forecast_component
from fbprophet.plot import plot_yearly
from fbprophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
from fbprophet.diagnostics import cross_validation


"""
Prophet包含时间序列交叉验证功能，以测量使用历史数据的预测误差。这是通过在历史记录中选择截止点来完成的，
对于每一个都只使用该截止点之前的数据来拟合模型

    这个交叉验证过程可以使用cross_validation函数自动完成一系列历史截断。我们指定预测水平(horizon)，
    然后选择初始训练期(initial)的大小和截断之间的间隔(period)。默认情况下，初始训练期设置为horizon的三倍，
    每半个horizon就有一个截断。
    注：这里需要解释下horizon，initial和period的意义：initial代表了一开始的时间是多少，
    period代表每隔多长时间设置一个cutoff，horizon代表每次从cutoff往后预测多少天。
"""

# 读入数据集
df = pd.read_csv('data/example_wp_log_peyton_manning.csv')


# 交叉验证
m = Prophet()
m.fit(df)
df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days')

# cross_validation的输出是一个dataframe，在每个模拟预测日期和每个截断日期都有真实值y和样本预测值yhat。
# 特别地，对在cutoff 和cutoff + horizon之间的每一个观测点都进行了预测。
# 然后，这个dataframe可以用来度量yhat和y的错误。
df_cv.head()


# performance_metrics作为离截止点(预测的未来距离)的函数，
# 可用于计算关于预测性能的一些有用统计数据(如与y相比时yhat、yhat_lower和yhat_upper)。
# 计算得到的统计信息包括均方误差(mean squared error, MSE)、均方根误差(root mean squared error, RMSE)、
# 平均绝对误差(mean absolute error, MAE)、平均绝对误差(mean absolute percent error, MAPE)以及
# yhat_lower和yhat_upper估计的覆盖率。这些都是在df_cv中通过horizon (ds - cutoff)排序后预测的滚动窗口中计算出来的。
# 默认情况下，每个窗口都会包含10%的预测，但是可以通过rolling_window参数来更改。 
from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p.head()



"""
交叉验证性能指标可以用plot_cross_validation_metric可视化，这里显示的是MAPE。
点表示df_cv中每个预测的绝对误差百分比。蓝线显示的是MAPE，均值被取到滚动窗口的圆点。
我们可以看到，对于一个月后的预测，误差在5%左右，而对于一年之后的预测，误差会增加到11%左右。 
"""
from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')


"""
图中滚动窗口的大小可以通过可选参数rolling_window更改，该参数指定在每个滚动窗口中使用的预测比例。默认值为0.1，
即每个窗口中包含的df_cv的10%行;增大值得话将导致图中平均曲线更平滑。
"""
