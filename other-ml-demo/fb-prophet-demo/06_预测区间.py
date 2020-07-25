from fbprophet.plot import plot_forecast_component
from fbprophet.plot import plot_yearly
from fbprophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd


"""
    在预测时，不确定性主要来源于三个部分：趋势中的不确定性、季节效应估计中的不确定性和观测值的噪声影响。
    1. 趋势的不确定性：
        变化速率灵活性更大时（通过增大参数 changepoint_prior_scale 的值），预测的不确定性也随之增大
        原因在于如果将历史数据中更多的变化速率加入了模型，也就代表我们认为未来也会变化得更多，就会使得预测区间成为反映过拟合的标志。
        预测区间的宽度（默认下，是 80% ）可以通过设置 interval_width 参数来控制
        m = Prophet(interval_width=0.95).fit(df)
    
    2. 季节不确定性
        默认情况下， Prophet 只会返回趋势中的不确定性和观测值噪声的影响。你必须使用贝叶斯取样的方法来得到季节效应的不确定性，
        可通过设置 mcmc.samples 参数（默认下取 0 ）来实现。
        m = Prophet(mcmc_samples=500).fit(df)   
"""
















