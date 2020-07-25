from fbprophet.plot import plot_forecast_component
from fbprophet.plot import plot_yearly
from fbprophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd


"""
    专门对节假日或者其它的事件进行建模，你就必须得为此创建一个新的 dataframe，
    其中包含两列（节假日 holiday 和日期戳 ds ），
    可以在这个数据框基础上再新建两列 lower_window 和 upper_window ，从而将节假日的时间扩展成一个区间

    note:
        这个数据框必须包含所有出现的节假日，不仅是历史数据集中还是待预测的时期中的
"""

# 某明星参加所有决赛的日期
playoffs = pd.DataFrame({
    'holiday': 'playoff',
    'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16', '2010-01-24', '2010-02-07', '2011-01-08',
                          '2013-01-12', '2014-01-12', '2014-01-19', '2014-02-02', '2015-01-11', '2016-01-17',
                          '2016-01-24', '2016-02-07']),
    'lower_window': 0,
    'upper_window': 1,
})
# 他参加超级杯的日期
superbowls = pd.DataFrame({
    'holiday': 'superbowl',
    'ds': ['2010-02-07', '2014-02-02', '2016-02-07'],
    'lower_window': 0,
    'upper_window': 1,
})

holidays = pd.concat((playoffs, superbowls))
df = pd.read_csv("data/example_wp_log_peyton_manning.csv")
future = 0


def test1():
    """
    节假日因素
    """
    m = Prophet(holidays=holidays)
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    # 看一下假期的最后10行数据
    print(forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
            ['ds', 'playoff', 'superbowl']][-10:])

    # 查看节假日效应
    fig = m.plot_components(forecast)

    # 查看独立的节假日成分
    plot_forecast_component(m, forecast, 'superbowl')
    plt.show()




def test2():
    """
    季节性的傅里叶级数
    增加傅立叶项的数量可以使季节性适应更快的变化周期，但也可能导致过度拟合
    """
    df = pd.read_csv("data/example_wp_log_peyton_manning.csv")
    m = Prophet().fit(df)
    a = plot_yearly(m)

    # 将傅里叶级数增加到二十，默认是10
    m = Prophet(yearly_seasonality=20).fit(df)
    a = plot_yearly(m)

    plt.show()


def test3():
    """
    自定义季节性因素
    如果时间序列超过两个周期，Prophet将默认适合 每周 和 每年 的季节性
    Prophet为周季节性设定的傅立叶order为3，为年季节性设定的为 10。
    add_seasality的一个可选输入是该季节性组件的先验规模。
    每月的季节性将出现在组件图中
    """
    df = pd.read_csv("data/example_wp_log_peyton_manning.csv")
    m = Prophet(weekly_seasonality=False)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    fig = m.plot_components(forecast)
    plt.show()


def test4():
    """
    对节假日和季节性设定先验规模
    如果发现节假日效应被过度拟合了，通过设置参数 holidays_prior_scale 可以调整它们的先验规模来使之平滑，
    默认下该值取 10 。
    seasonality_prior_scale 参数可以用来调整模型对于季节性的拟合程度
    """
    m = Prophet(holidays=holidays, holidays_prior_scale=0.05)

    # 可以通过在节假日的dataframe中包含一个列prior_scale来单独设置先验规模。
    # 独立的季节性的先验规模可以作为add_seasonality的参数传递。例如，可以使用以下方法设置每周季节性的先验规模:
    m = Prophet()
    m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)


def test5():
    """
    附加的回归量
    可以使用add_regressor方法将附加的回归量添加到模型的线性部分。
    包含回归值的列需要同时出现在拟合数据格式（fit）和预测数据格式(predict)中
    """
    # 判断是否是NFL赛季的周日
    def nfl_sunday(ds):
        date = pd.to_datetime(ds)
        if date.weekday() == 6 and (date.month > 8 or date.month < 2):
            return 1
        else:
            return 0

    df['nfl_sunday'] = df['ds'].apply(nfl_sunday)

    m = Prophet()
    m.add_regressor('nfl_sunday')
    m.fit(df)
    future = m.make_future_dataframe(periods=365)

    future['nfl_sunday'] = future['ds'].apply(nfl_sunday)
    forecast = m.predict(future)
    fig = m.plot_components(forecast)



if __name__ == "__main__":
    test1()
    plt.show()








































