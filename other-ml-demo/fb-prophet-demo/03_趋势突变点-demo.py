from fbprophet.plot import add_changepoints_to_plot
from fbprophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd


"""
    默认情况下， Prophet 将自动监测到这些突变点，并对趋势做适当地调整。
    Prophet 首先是通过在大量潜在的突变点（变化速率突变）中进行识别来监测突变点的。
    之后对趋势变化的幅度做稀疏先验（等同于 L1 正则化）
    ——实际上 Prophet 在建模时会存在很多变化速率突变的点，但只会尽可能少地使用它们。

    note:
         潜在突变点的数量可以通过设置 n_changepoints 参数来指定，但最好还是利用调整正则化过程来修正。
         
    parameter:
        n_changepoints
        changepoint_range
        changepoint_prior_scale
        changepoints
"""


def test1():
    """
    自动检测突变点
    """
    df = pd.read_csv('data/example_wp_log_peyton_manning.csv')
    m = Prophet(n_changepoints=5)
    m.fit(df)

    # 预测
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    # 获取显著突变点的位置
    # 默认情况下，只有在时间序列的前80%才会推断出突变点，以便有足够的长度来预测未来的趋势，并避免在时间序列的末尾出现过度拟合的波动。
    # 这个默认值可以在很多情况下工作，但不是所有情况下都可以，可以使用changepoint_range参数进行更改。
    # 例如，Python中的m = Prophet(changepoint_range=0.9)。这意味着将在时间序列的前90%处寻找潜在的变化点
    fig = m.plot(forecast)
    add_changepoints_to_plot(fig.gca(), m, forecast)

    plt.show()





def test2():
    """
        如果趋势的变化被过度拟合（即过于灵活）或者拟合不足（即灵活性不够），可以利用输入参数 changepoint_prior_scale
        来调整稀疏先验的程度。默认下，这个参数被指定为 0.05 。
        增加这个值，使趋势变化更加灵活。减少这个值会使灵活度降低
    """
    df = pd.read_csv('data/example_wp_log_peyton_manning.csv')

    # 拟合模型
    m = Prophet(changepoint_prior_scale=0.5)
    m.fit(df)

    future = m.make_future_dataframe(periods=365)

    forecast = m.predict(future)
    fig = m.plot(forecast)
    add_changepoints_to_plot(fig.gca(), m, forecast)
    plt.show()


def test3():
    """
        手动指定潜在突变点的位置，而非利用自动的突变点检测，可以使用changepoints 参数
        m = Prophet(changepoints=['2014-01-01'])
    """
    df = pd.read_csv('data/example_wp_log_peyton_manning.csv')
    m = Prophet(changepoints=['2014-01-01'])
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    fig = m.plot(forecast)
    add_changepoints_to_plot(fig.gca(), m, forecast)
    m.plot(forecast)
    plt.show()


if __name__ == "__main__":
    # test1()
    # test2()
    test3()








