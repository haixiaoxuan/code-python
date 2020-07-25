import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import AutoDateLocator, AutoDateFormatter, MonthLocator, num2date
import numpy as np
from matplotlib.ticker import FuncFormatter



# 显示中文
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def save_confusion_matrix(matrix, label, path):
    """
    保存混淆矩阵到指定路径
    :return:
    """
    # 设置大小
    plt.figure(figsize=(5, 5))
    plt.imshow(matrix, interpolation="nearest", cmap=plt.get_cmap('Blues'))
    plt.title("confusion_matrix")

    # 添加颜色渐变条
    plt.colorbar()

    plt.xticks(range(len(matrix)), label, rotation=0)
    plt.yticks(range(len(matrix)), label, rotation=0)

    # 将矩阵中的值添加进入图片
    for i, j in itertools.product(range(len(matrix)), range(len(matrix))):
        plt.text(j, i, matrix[i][j], horizontalalignment="center",
                 color="black")

    # 自动调整
    plt.tight_layout()
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")

    # 调节图片边距
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(path)
    print("=======> 保存混淆矩阵图片到 ", path)
    plt.close()


def save_text_to_pic(text, path):
    """ 保存文本成图片 """

    # 设置大小
    plt.figure(figsize=(5, 5))

    plt.axis([0, 2, 0, 2])
    plt.text(1, 1, text, ha="center", va='center', fontsize=12, wrap=True)

    # 去掉坐标轴
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(path)
    print("=======> 保存分类报告图片到 ", path)
    plt.close()


def save_roc_to_pic(fpr, tpr, roc_auc, path):
    """ 绘制 ROC曲线并保存成图片到指定路径 """

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='b',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线

    # 画出对角虚线
    plt.plot([0, 1], [0, 1], color='r', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(path)
    print("=======> 保存ROC曲线图片到 ", path)
    plt.close()


def plot(m, fcst, splitFlag, ax=None, uncertainty=True, plot_cap=True, xlabel='时间', ylabel='流量',
         figsize=(10, 6)):
    """ prophet 绘图 """
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    fcst_t = fcst['ds'].dt.to_pydatetime()
    ax.plot(m.history['ds'].dt.to_pydatetime(), m.history['y'], 'k.')
    # ax.plot(fcst_t, fcst['yhat'], ls='-', c='#0072B2')
    ax.plot(fcst_t[0: -splitFlag], fcst['yhat'][0: -splitFlag], ls='-', c='#0072B2')
    ax.plot(fcst_t[-splitFlag:], fcst['yhat'][-splitFlag:], ls='-', c='r')
    if uncertainty:
        ax.fill_between(fcst_t, fcst['yhat_lower'], fcst['yhat_upper'],
                        color='#0072B2', alpha=0.2)

    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_components(m, fcst, uncertainty=True, plot_cap=True, weekly_start=0, yearly_start=0, figsize=None):
    """ prophet 绘制成分组件 """
    components = ['trend']
    if m.train_holiday_names is not None and 'holidays' in fcst:
        components.append('holidays')
    if 'weekly' in m.seasonalities and 'weekly' in fcst:
        components.append('weekly')
    if 'yearly' in m.seasonalities and 'yearly' in fcst:
        components.append('yearly')

    components.extend([
        name for name in sorted(m.seasonalities)
        if name in fcst and name not in ['weekly', 'yearly']
    ])

    regressors = {'additive': False, 'multiplicative': False}
    for name, props in m.extra_regressors.items():
        regressors[props['mode']] = True
    for mode in ['additive', 'multiplicative']:
        if regressors[mode] and 'extra_regressors_{}'.format(mode) in fcst:
            components.append('extra_regressors_{}'.format(mode))
    npanel = len(components)

    figsize = figsize if figsize else (9, 3 * npanel)
    fig, axes = plt.subplots(npanel, 1, facecolor='w', figsize=figsize)

    if npanel == 1:
        axes = [axes]

    multiplicative_axes = []

    for ax, plot_name in zip(axes, components):
        if plot_name == 'trend':
            plot_forecast_component(
                m=m, fcst=fcst, name='trend', ax=ax, uncertainty=uncertainty,
                plot_cap=plot_cap, ylabel="趋势分量"
            )
        elif plot_name in m.seasonalities:
            if plot_name == 'weekly' or m.seasonalities[plot_name]['period'] == 7:
                plot_weekly(
                    m=m, name=plot_name, ax=ax, uncertainty=uncertainty, weekly_start=weekly_start
                )
            elif plot_name == 'yearly' or m.seasonalities[plot_name]['period'] == 365.25:
                plot_yearly(
                    m=m, name=plot_name, ax=ax, uncertainty=uncertainty, yearly_start=yearly_start
                )
            else:
                plot_seasonality(
                    m=m, name=plot_name, ax=ax, uncertainty=uncertainty,
                )
        elif plot_name in [
            'holidays',
            'extra_regressors_additive',
            'extra_regressors_multiplicative',
        ]:
            name = plot_name
            if plot_name == 'holidays':
                name = "节假日分量"
            plot_forecast_component(
                m=m, fcst=fcst, name=plot_name, ax=ax, uncertainty=uncertainty,
                plot_cap=False, ylabel=name
            )
        if plot_name in m.component_modes['multiplicative']:
            multiplicative_axes.append(ax)

    fig.tight_layout()
    # Reset multiplicative axes labels after tight_layout adjustment
    for ax in multiplicative_axes:
        ax = set_y_as_percent(ax)
    return fig


def plot_forecast_component(m, fcst, name, ax=None, uncertainty=True, plot_cap=False,
                            figsize=(10, 6), ylabel=""):
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    fcst_t = fcst['ds'].dt.to_pydatetime()
    artists += ax.plot(fcst_t, fcst[name], ls='-', c='#0072B2')
    if 'cap' in fcst and plot_cap:
        artists += ax.plot(fcst_t, fcst['cap'], ls='--', c='k')
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
    if uncertainty:
        artists += [ax.fill_between(
            fcst_t, fcst[name + '_lower'], fcst[name + '_upper'],
            color='#0072B2', alpha=0.2)]
    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel('时间')

    ax.set_ylabel(name if ylabel == "" else ylabel)
    if name in m.component_modes['multiplicative']:
        ax = set_y_as_percent(ax)
    return artists


def plot_weekly(m, ax=None, uncertainty=True, weekly_start=0, figsize=(10, 6), name='weekly'):

    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    # Compute weekly seasonality for a Sun-Sat sequence of dates.
    days = (pd.date_range(start='2017-01-01', periods=7) +
            pd.Timedelta(days=weekly_start))
    df_w = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_w)
    days = days.weekday_name
    artists += ax.plot(range(len(days)), seas[name], ls='-',
                    c='#0072B2')
    if uncertainty:
        artists += [ax.fill_between(range(len(days)),
                                    seas[name + '_lower'], seas[name + '_upper'],
                                    color='#0072B2', alpha=0.2)]
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days)
    ax.set_xlabel('时间')
    ax.set_ylabel("周分量")
    if m.seasonalities[name]['mode'] == 'multiplicative':
        ax = set_y_as_percent(ax)
    return artists


def plot_yearly(m, ax=None, uncertainty=True, yearly_start=0, figsize=(10, 6), name='yearly'):
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
    days = (pd.date_range(start='2017-01-01', periods=365) +
            pd.Timedelta(days=yearly_start))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    artists += ax.plot(
        df_y['ds'].dt.to_pydatetime(), seas[name], ls='-', c='#0072B2')
    if uncertainty:
        artists += [ax.fill_between(
            df_y['ds'].dt.to_pydatetime(), seas[name + '_lower'],
            seas[name + '_upper'], color='#0072B2', alpha=0.2)]
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    months = MonthLocator(range(1, 13), bymonthday=1, interval=2)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos=None: '{dt:%B} {dt.day}'.format(dt=num2date(x))))
    ax.xaxis.set_major_locator(months)
    ax.set_xlabel('时间')
    ax.set_ylabel("年分量")
    if m.seasonalities[name]['mode'] == 'multiplicative':
        ax = set_y_as_percent(ax)
    return artists


def plot_seasonality(m, name, ax=None, uncertainty=True, figsize=(10, 6)):

    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    # Compute seasonality from Jan 1 through a single period.
    start = pd.to_datetime('2017-01-01 0000')
    period = m.seasonalities[name]['period']
    end = start + pd.Timedelta(days=period)
    plot_points = 200
    days = pd.to_datetime(np.linspace(start.value, end.value, plot_points))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    artists += ax.plot(df_y['ds'].dt.to_pydatetime(), seas[name], ls='-',
                        c='#0072B2')
    if uncertainty:
        artists += [ax.fill_between(
            df_y['ds'].dt.to_pydatetime(), seas[name + '_lower'],
            seas[name + '_upper'], color='#0072B2', alpha=0.2)]
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    xticks = pd.to_datetime(np.linspace(start.value, end.value, 7)
        ).to_pydatetime()
    ax.set_xticks(xticks)
    if period <= 2:
        fmt_str = '{dt:%T}'
    elif period < 14:
        fmt_str = '{dt:%m}/{dt:%d} {dt:%R}'
    else:
        fmt_str = '{dt:%m}/{dt:%d}'
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos=None: fmt_str.format(dt=num2date(x))))
    ax.set_xlabel('时间')
    ax.set_ylabel("季节分量")
    if m.seasonalities[name]['mode'] == 'multiplicative':
        ax = set_y_as_percent(ax)
    return artists


def seasonality_plot_df(m, ds):

    df_dict = {'ds': ds, 'cap': 1., 'floor': 0.}
    for name in m.extra_regressors:
        df_dict[name] = 0.
    # Activate all conditional seasonality columns
    for props in m.seasonalities.values():
        if props['condition_name'] is not None:
            df_dict[props['condition_name']] = True
    df = pd.DataFrame(df_dict)
    df = m.setup_dataframe(df)
    return df



def set_y_as_percent(ax):
    yticks = 100 * ax.get_yticks()
    yticklabels = ['{0:.4g}%'.format(y) for y in yticks]
    ax.set_yticklabels(yticklabels)
    return ax


