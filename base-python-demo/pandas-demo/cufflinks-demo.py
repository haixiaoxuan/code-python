
"""
    安装：   pip install plotly
            pip install cufflinks
    用途：完成交互式图表
"""

import pandas as pd

import cufflinks as cf
import plotly.offline

cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

df = pd.read_csv("E:\\data\\data_cell_lable_0521_rsrp_five3_all.csv")

df.iplot()