

"""
    安装：pip install pandas-profiling
    简介：更加简单的使用方法统计出数据更全面的信息
"""

import pandas as pd
import pandas_profiling

df = pd.read_csv("E:\\data\\data_cell_lable_0521_rsrp_five3_all.csv")
profiles = pandas_profiling.ProfileReport(df)

# 导出交互式的 html
profiles.to_file(output_file="report.html")



