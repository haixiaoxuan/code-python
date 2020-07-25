
"""
    tsfresh是一个Python的时序数据特征挖掘的模块
"""

from tsfresh.examples.robot_execution_failures import \
    download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute


# download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()

print("=======> ", timeseries)
print("=======> ", y)

extract_f = extract_features(timeseries, column_id="id", column_sort="time")
print("=======> " + extract_f)

impute(extract_f)
select_features(extract_f, y)

