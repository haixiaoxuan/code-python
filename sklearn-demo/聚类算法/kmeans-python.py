import pandas as pd
from sklearn.cluster import KMeans
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

"""
    参数：
        n_clusters              类别个数
        init                    初始点选择，有random | kmeans++ | 指定
        n_init                  进行n次初始化选择，选取其中最好的
        max_iter                每个簇的最大迭代次数
        tol                     阈值，决定收敛条件
        precompute_distances    预先计算距离(更快但消耗更多内存)
        verbose                 详细的
        random_state
        copy_x                  ??
        n_jobs                  可以通过并行计算 n_init
        algorithm               ??
    
    属性方法：
        cluster_centers_        所有簇的中心
        labels_                 每个样本点的label
        inertia_                样本到其最近聚类中心距离的平方和。
        
    轮廓系数：
        sample_size             采样计算(不是全量)
    
    ==================================================================
    坑：
        sklearn 0.19.2  bug
            计算轮廓系数时会将 所有距离装进一个矩阵中，没有优化措施
            如果有n个样本，则 矩阵就是 N x N 维，如果样本数较大，则会发生memory error
    建议：
        1.修改源码
        2.升级版本
    ===================================================================  
"""

warnings.filterwarnings("ignore")
filemane1 = 'C:/Users/xiexiaoxuan/Desktop/cell_building_0520_info.csv'
df_data = pd.read_csv(filemane1)
df_data.to_csv()
df_data = df_data.dropna(axis=0, how='any')
df_data = df_data.reset_index(drop=True)
df_data = df_data[['alt_angle', 'alth', 'height', 'area', 'st_distance']]
ss = StandardScaler()
data = ss.fit_transform(df_data)

for clusters in range(5, 60):
    km = KMeans(n_clusters=clusters, random_state=28)
    km.fit(data)
    # 打印平均轮廓系数平均轮廓系数
    print(km.labels_)
    s = silhouette_score(data, km.labels_)
    print(clusters, '平均轮廓系数:', s)

    predict = km.predict(data)  # 得到聚类结果
    km.cluster_centers_         # 聚类中心



