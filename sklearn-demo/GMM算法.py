from sklearn.mixture import GaussianMixture

"""
    GMM 高斯混合模型
        n_components    高斯分布的个数
        covariance_type 各个高斯分布的方差关系，
                        full -> 各个分布之间没有关系
"""
data = []
g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
g.fit(data)

print('类别概率:\t', g.weights_[0])
print('均值:\n', g.means_, '\n')
print('方差:\n', g.covariances_, '\n')


# DPGMM 可以解决自动调整 n_components 的问题
from sklearn.mixture import BayesianGaussianMixture
dpgmm = BayesianGaussianMixture(n_components=3,
                                covariance_type='full',
                                max_iter=1000,
                                n_init=5,
                                weight_concentration_prior_type='dirichlet_process',
                                weight_concentration_prior=10)
