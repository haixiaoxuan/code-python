
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

"""
    高斯朴素贝叶斯
    多项式朴素贝叶斯
    伯努利朴素贝叶斯
        如果特征为连续分布，则可以使用高斯贝叶斯
"""


mnb = MultinomialNB(alpha=1)