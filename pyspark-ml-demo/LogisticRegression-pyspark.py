
from pyspark.ml.classification import LogisticRegression

"""
    可以用来做多分类  
    
        family="auto | binomial | multinomial" 
"""

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit([])

lrModel.coefficientMatrix   # 系数矩阵
lrModel.interceptVector     # 截距


trainingSummary = lrModel.summary

trainingSummary.objectiveHistory         # 获得每次迭代的损失函数
trainingSummary.falsePositiveRateByLabel
trainingSummary.truePositiveRateByLabel
trainingSummary.precisionByLabel
trainingSummary.recallByLabel
trainingSummary.fMeasureByLabel()

trainingSummary.accuracy                    # 准确率
trainingSummary.weightedFalsePositiveRate   # FPR
trainingSummary.weightedTruePositiveRate    # TPR
trainingSummary.weightedFMeasure()          # F-meature
trainingSummary.weightedPrecision           # precision
trainingSummary.weightedRecall              # recall

# 设置模型阈值，使F-Measure最大化
fMeasure = trainingSummary.fMeasureByThreshold
maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
    .select('threshold').head()['threshold']
lr.setThreshold(bestThreshold)