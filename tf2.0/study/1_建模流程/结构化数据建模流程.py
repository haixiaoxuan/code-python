import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

"""  泰坦尼克数据
"""

dftrain_raw = pd.read_csv('../../data/titanic/train.csv')
dftest_raw = pd.read_csv('../../data/titanic/test.csv')
print(dftrain_raw.head(10))

# 使用 Pandas对数据进行分析
# ax = dftrain_raw['Survived'].value_counts().plot(kind = 'bar', figsize = (12, 8), fontsize=15, rot = 0)
# ax.set_ylabel('Counts', fontsize = 15)
# ax.set_xlabel('Survived', fontsize = 15)
# plt.show()


# ax = dftrain_raw['Age'].plot(kind = 'hist', bins = 20, color= 'purple', figsize = (12,8), fontsize=15)
# ax.set_ylabel('Frequency',fontsize = 15)
# ax.set_xlabel('Age',fontsize = 15)
# plt.show()

ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
dftrain_raw.query('Survived == 1')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
ax.legend(['Survived==0', 'Survived==1'], fontsize=12)
ax.set_ylabel('Density', fontsize=15)
ax.set_xlabel('Age', fontsize=15)
# plt.show()


def preprocessing(dfdata):
    """ 数据处理
    """
    dfresult = pd.DataFrame()
    # Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])     # one-hot编码
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)
    # Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)
    # Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')
    # SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']
    # Carbin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')
    # Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)
    return (dfresult)


x_train = preprocessing(dftrain_raw)
y_train = dftrain_raw['Survived'].values
x_test = preprocessing(dftest_raw)
y_test = dftest_raw['Survived'].values
print("x_train.shape =", x_train.shape )
print("x_test.shape =", x_test.shape )


# 定义模型
tf.keras.backend.clear_session()
model = models.Sequential()
model.add(layers.Dense(20, activation = 'relu',input_shape=(15,)))
model.add(layers.Dense(10, activation = 'relu' ))
model.add(layers.Dense(1, activation = 'sigmoid' ))
model.summary()


# ⼆分类问题选择⼆元交叉熵损失函数
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
history = model.fit(x_train, y_train, batch_size= 64, epochs= 30,
                    validation_split=0.2 #分割⼀部分训练数据⽤于验证
)


def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()


plot_metric(history, "loss")
plot_metric(history, "AUC")
print("测试集验证：", model.evaluate(x = x_test, y = y_test))


# 预测概率
model.predict(x_test[0:10])
# 预测类别
model.predict_classes(x_test[0:10])


# 保存模型
model.save('./data/keras_model.h5')
del model #删除现有模型
# identical to the previous one
model = models.load_model('./data/keras_model.h5')
model.evaluate(x_test, y_test)


# 保存模型结构
json_str = model.to_json()
# 恢复模型结构
model_json = models.model_from_json(json_str)


#保存模型权᯿
model.save_weights('./data/keras_model_weight.h5')
# 恢复模型结构
model_json = models.model_from_json(json_str)
model_json.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['AUC']
)
# 加载权᯿
model_json.load_weights('./data/keras_model_weight.h5')
model_json.evaluate(x_test,y_test)


