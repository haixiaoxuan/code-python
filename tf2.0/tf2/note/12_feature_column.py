import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    特征处理：tf.feature_column
        对类别特征进行one-hot编码
    交叉特征使用
    
    除了bucketized_column 外的函数要么返回一个 Categorical Column 对象，要么返回一个DenseColumn对象
    
    categorical column 
        tf.feature_column.categorical_column_with_identity          输入列就是固定的离散值，比如 0,1,2,3
        tf.feature_column.categorical_column_with_vocabulary_list   输入列是字符串，先将字符串映射为整形，然后在进行 one-hot
        tf.feature_column.categorical_column_with_vocabulary_file
        tf.feature_column.categorical_column_with_hash_bucket       类别数据量过大，可以使用此函数
        tf.feature_column.crossed_column
        
    DenseColumn
        tf.feature_column.numeric_column
        tf.feature_column.indicator_column
        tf.feature_column.embedding_column
"""


train_file = "./data/Titanic/train.csv"
eval_file = "./data/Titanic/eval_csv"

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)

y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

# 对数据统计分析
train_df.describe()
print(train_df.shape, eval_df.shape)
train_df.agg.hist(bins=20)
train_df.sex.value_counts().plot(kind='barh')
train_df['class'].value_counts().plot(kind='barh')
pd.concat([train_df, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh')


categorial_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
numeric_columns = ['age', 'fare']
# 存储对每个特征的处理
feature_columns = []

for categorial_column in categorial_columns:
    vocab = train_df[categorial_column].unique()        # 对类别数据进行去重
    feature_columns.append(
        tf.feature_column.indicator_column(     # 对离散特征进行one-host编码
            tf.feature_column.categorical_column_with_vocabulary_list(categorial_column, vocab)))


for categorial_column in numeric_columns:
    feature_columns.append(
        tf.feature_column.numeric_column(categorial_column, dtype=tf.float32))


def make_dataset(data_df, label_df, epochs=10, shuffle=True, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(data_df), label_df))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset


train_dataset = make_dataset(train_df, y_train, batch_size=5)
for x, y in train_dataset.take(1):
    print(x, y)


# 真正的进行特征处理的环节
for x, y in train_dataset.take(1):
    age_column = feature_columns[7]
    gender_cloumn = feature_columns[0]
    print(keras.layers.DenseFeatures(age_column)(x).numpy())
    print(keras.layers.DenseFeatures(gender_cloumn)(x).numpy())

# 对特征进行处理
for x, y in train_dataset.take(1):
    print(keras.layers.DenseFeatures(feature_columns)(x).numpy())




# 使用离散特征做笛卡儿积
# cross feature: age [1,2,3,4,5],gender:[male,female]
# age_x_gender:[(1,male),(2,male),...，(5,male),...,(5,female)]
feature_columns.append(
    tf.feature_column.indicator_column(
        tf.feature_column.crossed_column(['age', 'sex'], hash_backet_size=100)))






