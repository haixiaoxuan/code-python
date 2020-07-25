

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from gensim.models.word2vec import Word2Vec

import nltk

import matplotlib.pyplot as plt

import pandas as pd

import re
import itertools


"""
    步骤：
        1. 数据预处理
            移除标点 -> 分词 -> 去掉停用词 -> 重组为新句子
        2. 将单词转换为特征两种方式
            1) 使用 bag-of-word
            2) 使用 word2vec
        3. 使用逻辑回归进行预测
    
    note:
        nltk 不支持中文
    
"""

# from nltk.corpus import stopwords
# import nltk
# nltk.download()
# 停用词可以这样下载


def get_data():
    """
    得到数据
    :return: 返回DataFrame，clean_review 是处理之后的列
    """

    df = pd.read_csv("movie-data", delimiter="|")

    # 停用词
    stopwords = {}.fromkeys([i.rstrip() for i in open("english-stopwords")])

    def etl(review):
        # 将标点符号替换为空格
        review = re.sub(r"[^A-Za-z]", " ", review)
        # 全部转换为小写,并切分
        words =[i.strip() for i in review.lower().split(" ") if i.strip() != ""]
        # 去除停用词
        words_nostop = [word for word in words if word not in stopwords]
        return " ".join(words_nostop)

    # 增加一列 clean_review, 处理完成之后的列
    df["clean_review"] = df["review"].apply(etl)
    return df


def get_vec_by_bag_of_word(df):
    """
    词袋模型
        使用 sklearn 来将单词转换为向量
    :return: <class 'numpy.ndarray'>
    """

    # max_features 表示向量的维数，如果单词大于此值，则选择频率最高的前5000
    cv = CountVectorizer(max_features=5000)
    res = cv.fit_transform(df["clean_review"]).toarray()
    return res


def get_vec_by_word2vec(df):
    """
    word2vec 模型，使用 nltk 来实现
    :param df:
    :return:
    """
    def _split_sentences(sentence):
        # 分句
        tokenizer = nltk.data.load("english.pickle")
        sentence_list = tokenizer.tokenize(sentence.strip())

        # 应该执行 去标点，去停用词，转换为小写，  这里省略...
        sentences = [sen for sen in sentence_list if sen]

        # 使用nltk去停用词
        # from nltk.corpus import stopwords
        # eng_stopwords = set(stopwords.words("english"))
        return sentences

    # sum([[],[]], 将嵌套的可迭代对象放入一个列表中)
    sentences = sum(df["review"].apply(_split_sentences), [])

    sentences_list = []
    for sen in sentences:
        # 进行分词（需要将punkt包放入指定目录下）
        sentences_list.append(nltk.word_tokenize(sen))

    # 参数含义:
    #   sg  用于设置训练算法，默认为0，对应CBOW算法，为1表示skip-gram算法
    #   size    指定特征向量维度，默认是100，推荐几十到几百
    #   window  表示当前词与预测词在一个句子中的最大距离
    #   alpha   学习率
    #   min_count   可以对字典做截断，词频少于此值会被丢弃，默认是5
    #   max_vocab_size  设置词向量构建期间的RAM设置，如果超过就丢弃一些不频繁的词
    #   workers     并行数
    #   hs  为1，则采用 hierarchica：softmax技巧，如果设置为0，则negative sample 会被使用
    #   negative    如果>0，会使用negative sample，用于设置多少个noise word
    #   iter    迭代次数
    model = Word2Vec(sentences_list, size=300, min_count=40, window=10)

    # if you don't plan to train the model any further,calling init_sims will
    # make the model much more memory-efficient
    model.init_sims(replace=True)

    model.save("word2v.model")

    # 构建向量
    # 判断单词是否存在于model中   ‘word’ in model
    # 从模型中拿出单词对应向量     model["word"]
    # 构建向量时对所有单词的向量加和求平均
    # 与之相对应的是 tf/rdf (每个单词权重不同)







def train_model(df, feature_data):
    """ 训练模型 """
    x_train, x_test, y_train, y_test = \
        train_test_split(feature_data, df["sentiment"], test_size=0.2, random_state=1)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pre)
    print(conf_matrix)

    # recall accuracy
    recall = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    accuracy = (conf_matrix[1, 1] + conf_matrix[0, 0]) / \
               (conf_matrix[0, 0] + conf_matrix[0, 1] + conf_matrix[1, 0] + conf_matrix[1, 1])
    print("recall:"+str(recall), "accuracy:"+str(accuracy))
    return conf_matrix


def draw_res(matrix):
    """ 将混淆矩阵绘制出来 """

    # 热图，使用色差亮度来显示差异
    # cmap -> gray spring summer autumn winter
    # nearest 将某一块显示为一种颜色，不使用渐变色
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.winter)
    plt.title("confusion_matrix")

    # 添加颜色渐变条
    plt.colorbar()

    plt.xticks([0, 1], [0, 1], rotation=0)
    plt.yticks([0, 1], [0, 1], rotation=0)

    # 将文字写入图片
    thresh = matrix.max() / 2
    for i, j in itertools.product(range(2), range(2)):
        # 设置文字说明
        plt.text(j, i, matrix[i][j], horizontalalignment="center",
                 color="white" if matrix[i][j] > thresh else "black")

    # 自动调整
    plt.tight_layout()

    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.show()




if __name__ == "__main__":
    data = get_data()

    # 使用 word2vec构建向量
    # get_vec_by_word2vec(data)

    feature = get_vec_by_bag_of_word(data)
    confusion_matrix = train_model(data, feature)
    draw_res(confusion_matrix)


