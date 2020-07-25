from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

"""使用gensim构建文本相似度分析模型
        
        使用文本数据，先分词，然后使用word2vec训练模型
        可以用来检测出和某单词最相似的前几个单词
"""


def get_txt_data(path):
    """ 从xml中解析出文本内容 """
    l = []
    with open(path, "r", encoding="utf8") as f:
        for i in f:
            if i.startswith("<contenttitle>") or \
                    i.startswith("<content>"):
                line = i. \
                    replace("<contenttitle>", ""). \
                    replace("<content>", ""). \
                    replace("</contenttitle>", ""). \
                    replace("</content>", "")
                l.append(line)
    return l


def participles(txt):
    """ 对中文文本进行分词 """
    import jieba
    import jieba.analyse

    def cut_word(sentence):
        """ 分词之后用空格连接 """
        return " ".join(jieba.cut(sentence))

    path = "data.txt"
    with open("data.txt", "w", encoding="utf8") as f:
        for line in txt:
            f.writelines(cut_word(line))
            f.writelines("\n")

    return path


def get_mode(path):
    """ 训练并持久化模型 """


    """
        min_count: 在不同大小的语料集中，我们对基准词频的需求也是不一样的。
                比如在较大的语料集中我们希望忽略掉那些只出现过一两次的单词
                可以通过min_count控制，一般范围 0~100
        size: 设置神经网络的层数，默认100，一般范围 10~数百
                    
    """
    model = Word2Vec(LineSentence(path), size=400, window=5, min_count=5)
    model.save("model.dat")

    # 保存所有单词的向量
    # model.wv.save_word2vec_format("model2", binary=False)


def use_model():
    """ 加载使用模型 """
    model = Word2Vec.load("model.dat")
    test_word = ["中国", "上海", "苹果"]
    for word in test_word:
        res = model.most_similar(word)
        print(res)


if __name__ == "__main__":
    data = get_txt_data("news_sougou.dat")
    path = participles(data)
    get_mode("data.txt")
    use_model()

