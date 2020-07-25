import pkuseg

"""
    中文分词工具，目前主流的还有 THULAC | jieba
        pkuseg目前提供三个预训练模型
            1. 使用MSRA（新闻语料训练出来的模型）
            2. 使用CTBB（新闻文本及网络文本的混合型语料）
            3. 使用微博（网络文本语料）
        此外，也可以通过全新的标注数据重新训练模型
        
    github: https://github.com/lancopku/PKUSeg-python
"""

# 1.
seg = pkuseg.pkuseg()   # 以默认配置加载模型
text = seg.cut("我在上海")
print(text)

# 2.
lexicon = ["在上海"]
seg = pkuseg.pkuseg(user_dict=lexicon)  # 希望分词时lexicon中的固定词不分开
print(seg.cut("我在上海"))

# 3.
pkuseg.pkuseg(model_name="./ctb8")  # 直接加载已经训练好模型


# 4. 自己训练模型
pkuseg.train("x_path", "y_path", "./models", nthread=20)








