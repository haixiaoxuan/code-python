from captcha.image import ImageCaptcha
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random


def get_random_num(n):
    """
    得到 n 个随机数
    """
    l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    res = []
    for i in range(n):
        res.append(str(random.choice(l)))
    return res


def gen_captcha_image():
    """
    生成验证码图片
    """
    image = ImageCaptcha()

    text = "".join(get_random_num(4))
    captcha = image.generate(text)

    # 将图片保存到磁盘上
    # image.write(text, "hh.txt")

    image_arr = np.array(Image.open(captcha))[:, :, 2]
    # print(image_arr.shape) # (60, 160)
    return image_arr, text


def train_by_tf():
    """ 通过tf训练卷积神经网络模型 """
    pass


if __name__ == "__main__":

    plt.imshow(gen_captcha_image()[0])
    plt.show()
