from PIL import Image
import numpy as np

"""
    模式
        1             1位像素，黑和白，存成8位的像素
        L             8位像素，黑白
        P             8位像素，使用调色板映射到任何其他模式
        RGB           3×8位像素，真彩
        RGBA          4×8位像素，真彩+透明通道
        CMYK          4×8位像素，颜色隔离
        YCbCr         3×8位像素，彩色视频格式
        I             32位整型像素
        F             32位浮点型像素
"""

# 查看图片模式
Image.fromarray().mode

# 模式转换
Image.fromarray().convert("RGB")

# 保存图片
Image.fromarray().save()

# 打开图片
im = Image.open("xx.png")

# 将图片转为数组
np.array(im)



