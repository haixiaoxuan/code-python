import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from multiprocessing import Process


writer = SummaryWriter(comment='test_tensorboard', flush_secs=1, max_queue=100)


def test1():
    """
        add_scalar
        add_scalars
    """
    for x in range(100):
        writer.add_scalar('y=2x', x * 2, x)
        writer.add_scalar('y=pow(2, x)', 2 ** x, x)
        writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x), "xcosx": x * np.cos(x), "arctanx": np.arctan(x)}, x)


def test2():
    """
        统计直方图与多分位数直线图
        add_histogram
        tag 表示图像的标签名，图的唯一标识；values 表示要统计的参数，global_step 表示y轴，bins 表示取直方图的 bins
        可以统计参数分布以及梯度分布
    """
    for i in range(20):
        writer.add_histogram("param_grad", np.log(np.random.random((5, 5))), i)
        writer.add_histogram("param_param", np.random.random((5, 5)), i)


def test3():
    """
        记录图像
        add_image
        img_tensor这个要注意，表示的我们图像数据，但是要「注意尺度」， 如果我们的图片像素值都是0-1， 那么会默认在这个基础上*255来可视化，毕竟我们的图片都是0-255， 如果像素值有大于1的，那么机器就以为是0-255的范围了，不做任何改动
        global_step: x轴
        dataformats: 数据形式，有 CHW，HWC，HW（灰度图）
    """
    writer.add_image("random_image", np.random.random((3, 200, 200)), 1)
    writer.add_image("random_image", np.random.random((3, 200, 200)), 2)
    writer.add_image("random_image", np.random.random((3, 200, 200)), 3)
    writer.add_image("random_image", np.random.random((3, 200, 200)), 4)
    writer.add_image("random_image", np.random.random((3, 200, 200)), 5)


def test4():
    """
        pip install torchvision
        可视化计算图
        add_graph
        model: 模型，必须时nn.Module
        input_to_model: 输出给模型的数据
        verbose: 是否打印计算图结构信息
    """


def test5():
    """
        pip install torchsummary
        summary
            model: pytorch模型
            input_size: 模型输入size
            batch_size: batch size
            device: "cuda" or "cpu"， 通常选CPU
    """


def test6():
    """
        add_scalar
        add_scalars
    """
    writer = SummaryWriter(comment='test_tensorboard', flush_secs=1, max_queue=100)
    for x in range(100):
        writer.add_scalar('y=2x', x * 2, x)
        writer.add_scalar('y=pow(2, x)', 2 ** x, x)
        writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x), "xcosx": x * np.cos(x), "arctanx": np.arctan(x)}, x)
        print(x)


if __name__ == "__main__":
    p = Process(target=test6)
    p.start()
    p.join()
    # test6()
    writer.close()




