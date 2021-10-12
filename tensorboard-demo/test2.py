import numpy as np
import tensorflow as tf


writer = tf.summary.create_file_writer("./tf_log")
writer_test1 = tf.summary.create_file_writer("./tf_log/tf_log_test1")
writer_test2 = tf.summary.create_file_writer("./tf_log/tf_log_test2")


def test0():
    writer_1 = tf.summary.create_file_writer("./mylogs")  # （1）创建一个 SummaryWriter 对象，生成的日志将储存到 "./mylogs" 路径中
    with writer_1.as_default():  # （2）使用 writer_1 记录with包裹的context中，进行 summary 写入的操作
        for step in range(100):
            # other model code would go here
            tf.summary.scalar("loss1", 1, step=step)  # （3）将scalar("loss", loss, step)写入 summary
            tf.summary.scalar("loss2", 2, step=step)
            tf.summary.scalar("loss3", 3, step=step)
            writer.flush()


def test1():
    """
        add_scalar
        add_scalars
    """
    with writer.as_default():
        for x in range(100):
            tf.summary.scalar('y=2x', x * 2, x)
            tf.summary.scalar('y=pow(2, x)', x**2, x)
            writer.flush()


def test1_1():
    """

    """
    with writer_test1.as_default():
        for x in range(100):
            tf.summary.scalar('data/test', x * np.sin(x), x)
    with writer_test2.as_default():
        for x in range(100):
            tf.summary.scalar('data/test', x * np.cos(x), x)


def test2():
    """
        统计直方图与多分位数直线图
        add_histogram
        tag 表示图像的标签名，图的唯一标识；values 表示要统计的参数，global_step 表示y轴，bins 表示取直方图的 bins
        可以统计参数分布以及梯度分布
    """
    with writer.as_default():
        for i in range(20):
            tf.summary.histogram("param_grad", np.log(np.random.random((5, 5))), i)
            tf.summary.histogram("param_param", np.random.random((5, 5)), i)


def test3():
    """
        记录图像
        add_image
        img_tensor这个要注意，表示的我们图像数据，但是要「注意尺度」， 如果我们的图片像素值都是0-1， 那么会默认在这个基础上*255来可视化，毕竟我们的图片都是0-255， 如果像素值有大于1的，那么机器就以为是0-255的范围了，不做任何改动
        global_step: x轴
        dataformats: 数据形式，有 CHW，HWC，HW（灰度图）
    """
    tf.summary.image("random_image", np.random.random((3, 200, 200)), 1)
    tf.summary.image("random_image", np.random.random((3, 200, 200)), 2)
    tf.summary.image("random_image", np.random.random((3, 200, 200)), 3)
    tf.summary.image("random_image", np.random.random((3, 200, 200)), 4)
    tf.summary.image("random_image", np.random.random((3, 200, 200)), 5)


def test4():
    """
    trace_export()：停止trace，并将之前trace记录到的信息写入profiler日志文件。
    trace_off()：停止trace，并舍弃之前trace记录。
    trace_on()：开始trace
    """
    tf.summary.trace_on(graph=True, profiler=True)  # 开启Trace，可以记录图结构和profile信息
    """
    进行训练
    """
    # 最后将统计信息写入日志
    with writer.as_default():
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)  # 保存Trace信息到文件


if __name__ == "__main__":
    test2()



