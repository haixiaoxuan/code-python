import tensorflow as tf
import os

# 使用mnist 数据集构建 tfrecord mnist 数据集

def create_tfrecords(inpath, outpath):
    """ 构建 tfrecord """
    writer = tf.io.TFRecordWriter(outpath)

    with open(inpath, "r") as f:
        for line in f:
            line_arr = [float(i) for i in line.split(",")]
            features = line_arr[0: -1]
            label = int(line_arr[-1])

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        "features": tf.train.Feature(float_list=tf.train.FloatList(value=features))
                    }
                )
            )
            writer.write(example.SerializeToString())
    writer.close()


def load_data(data_path):
    """ 加载 tfrecord """
    dataset = tf.data.TFRecordDataset([data_path])

    # 将序列化的进行反序列化
    expected_features = {
        "features": tf.io.FixedLenFeature(784, dtype=tf.float32),
        "label": tf.io.FixedLenFeature([], dtype=tf.int64),
    }

    for serialized_example_tensor in dataset:
        example = tf.io.parse_single_example(serialized_example_tensor, expected_features)
        print(example["label"])
        print(example["features"])
        print("==========================")


if __name__ == "__main__":
    data_path = r"C:\Users\xiaoxuan\Desktop\work\tensorflow\data\mnist.csv"
    out_path = r"C:\Users\xiaoxuan\Desktop\work\tensorflow\data\mnist.tfrecord"
    import time
    start = time.time()
    # create_tfrecords(data_path, out_path)
    end = time.time()
    print("耗时：", end - start)

    load_data(out_path)

