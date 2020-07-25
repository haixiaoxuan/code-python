import tensorflow as tf






exam = tf.train.Example (
            features=tf.train.Features(
                feature={
                    'name': tf.train.Feature(bytes_list=tf.train.BytesList (value=[splits[-1].encode('utf-8')])),
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List (value=[img.shape[0], img.shape[1], img.shape[2]])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList (value=[bytes(img.numpy())]))
                }
            )
        )






