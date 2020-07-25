
"""
    https://blog.csdn.net/weixin_42933718/article/details/88245430
"""

def test1():
    import pandas as pd
    import tensorflow as tf

    train_data = pd.read_csv("/home/etluser/xiexiaoxuan/train")
    y_train = train_data.pop('label')

    fc = tf.feature_column

    CATEGORICAL_COLUMNS = ["label"]
    NUMERIC_COLUMNS = ["col_"+str(i) for i in range(784)]

    def one_hot_cat_column(feature_name, vocab):
      # 指示列并不直接操作数据，但它可以把各种分类特征列转化成为input_layer()方法
      # 接受的特征列。
      return fc.indicator_column(
          fc.categorical_column_with_vocabulary_list(feature_name,
                                                     vocab))

    feature_columns = []

    # for feature_name in CATEGORICAL_COLUMNS:
    #   # one-hot
    #   vocabulary = train_data[feature_name].unique()
    #   feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(fc.numeric_column(feature_name, dtype=tf.float32))


    # 样本个数
    NUM_EXAMPLES = len(y_train)

    def make_input_fn(X, y, n_epochs=None, shuffle=True):
      def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
          dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
      return input_fn

    # Training and evaluation input functions.
    # 包装方法，model.train不接受带参数的input_fn
    train_input_fn = make_input_fn(train_data, y_train)
    eval_input_fn = make_input_fn(train_data, y_train, shuffle=False, n_epochs=1)


    # 逻辑回归模型
    # linear_est = tf.estimator.LinearClassifier(feature_columns)
    # # Train model.
    # linear_est.train(train_input_fn, max_steps=100)
    # # Evaluation.
    # results = linear_est.evaluate(eval_input_fn)
    # print('Accuracy : ', results['accuracy'])
    # print('Dummy model: ', results['accuracy_baseline'])



    # Since data fits into memory, use entire dataset per layer. It will be faster.
    # Above one batch is defined as the entire dataset.
    n_batches = 1
    est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                              n_batches_per_layer=n_batches)
    est.train(train_input_fn, max_steps=100)
    results = est.evaluate(eval_input_fn)
    print('Accuracy : ', results['accuracy'])
    print('Dummy model: ', results['accuracy_baseline'])


def test2():

    import tensorflow as tf
    from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
    from tensorflow.contrib.boosted_trees.proto import learner_pb2 as gbdt_learner

    # Ignore all GPUs (current TF GBDT does not support GPU).
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # 设置日志级别
    tf.logging.set_verbosity(tf.logging.ERROR)
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data", one_hot=False,source_url='http://yann.lecun.com/exdb/mnist/')

    # 参数
    batch_size = 4096  # 批次大小
    num_classes = 10  # 标签数
    num_features = 784  # 特征数
    max_steps = 10000   # 最大步数

    # GBDT Parameters
    learning_rate = 0.1
    l1_regul = 0.
    l2_regul = 1.
    examples_per_layer = 1000
    num_trees = 10
    max_depth = 16

    # 设置参数
    learner_config = gbdt_learner.LearnerConfig()
    learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate
    learner_config.regularization.l1 = l1_regul
    learner_config.regularization.l2 = l2_regul / examples_per_layer
    learner_config.constraints.max_tree_depth = max_depth
    growing_mode = gbdt_learner.LearnerConfig.LAYER_BY_LAYER
    learner_config.growing_mode = growing_mode
    run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)
    learner_config.multi_class_strategy = (gbdt_learner.LearnerConfig.DIAGONAL_HESSIAN)

    # 创建模型
    gbdt_model = GradientBoostedDecisionTreeClassifier(
        model_dir=None,  # No save directory specified
        learner_config=learner_config,
        n_classes=num_classes,
        examples_per_layer=examples_per_layer,
        num_trees=num_trees,
        center_bias=False,
        config=run_config)

    # Display TF info logs
    tf.logging.set_verbosity(tf.logging.INFO)

    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images}, y=mnist.train.labels,
        batch_size=batch_size, num_epochs=None, shuffle=True)

    # 训练
    gbdt_model.fit(input_fn=input_fn, max_steps=max_steps)

    # 预测
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.test.images}, y=mnist.test.labels,
        batch_size=batch_size, shuffle=False)
    e = gbdt_model.evaluate(input_fn=input_fn)

    print("Testing Accuracy:", e['accuracy'])

if __name__ == "__main__":
    # test2()
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
    print(mnist.train.images.shape)
    print(mnist.test.labels.shape)





