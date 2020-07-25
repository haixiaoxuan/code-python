import tensorflow as tf

"""
    使用 @tf.function
    需要将我们希望以 Graph Execution 模式运行的代码封装在一个函数内，并在函数前加上 @tf.function 即可
    
    当被 @tf.function 修饰的函数第一次被调用的时候，进行以下操作：
        在 Eager Execution 模式关闭的环境下，函数内的代码依次运行。也就是说，每个 tf. 方法都只是定义了计算节点，而并没有进行任何实质的计算。这与 TensorFlow 1.X 的 Graph Execution 是一致的；
        使用 AutoGraph 将函数中的 Python 控制流语句转换成 TensorFlow 计算图中的对应节点（比如说 while 和 for 语句转换为 tf.while ， if 语句转换为 tf.cond 等等；
        基于上面的两步，建立函数内代码的计算图表示（为了保证图的计算顺序，图中还会自动加入一些 tf.control_dependencies 节点）；
        运行一次这个计算图；
        基于函数的名字和输入的函数参数的类型生成一个哈希值，并将建立的计算图缓存到一个哈希表中。
"""

@tf.function
def train_one_step(X, y):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        # 注意这里使用了TensorFlow内置的tf.print()。@tf.function不支持Python内置的print方法
        tf.print("loss", loss)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


































