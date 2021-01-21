import tensorflow as tf


"""
    并不是任何函数都可以被 @tf.function 修饰！@tf.function 使用静态编译将函数内的代码转换成计算图，
    因此对函数内可使用的语句有一定限制（仅支持 Python 语言的一个子集），且需要函数内的操作本身能够被构建为计算图。
    建议在函数内只使用 TensorFlow 的原生操作，不要使用过于复杂的 Python 语句


    tf.function 使用的两种方式：
        1. 函数
        2. 注解    
    
    note:
        tf.variable 需要放在函数外面进行初始化
"""


def scaled_elu(z, scale=1.0, alpha=1.0):
    # z>=0?scale*z:scale*alpha*tf.nn.elu(z)
    is_positive = tf.greater_equal(z, 0.0)
    return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))


print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant([-3, -2.5])))

# 使用函数进行转换
scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3., -2.5])))
print(scaled_elu_tf.python_function is scaled_elu_tf)


def display_tf_code(func):
    """ 可以将python函数转换为 tf 代码 """
    code = tf.autograph.to_code(func)
    return code


# 增加类型限定，指定传参类型
@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name="x")])
def cube(z):
    # 注意这里使用了TensorFlow内置的tf.print()。@tf.function不支持Python内置的print方法
    tf.print("param", z)
    return tf.pow(z, 3)


print(cube(tf.constant([2, 3])))
print(cube(tf.constant([3])))
# print(cube(tf.constant[2.0]))
print(cube([2]))


# 将 tf.function 标注的函数转变为可导出的 save_model 函数， 需要使用 input_signature 限定
cube_concrete = cube.get_concrete_function(tf.TensorSpec([None], tf.int32))
print(cube_concrete)
print(cube_concrete.graph)
print(cube_concrete.graph.get_operations())
print(cube_concrete.graph.get_operation_by_name("x"))
print(cube_concrete.graph.get_tensor_by_name("x:0"))
print(cube_concrete.graph.as_graph_def())           # 打印图结构



"""
    tensorflow 动态数组
        在部分网络结构，尤其是涉及到时间序列的结构中，我们可能需要将一系列张量以数组的方式依次存放起来，以供进一步处理。
        当然，在即时执行模式下，你可以直接使用一个 Python 列表（List）存放数组。不过，如果你需要基于计算图的特性
        （例如使用 @tf.function 加速模型运行或者使用 SavedModel 导出模型），就无法使用这种方式了。
        因此，TensorFlow 提供了 tf.TensorArray ，一种支持计算图特性的 TensorFlow 动态数组。
        
        arr = tf.TensorArray(dtype, size, dynamic_size=False) ：声明一个大小为 size ，类型为 dtype 的 TensorArray arr 。
                如果将 dynamic_size 参数设置为 True ，则该数组会自动增长空间。
        其读取和写入的方法为：
            write(index, value) ：将 value 写入数组的第 index 个位置；
            read(index) ：读取数组的第 index 个值；
        除此以外，TensorArray 还包括 stack() 、 unstack() 等常用操作
        
        note: 
            arr = arr.write(index, value)       必须使用变量接受
"""





