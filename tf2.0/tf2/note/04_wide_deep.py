from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


"""
    使用两种方式定义 wide&deep模型
"""


# 获取房价预测数据
housing = fetch_california_housing()
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=8)
print(x_train.shape, x_valid.shape, x_test.shape)


# 标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)


# 定义wide&deep模型
input = keras.layers.Input(shape=x_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)

concat = keras.layers.concatenate([input, hidden2])
output = keras.layers.Dense(1)(concat)

# 固化模型
model = keras.models.Model(inputs=[input], outputs=[output])
model.compile(loss="mean_squared_error", optimizer="sgd")


# 第二种定义wide&deep的方法
class WideDeepModel(keras.models.Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()
        """ 定义模型层次 """
        self.hidden1_layer = keras.layers.Dense(30, activation="relu")
        self.hidden2_layer = keras.layers.Dense(30, activation="relu")
        self.output_layer = keras.layers.Dense(1)

    def call(self, input):
        """ 完成模型的正向计算 """
        hidden1 = self.hidden1_layer(input)
        hidden2 = self.hidden2_layer(hidden1)
        concat = keras.layers.concatenate([input, hidden2])
        output = self.output_layer(concat)
        return output


# model = WideDeepModel()  # 或者用下面的方法
model = keras.models.Sequential([
    WideDeepModel()
])

model.build(input_shape=(None, 8))
model.compile(loss="mean_squared_error", optimizer="sgd")


history = model.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_valid_scaled, y_valid))



"""
    多输入单输出
"""
input_wide = keras.layers.Input(shape=(5))
input_deep = keras.layers.Input(shape=(6))
hidden1 = keras.layers.Dense(30, activation="relu")(input_deep)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)

concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)

# 固化模型
model = keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])

model.compile(loss="mean_squared_error", optimizer="sgd")



"""
    多输入多输出
    并不是 wide&deep 所属的，很多情况下都可以用
"""
input_wide = keras.layers.Input(shape=(5))
input_deep = keras.layers.Input(shape=(6))
hidden1 = keras.layers.Dense(30, activation="relu")(input_deep)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)

concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)

# 使用hidden2 单独构建一个输出
output_2 = keras.layers.Dense(1)(hidden2)

model = keras.models.Model(inputs=[input_wide, input_deep], outputs=[output, output_2])
model.compile(loss="mean_squared_error", optimizer="sgd")

# 训练时相应的要做两个输入数据
history = model.fit([x_train_wide, x_train_deep], [y_train, y_train], epochs=10, validation_data=([x_valid_wide, x_valid_deep], [y_valid, y_valid]))
draw_res(history)
# history中 包含总的loss和各自的loss，主要是两个标签y1 y2



























