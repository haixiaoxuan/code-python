import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time

# 禁用GPU
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 下载数据集
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

start = time.time()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 特征缩放[0, 1]区间
train_images, test_images = train_images / 255.0, test_images / 255.0


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.summary() # 显示模型的架构


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary() # 显示模型的架构

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)


test_loss, test_acc = model.evaluate(test_images, test_labels)
print("准确率: " + str(test_acc))

end = time.time()
print("耗时: " + str(end - start))

